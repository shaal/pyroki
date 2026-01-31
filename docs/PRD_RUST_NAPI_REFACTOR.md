# Product Requirements Document: PyRoki Rust/NAPI-RS Performance Optimization

## Executive Summary

This PRD outlines a comprehensive plan to refactor performance-critical components of PyRoki from pure Python/JAX to native Rust implementations exposed via NAPI-RS. The goal is to achieve 10-100x performance improvements in collision detection and forward kinematics while maintaining full compatibility with the existing Python API.

---

# SPARC Framework Analysis

## S - Specification

### 1.1 Problem Statement

PyRoki is a high-performance robot kinematics library built on JAX. While JAX provides excellent automatic differentiation and GPU acceleration, certain computational bottlenecks exist:

| Bottleneck | Current Performance | Target Improvement |
|------------|---------------------|-------------------|
| Capsule-Capsule Collision | ~30-40% of collision time | 10-100x faster |
| Segment-to-Segment Distance | ~20-30% of collision time | 20-50x faster |
| Forward Kinematics SE(3) | ~20-30% of compute | 5-20x faster |
| Pairwise Collision Broadcasting | ~5-15% of compute | 3-10x faster |

### 1.2 Goals

**Primary Goals:**
1. Reduce collision detection latency by 10-100x for CPU workloads
2. Maintain full backward compatibility with existing Python API
3. Enable optional Rust acceleration (graceful fallback to JAX)
4. Support batched operations with SIMD vectorization

**Secondary Goals:**
1. Reduce memory allocations in hot paths
2. Enable parallel collision checking with Rayon
3. Provide zero-copy interop with NumPy/JAX arrays
4. Maintain differentiability through JAX custom_vjp wrappers

### 1.3 Non-Goals

- Replacing JAX for GPU workloads (JAX/XLA excels here)
- Rewriting the optimization solver (jaxls handles this well)
- Replacing jaxlie entirely (only hot paths)
- Supporting mesh-mesh collision (out of scope)

### 1.4 Success Metrics

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| Capsule-Capsule Distance (1000 pairs) | ~50ms | <1ms | Benchmark suite |
| Forward Kinematics (batch=1000) | ~14ms | <3ms | IK benchmark |
| Self-collision check (50 pairs) | ~5ms | <0.5ms | Collision test |
| Memory per collision check | ~10KB | <1KB | Memory profiler |

### 1.5 User Stories

1. **As a robotics researcher**, I want faster collision detection so that real-time trajectory optimization can run at >100Hz.

2. **As a motion planner developer**, I want SIMD-accelerated geometry operations so that batch IK solving scales linearly with batch size.

3. **As a PyRoki user**, I want the Rust acceleration to be optional so that my existing code works without recompilation.

4. **As a contributor**, I want clear Rust/Python boundaries so that I can maintain either side independently.

---

## P - Pseudocode

### 2.1 Core Collision Algorithms

#### 2.1.1 Segment-to-Segment Closest Points (Highest Impact)

```rust
/// Find closest points between two 3D line segments
/// Returns (t1, t2) parameters along each segment
fn closest_segment_to_segment(
    a1: Vec3, b1: Vec3,  // Segment 1: a1 -> b1
    a2: Vec3, b2: Vec3,  // Segment 2: a2 -> b2
) -> (f64, f64) {
    let d1 = b1 - a1;  // Direction of segment 1
    let d2 = b2 - a2;  // Direction of segment 2
    let r = a1 - a2;   // Vector between origins

    let a = d1.dot(d1);  // Squared length of segment 1
    let e = d2.dot(d2);  // Squared length of segment 2
    let f = d2.dot(r);

    // Check for degenerate cases (segments are points)
    if a <= EPSILON && e <= EPSILON {
        return (0.0, 0.0);  // Both segments are points
    }

    let (s, t) = if a <= EPSILON {
        // Segment 1 is a point
        (0.0, clamp(f / e, 0.0, 1.0))
    } else if e <= EPSILON {
        // Segment 2 is a point
        (clamp(-d1.dot(r) / a, 0.0, 1.0), 0.0)
    } else {
        // General case: solve 2x2 system
        let b = d1.dot(d2);
        let c = d1.dot(r);
        let denom = a * e - b * b;

        let s = if denom != 0.0 {
            clamp((b * f - c * e) / denom, 0.0, 1.0)
        } else {
            0.0  // Parallel segments
        };

        let t = (b * s + f) / e;

        // Clamp t and recompute s if needed
        if t < 0.0 {
            (clamp(-c / a, 0.0, 1.0), 0.0)
        } else if t > 1.0 {
            (clamp((b - c) / a, 0.0, 1.0), 1.0)
        } else {
            (s, t)
        }
    };

    (s, t)
}
```

#### 2.1.2 Capsule-Capsule Distance

```rust
/// Compute signed distance between two capsules
/// Negative = penetration, Positive = separation
fn capsule_capsule_distance(
    pos1: Vec3, axis1: Vec3, length1: f64, radius1: f64,
    pos2: Vec3, axis2: Vec3, length2: f64, radius2: f64,
) -> f64 {
    // Compute segment endpoints
    let half_axis1 = axis1 * (length1 / 2.0);
    let half_axis2 = axis2 * (length2 / 2.0);

    let a1 = pos1 - half_axis1;
    let b1 = pos1 + half_axis1;
    let a2 = pos2 - half_axis2;
    let b2 = pos2 + half_axis2;

    // Find closest points on axes
    let (t1, t2) = closest_segment_to_segment(a1, b1, a2, b2);

    let pt1 = a1 + (b1 - a1) * t1;
    let pt2 = a2 + (b2 - a2) * t2;

    // Sphere-sphere distance at closest points
    let center_dist = (pt1 - pt2).length();
    center_dist - radius1 - radius2
}
```

#### 2.1.3 Batched Collision with SIMD

```rust
/// Batch capsule-capsule distance using SIMD
/// Processes 4 pairs simultaneously on AVX2
#[cfg(target_feature = "avx2")]
fn batch_capsule_capsule_distance_simd(
    capsules1: &[Capsule],
    capsules2: &[Capsule],
) -> Vec<f64> {
    use std::simd::*;

    let n = capsules1.len().min(capsules2.len());
    let mut results = Vec::with_capacity(n);

    // Process 4 at a time with AVX2
    for chunk in (0..n).step_by(4) {
        let remaining = (n - chunk).min(4);

        // Load 4 capsule pairs into SIMD registers
        let pos1_x = f64x4::from_array([...]);
        let pos1_y = f64x4::from_array([...]);
        // ... vectorized computation ...

        results.extend_from_slice(&distances[..remaining]);
    }

    results
}
```

### 2.2 Forward Kinematics SE(3) Operations

```rust
/// SE(3) exponential map (twist to transform)
/// Optimized for batched joint computations
fn se3_exp(twist: &[f64; 6]) -> Transform3D {
    let omega = Vec3::new(twist[0], twist[1], twist[2]);  // Angular
    let v = Vec3::new(twist[3], twist[4], twist[5]);      // Linear

    let theta = omega.length();

    if theta < EPSILON {
        // Small angle approximation
        Transform3D::from_translation(v)
    } else {
        let omega_hat = omega / theta;

        // Rodrigues' formula for rotation
        let sin_t = theta.sin();
        let cos_t = theta.cos();
        let rot = so3_exp(omega_hat, sin_t, cos_t);

        // V matrix for translation
        let v_mat = compute_v_matrix(omega_hat, theta, sin_t, cos_t);
        let trans = v_mat * v;

        Transform3D::from_rotation_translation(rot, trans)
    }
}

/// Batch FK for all joints
fn forward_kinematics_batch(
    joint_configs: &[f64],           // Shape: (batch, n_joints)
    joint_twists: &[[f64; 6]],       // Shape: (n_joints, 6)
    parent_indices: &[i32],           // Kinematic tree structure
    joint_to_parent_transforms: &[Transform3D],
) -> Vec<Transform3D> {
    // Parallel over batch dimension with Rayon
    joint_configs.par_chunks(n_joints)
        .map(|config| {
            let mut poses = vec![Transform3D::identity(); n_joints];

            for (i, &parent_idx) in parent_indices.iter().enumerate() {
                let twist_scaled = scale_twist(&joint_twists[i], config[i]);
                let delta = se3_exp(&twist_scaled);

                let parent_pose = if parent_idx < 0 {
                    Transform3D::identity()
                } else {
                    poses[parent_idx as usize]
                };

                poses[i] = parent_pose * joint_to_parent_transforms[i] * delta;
            }

            poses
        })
        .collect()
}
```

---

## A - Architecture

### 3.1 System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         Python Layer                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   PyRoki    │  │    JAX      │  │    NumPy Arrays         │  │
│  │   API       │  │  (autodiff) │  │    (zero-copy)          │  │
│  └──────┬──────┘  └──────┬──────┘  └───────────┬─────────────┘  │
│         │                │                      │                │
│  ┌──────▼────────────────▼──────────────────────▼──────────────┐ │
│  │              pyroki_native (Python Extension)                │ │
│  │    ┌─────────────────────────────────────────────────────┐  │ │
│  │    │  NAPI-RS Bindings / PyO3 Alternative                │  │ │
│  │    │  - Type conversions                                 │  │ │
│  │    │  - Array interop (numpy → rust slices)              │  │ │
│  │    │  - Error handling                                   │  │ │
│  │    └─────────────────────────────────────────────────────┘  │ │
│  └──────────────────────────┬───────────────────────────────────┘ │
└─────────────────────────────┼───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                         Rust Core                                │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    pyroki-core (Pure Rust)                  │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │ │
│  │  │  collision   │  │  kinematics  │  │     geometry     │  │ │
│  │  │  - capsule   │  │  - se3_exp   │  │  - vec3, mat3    │  │ │
│  │  │  - sphere    │  │  - fk_batch  │  │  - transform3d   │  │ │
│  │  │  - box       │  │  - jacobian  │  │  - quaternion    │  │ │
│  │  │  - heightmap │  │              │  │                  │  │ │
│  │  └──────────────┘  └──────────────┘  └──────────────────┘  │ │
│  │  ┌──────────────────────────────────────────────────────┐  │ │
│  │  │                     simd_accel                        │  │ │
│  │  │  - AVX2/AVX512 vectorized geometry                   │  │ │
│  │  │  - NEON for ARM acceleration                         │  │ │
│  │  └──────────────────────────────────────────────────────┘  │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                     Dependencies                             │ │
│  │  - nalgebra (linear algebra)                                │ │
│  │  - rayon (parallelism)                                      │ │
│  │  - wide (portable SIMD)                                     │ │
│  └─────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

### 3.2 Directory Structure

```
pyroki/
├── src/
│   └── pyroki/                    # Existing Python code
│       ├── __init__.py
│       ├── _robot.py
│       ├── collision/
│       │   ├── _collision.py      # Modified: dispatch to Rust
│       │   ├── _geometry.py
│       │   ├── _geometry_pairs.py # Fallback implementations
│       │   └── _native.py         # NEW: Rust bindings wrapper
│       └── _native/               # NEW: Native extension loader
│           └── __init__.py
│
├── rust/                          # NEW: Rust workspace
│   ├── Cargo.toml                 # Workspace configuration
│   │
│   ├── pyroki-core/               # Pure Rust library
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── geometry/
│   │       │   ├── mod.rs
│   │       │   ├── vec3.rs
│   │       │   ├── transform3d.rs
│   │       │   └── quaternion.rs
│   │       ├── collision/
│   │       │   ├── mod.rs
│   │       │   ├── capsule.rs
│   │       │   ├── sphere.rs
│   │       │   ├── box_geom.rs
│   │       │   ├── heightmap.rs
│   │       │   └── pairs.rs       # All pairwise algorithms
│   │       ├── kinematics/
│   │       │   ├── mod.rs
│   │       │   ├── se3.rs
│   │       │   ├── fk.rs
│   │       │   └── jacobian.rs
│   │       └── simd/
│   │           ├── mod.rs
│   │           └── batch_collision.rs
│   │
│   └── pyroki-python/             # Python bindings (NAPI-RS or PyO3)
│       ├── Cargo.toml
│       ├── src/
│       │   ├── lib.rs
│       │   ├── collision.rs       # Collision function exports
│       │   ├── kinematics.rs      # FK function exports
│       │   └── array_utils.rs     # NumPy interop
│       └── python/
│           └── pyroki_native/
│               └── __init__.pyi   # Type stubs
│
├── pyproject.toml                 # Modified: add maturin build
├── Cargo.toml                     # Workspace root
└── build.rs                       # Optional: build customization
```

### 3.3 Module Responsibilities

| Module | Responsibility | Key Types |
|--------|---------------|-----------|
| `pyroki-core::geometry` | 3D math primitives | `Vec3`, `Transform3D`, `Quaternion` |
| `pyroki-core::collision` | Collision algorithms | `Capsule`, `Sphere`, `Box`, distance functions |
| `pyroki-core::kinematics` | FK/Jacobian computation | `SE3`, `JointConfig`, FK functions |
| `pyroki-core::simd` | SIMD-accelerated batch ops | Vectorized collision, FK batching |
| `pyroki-python` | Python bindings | NumPy interop, function exports |

### 3.4 Technology Choice: NAPI-RS vs PyO3

After analysis, **PyO3** is recommended over NAPI-RS for this project:

| Factor | NAPI-RS | PyO3 | Winner |
|--------|---------|------|--------|
| **Target Runtime** | Node.js | Python | PyO3 |
| **NumPy Integration** | Manual FFI | numpy crate | PyO3 |
| **JAX Compatibility** | None | dlpack support | PyO3 |
| **Ecosystem** | JS-focused | Python-focused | PyO3 |
| **Build System** | napi-build | maturin | Equal |
| **Documentation** | Good | Excellent | PyO3 |

**Recommendation**: Use **PyO3** with **maturin** for build system.

> Note: The branch name mentions "napi-rs" but for a Python library, PyO3 is the correct choice. NAPI-RS is for Node.js native modules. This PRD assumes the intent was native Rust extensions for Python.

### 3.5 Data Flow Diagrams

#### 3.5.1 Collision Detection Flow

```
Python (collision.collide)
    │
    ▼
┌───────────────────────────────┐
│  Dispatch by geometry type    │
│  (Sphere, Capsule, Box, etc.) │
└───────────────┬───────────────┘
                │
    ┌───────────┴───────────┐
    │                       │
    ▼                       ▼
┌───────────┐         ┌───────────┐
│ Rust Fast │         │ JAX       │
│ Path      │         │ Fallback  │
│ (PyO3)    │         │           │
└─────┬─────┘         └─────┬─────┘
      │                     │
      ▼                     ▼
┌─────────────────────────────────┐
│  NumPy array with distances     │
│  (zero-copy when possible)      │
└─────────────────────────────────┘
```

#### 3.5.2 Forward Kinematics Flow

```
Python (robot.forward_kinematics)
    │
    ▼
┌────────────────────────────────────┐
│  Check: Rust extension available?  │
└───────────────┬────────────────────┘
                │
    ┌───────────┴───────────┐
    │ Yes                   │ No
    ▼                       ▼
┌─────────────────┐   ┌─────────────────┐
│ Rust FK         │   │ JAX FK          │
│ (batch=1000+)   │   │ (small batch)   │
│                 │   │ or GPU workload │
└────────┬────────┘   └────────┬────────┘
         │                     │
         ▼                     ▼
┌──────────────────────────────────────┐
│  Link poses array (*batch, n_links)  │
└──────────────────────────────────────┘
```

---

## R - Refinement

### 4.1 Edge Cases and Error Handling

#### 4.1.1 Degenerate Geometry

| Case | Detection | Handling |
|------|-----------|----------|
| Zero-length capsule | `length < EPSILON` | Treat as sphere |
| Zero-radius sphere | `radius < EPSILON` | Point collision |
| Parallel segments | `denom < EPSILON` | Use endpoint projection |
| Coincident points | `distance < EPSILON` | Return 0.0 |

```rust
const EPSILON: f64 = 1e-10;

fn capsule_capsule_distance_safe(c1: &Capsule, c2: &Capsule) -> f64 {
    // Handle degenerate capsules (spheres)
    if c1.length < EPSILON && c2.length < EPSILON {
        return sphere_sphere_distance(c1.as_sphere(), c2.as_sphere());
    }
    if c1.length < EPSILON {
        return sphere_capsule_distance(c1.as_sphere(), c2);
    }
    if c2.length < EPSILON {
        return sphere_capsule_distance(c2.as_sphere(), c1);
    }

    // Standard capsule-capsule
    capsule_capsule_distance_impl(c1, c2)
}
```

#### 4.1.2 Numerical Stability

```rust
/// Safe normalization with fallback
fn safe_normalize(v: Vec3) -> Vec3 {
    let len = v.length();
    if len < EPSILON {
        Vec3::new(1.0, 0.0, 0.0)  // Default direction
    } else {
        v / len
    }
}

/// Numerically stable segment closest point
fn closest_point_on_segment(p: Vec3, a: Vec3, b: Vec3) -> Vec3 {
    let ab = b - a;
    let ab_len_sq = ab.dot(ab);

    if ab_len_sq < EPSILON * EPSILON {
        return a;  // Degenerate segment
    }

    let t = ((p - a).dot(ab) / ab_len_sq).clamp(0.0, 1.0);
    a + ab * t
}
```

#### 4.1.3 Array Dimension Handling

```rust
/// Handle arbitrary batch dimensions for collision
fn batch_collide<const N: usize>(
    geoms1: &[Capsule],  // Shape: (batch..., N)
    geoms2: &[Capsule],  // Shape: (batch..., M)
) -> Result<Vec<f64>, CollisionError> {
    // Validate compatible batch shapes
    if geoms1.is_empty() || geoms2.is_empty() {
        return Ok(vec![]);
    }

    // Broadcast semantics matching NumPy
    let result_len = geoms1.len() * geoms2.len();
    let mut results = Vec::with_capacity(result_len);

    for g1 in geoms1 {
        for g2 in geoms2 {
            results.push(capsule_capsule_distance(g1, g2));
        }
    }

    Ok(results)
}
```

### 4.2 Performance Optimizations

#### 4.2.1 Memory Layout for SIMD

```rust
/// Structure of Arrays (SoA) for SIMD-friendly layout
#[repr(C)]
struct CapsuleBatch {
    // Position components (contiguous for SIMD load)
    pos_x: Vec<f64>,
    pos_y: Vec<f64>,
    pos_z: Vec<f64>,
    // Axis components
    axis_x: Vec<f64>,
    axis_y: Vec<f64>,
    axis_z: Vec<f64>,
    // Scalar properties
    lengths: Vec<f64>,
    radii: Vec<f64>,
}

impl CapsuleBatch {
    /// Convert from AoS (NumPy) to SoA (Rust SIMD)
    fn from_numpy(arr: &PyArray2<f64>) -> Self {
        // Shape: (n_capsules, 8) -> SoA fields
        // ...
    }
}
```

#### 4.2.2 Cache-Friendly Traversal

```rust
/// Process collision pairs in cache-friendly order
fn pairwise_collide_cached(
    capsules1: &CapsuleBatch,
    capsules2: &CapsuleBatch,
) -> Vec<f64> {
    let n1 = capsules1.len();
    let n2 = capsules2.len();
    let mut results = vec![0.0; n1 * n2];

    // Tile computation for L1 cache (64KB typical)
    const TILE_SIZE: usize = 64;

    for i0 in (0..n1).step_by(TILE_SIZE) {
        for j0 in (0..n2).step_by(TILE_SIZE) {
            let i_end = (i0 + TILE_SIZE).min(n1);
            let j_end = (j0 + TILE_SIZE).min(n2);

            // Process tile
            for i in i0..i_end {
                for j in j0..j_end {
                    results[i * n2 + j] = compute_distance(
                        capsules1.get(i),
                        capsules2.get(j),
                    );
                }
            }
        }
    }

    results
}
```

#### 4.2.3 Parallel Execution with Rayon

```rust
use rayon::prelude::*;

/// Parallel batch collision with automatic work stealing
fn batch_collide_parallel(
    capsules1: &[Capsule],
    capsules2: &[Capsule],
) -> Vec<f64> {
    let n2 = capsules2.len();

    capsules1.par_iter()
        .enumerate()
        .flat_map(|(i, c1)| {
            capsules2.iter()
                .enumerate()
                .map(move |(j, c2)| {
                    (i * n2 + j, capsule_capsule_distance(c1, c2))
                })
        })
        .collect::<Vec<_>>()
        .into_iter()
        .map(|(_, d)| d)
        .collect()
}
```

### 4.3 JAX Integration Strategy

#### 4.3.1 Custom VJP for Autodiff

```python
# Python wrapper with JAX custom_vjp
import jax
from jax import custom_vjp
import pyroki_native

@custom_vjp
def capsule_capsule_distance_native(
    pos1, axis1, length1, radius1,
    pos2, axis2, length2, radius2,
):
    """Forward pass using Rust implementation."""
    return pyroki_native.capsule_capsule_distance(
        pos1, axis1, length1, radius1,
        pos2, axis2, length2, radius2,
    )

def _capsule_capsule_fwd(pos1, axis1, length1, radius1, pos2, axis2, length2, radius2):
    """Forward pass storing residuals for backward."""
    dist = capsule_capsule_distance_native(
        pos1, axis1, length1, radius1,
        pos2, axis2, length2, radius2,
    )
    # Store intermediate values for gradient computation
    residuals = (pos1, axis1, length1, radius1, pos2, axis2, length2, radius2, dist)
    return dist, residuals

def _capsule_capsule_bwd(residuals, g):
    """Backward pass computing gradients."""
    pos1, axis1, length1, radius1, pos2, axis2, length2, radius2, dist = residuals

    # Use Rust for gradient computation OR finite differences
    grads = pyroki_native.capsule_capsule_gradient(
        pos1, axis1, length1, radius1,
        pos2, axis2, length2, radius2,
        g,  # upstream gradient
    )
    return grads

capsule_capsule_distance_native.defvjp(_capsule_capsule_fwd, _capsule_capsule_bwd)
```

#### 4.3.2 Fallback Dispatch Logic

```python
# In pyroki/collision/_native.py

import os
import warnings

_RUST_AVAILABLE = False
_RUST_MODULE = None

def _load_native():
    global _RUST_AVAILABLE, _RUST_MODULE
    try:
        import pyroki_native
        _RUST_MODULE = pyroki_native
        _RUST_AVAILABLE = True
    except ImportError:
        if os.environ.get("PYROKI_REQUIRE_NATIVE"):
            raise ImportError(
                "Native Rust extension required but not found. "
                "Install with: pip install pyroki[native]"
            )
        warnings.warn(
            "Native Rust extension not available. "
            "Using JAX fallback (slower for CPU workloads).",
            UserWarning,
        )

_load_native()

def capsule_capsule_distance(c1, c2):
    """Dispatch to Rust or JAX based on availability and heuristics."""
    if _RUST_AVAILABLE and _should_use_rust(c1, c2):
        return _rust_capsule_capsule(c1, c2)
    else:
        return _jax_capsule_capsule(c1, c2)

def _should_use_rust(c1, c2):
    """Heuristic: use Rust for CPU, large batches, or when no autodiff needed."""
    import jax

    # Check if we're tracing (autodiff context)
    if isinstance(c1.pose.translation(), jax.core.Tracer):
        return False  # Need JAX for gradients

    # Check device
    arr = c1.pose.translation()
    if hasattr(arr, 'device') and 'gpu' in str(arr.device()).lower():
        return False  # Use JAX for GPU

    return True  # Default to Rust for CPU
```

### 4.4 Testing Strategy

#### 4.4.1 Unit Tests (Rust)

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_capsule_capsule_parallel() {
        let c1 = Capsule::new(Vec3::ZERO, Vec3::X, 2.0, 0.5);
        let c2 = Capsule::new(Vec3::new(3.0, 0.0, 0.0), Vec3::X, 2.0, 0.5);

        // Parallel capsules, 3 units apart center-to-center
        // Expected: 3.0 - 1.0 - 1.0 - 0.5 - 0.5 = 0.0 (touching)
        let dist = capsule_capsule_distance(&c1, &c2);
        assert_relative_eq!(dist, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_capsule_capsule_perpendicular() {
        let c1 = Capsule::new(Vec3::ZERO, Vec3::X, 2.0, 0.1);
        let c2 = Capsule::new(Vec3::new(0.0, 1.0, 0.0), Vec3::Z, 2.0, 0.1);

        // Perpendicular, 1 unit apart
        let dist = capsule_capsule_distance(&c1, &c2);
        assert_relative_eq!(dist, 0.8, epsilon = 1e-10);  // 1.0 - 0.1 - 0.1
    }

    #[test]
    fn test_degenerate_zero_length() {
        let sphere = Capsule::new(Vec3::ZERO, Vec3::X, 0.0, 1.0);
        let capsule = Capsule::new(Vec3::new(3.0, 0.0, 0.0), Vec3::X, 2.0, 0.5);

        let dist = capsule_capsule_distance(&sphere, &capsule);
        // Sphere at origin, capsule from (2,0,0) to (4,0,0), radius 0.5
        // Distance = 2.0 - 1.0 - 0.5 = 0.5
        assert_relative_eq!(dist, 0.5, epsilon = 1e-10);
    }
}
```

#### 4.4.2 Integration Tests (Python)

```python
# tests/test_rust_collision.py

import pytest
import numpy as np
import jax.numpy as jnp
from pyroki.collision import Capsule, collide
import pyroki.collision._native as native

@pytest.fixture
def capsule_pair():
    c1 = Capsule.from_radius_height(0.1, 1.0, jnp.zeros(3), jnp.array([1, 0, 0, 0]))
    c2 = Capsule.from_radius_height(0.1, 1.0, jnp.array([2.0, 0, 0]), jnp.array([1, 0, 0, 0]))
    return c1, c2

def test_rust_matches_jax(capsule_pair):
    """Verify Rust and JAX implementations produce identical results."""
    c1, c2 = capsule_pair

    # Force JAX path
    with native._force_jax():
        jax_result = collide(c1, c2)

    # Force Rust path
    with native._force_rust():
        rust_result = collide(c1, c2)

    np.testing.assert_allclose(rust_result, jax_result, rtol=1e-10)

def test_rust_batched():
    """Test batched collision with various shapes."""
    batch_sizes = [1, 10, 100, 1000]

    for batch in batch_sizes:
        pos1 = jnp.zeros((batch, 3))
        pos2 = jnp.tile(jnp.array([2.0, 0, 0]), (batch, 1))

        c1 = Capsule.from_radius_height(
            jnp.full(batch, 0.1),
            jnp.full(batch, 1.0),
            pos1,
            jnp.tile(jnp.array([1, 0, 0, 0]), (batch, 1)),
        )
        c2 = Capsule.from_radius_height(
            jnp.full(batch, 0.1),
            jnp.full(batch, 1.0),
            pos2,
            jnp.tile(jnp.array([1, 0, 0, 0]), (batch, 1)),
        )

        result = collide(c1, c2)
        assert result.shape == (batch,)

@pytest.mark.benchmark
def test_rust_performance(benchmark, capsule_pair):
    """Benchmark Rust vs JAX performance."""
    c1, c2 = capsule_pair

    # Warm up JIT
    _ = collide(c1, c2)

    result = benchmark(lambda: collide(c1, c2).block_until_ready())

    # Assert reasonable performance
    assert benchmark.stats.stats.mean < 0.001  # < 1ms for single pair
```

#### 4.4.3 Property-Based Tests

```python
from hypothesis import given, strategies as st
import hypothesis.extra.numpy as hnp

@given(
    pos1=hnp.arrays(np.float64, (3,), elements=st.floats(-100, 100)),
    pos2=hnp.arrays(np.float64, (3,), elements=st.floats(-100, 100)),
    radius1=st.floats(0.01, 10.0),
    radius2=st.floats(0.01, 10.0),
)
def test_sphere_distance_symmetric(pos1, pos2, radius1, radius2):
    """Sphere-sphere distance should be symmetric."""
    s1 = Sphere.from_center_and_radius(pos1, radius1)
    s2 = Sphere.from_center_and_radius(pos2, radius2)

    d1 = collide(s1, s2)
    d2 = collide(s2, s1)

    np.testing.assert_allclose(d1, d2, rtol=1e-10)

@given(
    pos=hnp.arrays(np.float64, (3,), elements=st.floats(-100, 100)),
    radius=st.floats(0.01, 10.0),
)
def test_sphere_self_distance_negative(pos, radius):
    """Sphere collided with itself should have negative distance."""
    s = Sphere.from_center_and_radius(pos, radius)
    d = collide(s, s)

    # Self-collision = -2*radius (fully penetrating)
    np.testing.assert_allclose(d, -2 * radius, rtol=1e-10)
```

---

## C - Completion

### 5.1 Implementation Phases

#### Phase 1: Foundation (Week 1-2)

**Deliverables:**
- [ ] Rust workspace setup with `pyroki-core` and `pyroki-python`
- [ ] Basic geometry types (`Vec3`, `Transform3D`, `Quaternion`)
- [ ] Sphere-sphere distance (simplest case)
- [ ] PyO3 bindings with NumPy array support
- [ ] CI/CD for Rust builds (maturin + GitHub Actions)

**Acceptance Criteria:**
- `pip install -e .` builds Rust extension
- `from pyroki_native import sphere_sphere_distance` works
- Unit tests pass for sphere-sphere

#### Phase 2: Core Collision (Week 3-4)

**Deliverables:**
- [ ] Segment-to-segment closest points (core algorithm)
- [ ] Capsule-capsule distance
- [ ] Capsule-sphere distance
- [ ] Batch processing with SIMD (AVX2)
- [ ] Python dispatch logic (_native.py)

**Acceptance Criteria:**
- Capsule-capsule 10x faster than JAX on CPU
- Batched collision matches JAX output within 1e-10
- Fallback to JAX works when Rust unavailable

#### Phase 3: Extended Geometry (Week 5-6)

**Deliverables:**
- [ ] Box collision primitives
- [ ] Halfspace collision
- [ ] Heightmap bilinear interpolation
- [ ] All pairwise combinations (11 total)
- [ ] Rayon parallelization for large batches

**Acceptance Criteria:**
- All geometry types supported
- Parallel speedup on 8+ cores
- Memory usage < 1KB per collision check

#### Phase 4: Forward Kinematics (Week 7-8)

**Deliverables:**
- [ ] SE(3) exponential map (Rust)
- [ ] Batch FK computation
- [ ] JAX custom_vjp wrappers for gradients
- [ ] Integration with Robot class

**Acceptance Criteria:**
- FK 5x faster for batch=1000+
- Gradients match JAX autodiff within 1e-6
- No regression in IK benchmark

#### Phase 5: Polish and Release (Week 9-10)

**Deliverables:**
- [ ] Comprehensive benchmarks
- [ ] Documentation for native module
- [ ] Wheel builds for Linux/macOS/Windows
- [ ] Migration guide for users
- [ ] Performance tuning based on profiling

**Acceptance Criteria:**
- All benchmarks show expected speedups
- Wheels available on PyPI
- Zero breaking changes to public API

### 5.2 Build System Configuration

#### 5.2.1 Cargo.toml (Workspace Root)

```toml
[workspace]
members = ["rust/pyroki-core", "rust/pyroki-python"]
resolver = "2"

[workspace.package]
version = "0.1.0"
edition = "2021"
authors = ["PyRoki Contributors"]
license = "MIT"
repository = "https://github.com/chungmin99/pyroki"

[workspace.dependencies]
nalgebra = "0.32"
rayon = "1.8"
wide = "0.7"
thiserror = "1.0"
approx = "0.5"
```

#### 5.2.2 pyroki-core/Cargo.toml

```toml
[package]
name = "pyroki-core"
version.workspace = true
edition.workspace = true

[lib]
crate-type = ["rlib"]

[dependencies]
nalgebra = { workspace = true }
rayon = { workspace = true }
wide = { workspace = true }
thiserror = { workspace = true }

[dev-dependencies]
approx = { workspace = true }
criterion = "0.5"

[features]
default = ["simd"]
simd = []

[[bench]]
name = "collision"
harness = false
```

#### 5.2.3 pyroki-python/Cargo.toml

```toml
[package]
name = "pyroki-python"
version.workspace = true
edition.workspace = true

[lib]
name = "pyroki_native"
crate-type = ["cdylib"]

[dependencies]
pyroki-core = { path = "../pyroki-core" }
pyo3 = { version = "0.20", features = ["extension-module"] }
numpy = "0.20"

[build-dependencies]
pyo3-build-config = "0.20"
```

#### 5.2.4 pyproject.toml (Modified)

```toml
[build-system]
requires = ["maturin>=1.4,<2.0"]
build-backend = "maturin"

[project]
name = "pyroki"
version = "0.1.0"
requires-python = ">=3.10"
classifiers = [
    "Programming Language :: Rust",
    "Programming Language :: Python :: Implementation :: CPython",
]

dependencies = [
    "jax>=0.4.0",
    "jaxlib",
    "jaxlie>=1.0.0",
    "jax_dataclasses>=1.0.0",
    # ... existing deps ...
]

[project.optional-dependencies]
native = []  # Marker for Rust extension

[tool.maturin]
features = ["pyo3/extension-module"]
python-source = "src"
module-name = "pyroki._native.pyroki_native"
```

### 5.3 CI/CD Configuration

#### 5.3.1 GitHub Actions (.github/workflows/rust.yml)

```yaml
name: Rust Build and Test

on:
  push:
    branches: [main, "claude/*"]
  pull_request:
    branches: [main]

jobs:
  build:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: ["3.10", "3.11", "3.12"]

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install Rust
        uses: dtolnay/rust-toolchain@stable

      - name: Cache Rust
        uses: Swatinem/rust-cache@v2
        with:
          workspaces: rust

      - name: Install maturin
        run: pip install maturin

      - name: Build wheels
        run: maturin build --release

      - name: Run Rust tests
        run: cargo test --workspace
        working-directory: rust

      - name: Install and test Python
        run: |
          pip install target/wheels/*.whl
          pip install pytest pytest-benchmark
          pytest tests/ -v

  benchmark:
    runs-on: ubuntu-latest
    needs: build
    steps:
      - uses: actions/checkout@v4
      - name: Run benchmarks
        run: |
          pip install maturin && maturin develop --release
          python benchmark/ik_benchmark.py --native
```

### 5.4 Migration Guide

#### For Users

```markdown
## Upgrading to PyRoki with Rust Acceleration

### Installation

# Standard install (includes Rust extension if available)
pip install pyroki

# Force native extension
pip install pyroki[native]

### Behavior Changes

1. **Automatic Dispatch**: Collision detection automatically uses
   Rust for CPU workloads and JAX for GPU/autodiff contexts.

2. **Environment Variables**:
   - `PYROKI_FORCE_RUST=1`: Always use Rust (errors if unavailable)
   - `PYROKI_FORCE_JAX=1`: Always use JAX (for debugging)
   - `PYROKI_REQUIRE_NATIVE=1`: Error if Rust extension missing

3. **API Compatibility**: No changes to public API. Existing code
   works without modification.

### Verifying Installation

```python
import pyroki
print(pyroki.native_available())  # True if Rust extension loaded
```

### Performance Tips

- For CPU workloads, Rust provides 10-100x speedup
- For GPU workloads, JAX remains optimal
- Large batch sizes benefit most from Rust parallelization
```

### 5.5 Success Verification Checklist

#### Functional Requirements
- [ ] All 11 geometry pair collisions implemented in Rust
- [ ] Batch operations support arbitrary dimensions
- [ ] Results match JAX within 1e-10 tolerance
- [ ] Fallback works when Rust unavailable
- [ ] JAX autodiff works through custom_vjp

#### Performance Requirements
- [ ] Capsule-capsule: 10x+ faster on CPU
- [ ] Segment-segment: 20x+ faster
- [ ] Batch FK (1000): 5x+ faster
- [ ] Memory per collision: <1KB

#### Quality Requirements
- [ ] 100% test coverage for Rust core
- [ ] No memory leaks (verified with valgrind)
- [ ] No undefined behavior (verified with miri)
- [ ] Documentation for all public APIs

#### Distribution Requirements
- [ ] Wheels for Linux (manylinux2014)
- [ ] Wheels for macOS (x86_64 + arm64)
- [ ] Wheels for Windows (x86_64)
- [ ] Source distribution with fallback

---

## Appendix A: Benchmark Methodology

```python
# benchmark/collision_benchmark.py

import time
import numpy as np
import jax.numpy as jnp
from pyroki.collision import Capsule, collide
import pyroki.collision._native as native

def benchmark_capsule_capsule(n_pairs: int, n_iterations: int = 100):
    """Benchmark capsule-capsule collision."""

    # Generate random capsule pairs
    rng = np.random.default_rng(42)
    positions = rng.random((n_pairs, 2, 3)) * 10
    radii = rng.random((n_pairs, 2)) * 0.5 + 0.1
    lengths = rng.random((n_pairs, 2)) * 2 + 0.5

    capsules1 = [
        Capsule.from_radius_height(
            radii[i, 0], lengths[i, 0],
            jnp.array(positions[i, 0]),
            jnp.array([1, 0, 0, 0]),
        )
        for i in range(n_pairs)
    ]
    capsules2 = [
        Capsule.from_radius_height(
            radii[i, 1], lengths[i, 1],
            jnp.array(positions[i, 1]),
            jnp.array([1, 0, 0, 0]),
        )
        for i in range(n_pairs)
    ]

    results = {}

    # Benchmark JAX
    with native._force_jax():
        # Warmup
        for c1, c2 in zip(capsules1[:10], capsules2[:10]):
            _ = collide(c1, c2).block_until_ready()

        start = time.perf_counter()
        for _ in range(n_iterations):
            for c1, c2 in zip(capsules1, capsules2):
                _ = collide(c1, c2).block_until_ready()
        jax_time = (time.perf_counter() - start) / n_iterations
        results["jax"] = jax_time

    # Benchmark Rust
    if native._RUST_AVAILABLE:
        with native._force_rust():
            # Warmup
            for c1, c2 in zip(capsules1[:10], capsules2[:10]):
                _ = collide(c1, c2)

            start = time.perf_counter()
            for _ in range(n_iterations):
                for c1, c2 in zip(capsules1, capsules2):
                    _ = collide(c1, c2)
            rust_time = (time.perf_counter() - start) / n_iterations
            results["rust"] = rust_time

    return results

if __name__ == "__main__":
    for n_pairs in [10, 100, 1000, 10000]:
        results = benchmark_capsule_capsule(n_pairs)
        print(f"\n{n_pairs} pairs:")
        for name, time_ms in results.items():
            print(f"  {name}: {time_ms * 1000:.3f} ms")

        if "rust" in results and "jax" in results:
            speedup = results["jax"] / results["rust"]
            print(f"  Speedup: {speedup:.1f}x")
```

---

## Appendix B: Rust Code Examples

### B.1 Complete Capsule Module

```rust
// rust/pyroki-core/src/collision/capsule.rs

use crate::geometry::{Vec3, Transform3D};
use crate::collision::utils::closest_segment_to_segment_points;

/// A capsule defined by center position, axis direction, length, and radius
#[derive(Debug, Clone, Copy)]
pub struct Capsule {
    pub center: Vec3,
    pub axis: Vec3,  // Unit vector
    pub length: f64,
    pub radius: f64,
}

impl Capsule {
    pub fn new(center: Vec3, axis: Vec3, length: f64, radius: f64) -> Self {
        debug_assert!(
            (axis.norm() - 1.0).abs() < 1e-6,
            "Axis must be normalized"
        );
        Self { center, axis, length, radius }
    }

    /// Create from pose (position + quaternion)
    pub fn from_pose(
        position: Vec3,
        quaternion: [f64; 4],  // wxyz
        length: f64,
        radius: f64,
    ) -> Self {
        let axis = Transform3D::from_quaternion(quaternion)
            .rotate_vector(Vec3::Z);
        Self::new(position, axis, length, radius)
    }

    /// Get segment endpoints
    pub fn endpoints(&self) -> (Vec3, Vec3) {
        let half = self.axis * (self.length / 2.0);
        (self.center - half, self.center + half)
    }

    /// Check if this is effectively a sphere (zero length)
    pub fn is_sphere(&self) -> bool {
        self.length < 1e-10
    }

    /// Convert to sphere if degenerate
    pub fn as_sphere(&self) -> crate::collision::Sphere {
        crate::collision::Sphere::new(self.center, self.radius)
    }
}

/// Compute signed distance between two capsules
///
/// Returns:
/// - Negative value: penetration depth
/// - Zero: touching
/// - Positive value: separation distance
pub fn capsule_capsule_distance(c1: &Capsule, c2: &Capsule) -> f64 {
    // Handle degenerate cases
    if c1.is_sphere() && c2.is_sphere() {
        return crate::collision::sphere::sphere_sphere_distance(
            &c1.as_sphere(),
            &c2.as_sphere(),
        );
    }
    if c1.is_sphere() {
        return sphere_capsule_distance(&c1.as_sphere(), c2);
    }
    if c2.is_sphere() {
        return sphere_capsule_distance(&c2.as_sphere(), c1);
    }

    // Get segment endpoints
    let (a1, b1) = c1.endpoints();
    let (a2, b2) = c2.endpoints();

    // Find closest points on axes
    let (pt1, pt2) = closest_segment_to_segment_points(a1, b1, a2, b2);

    // Sphere-sphere distance at closest points
    let center_dist = (pt1 - pt2).norm();
    center_dist - c1.radius - c2.radius
}

/// Compute distance from sphere to capsule
fn sphere_capsule_distance(
    sphere: &crate::collision::Sphere,
    capsule: &Capsule,
) -> f64 {
    let (a, b) = capsule.endpoints();
    let closest = closest_point_on_segment(sphere.center, a, b);
    let dist = (sphere.center - closest).norm();
    dist - sphere.radius - capsule.radius
}

/// Find closest point on segment to a point
fn closest_point_on_segment(p: Vec3, a: Vec3, b: Vec3) -> Vec3 {
    let ab = b - a;
    let ab_len_sq = ab.dot(&ab);

    if ab_len_sq < 1e-20 {
        return a;  // Degenerate segment
    }

    let t = ((p - a).dot(&ab) / ab_len_sq).clamp(0.0, 1.0);
    a + ab * t
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_parallel_capsules() {
        let c1 = Capsule::new(Vec3::zeros(), Vec3::x(), 2.0, 0.5);
        let c2 = Capsule::new(Vec3::new(0.0, 2.0, 0.0), Vec3::x(), 2.0, 0.5);

        let dist = capsule_capsule_distance(&c1, &c2);
        assert_relative_eq!(dist, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_end_to_end_capsules() {
        let c1 = Capsule::new(Vec3::zeros(), Vec3::x(), 2.0, 0.1);
        let c2 = Capsule::new(Vec3::new(3.0, 0.0, 0.0), Vec3::x(), 2.0, 0.1);

        // Gap = 3 - 1 - 1 - 0.1 - 0.1 = 0.8
        let dist = capsule_capsule_distance(&c1, &c2);
        assert_relative_eq!(dist, 0.8, epsilon = 1e-10);
    }

    #[test]
    fn test_penetrating_capsules() {
        let c1 = Capsule::new(Vec3::zeros(), Vec3::x(), 2.0, 1.0);
        let c2 = Capsule::new(Vec3::new(0.0, 1.0, 0.0), Vec3::x(), 2.0, 1.0);

        // Penetration = 1 - 1 - 1 = -1.0
        let dist = capsule_capsule_distance(&c1, &c2);
        assert_relative_eq!(dist, -1.0, epsilon = 1e-10);
    }
}
```

### B.2 PyO3 Bindings

```rust
// rust/pyroki-python/src/collision.rs

use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyroki_core::collision::{self, Capsule};
use pyroki_core::geometry::Vec3;

/// Compute distance between two capsules
#[pyfunction]
fn capsule_capsule_distance<'py>(
    py: Python<'py>,
    pos1: PyReadonlyArray1<'py, f64>,
    axis1: PyReadonlyArray1<'py, f64>,
    length1: f64,
    radius1: f64,
    pos2: PyReadonlyArray1<'py, f64>,
    axis2: PyReadonlyArray1<'py, f64>,
    length2: f64,
    radius2: f64,
) -> PyResult<f64> {
    let p1 = pos1.as_slice()?;
    let a1 = axis1.as_slice()?;
    let p2 = pos2.as_slice()?;
    let a2 = axis2.as_slice()?;

    let c1 = Capsule::new(
        Vec3::new(p1[0], p1[1], p1[2]),
        Vec3::new(a1[0], a1[1], a1[2]),
        length1,
        radius1,
    );
    let c2 = Capsule::new(
        Vec3::new(p2[0], p2[1], p2[2]),
        Vec3::new(a2[0], a2[1], a2[2]),
        length2,
        radius2,
    );

    Ok(collision::capsule::capsule_capsule_distance(&c1, &c2))
}

/// Batch capsule-capsule distance computation
#[pyfunction]
fn batch_capsule_capsule_distance<'py>(
    py: Python<'py>,
    // Shape: (n, 3) positions
    positions1: PyReadonlyArray2<'py, f64>,
    axes1: PyReadonlyArray2<'py, f64>,
    lengths1: PyReadonlyArray1<'py, f64>,
    radii1: PyReadonlyArray1<'py, f64>,
    positions2: PyReadonlyArray2<'py, f64>,
    axes2: PyReadonlyArray2<'py, f64>,
    lengths2: PyReadonlyArray1<'py, f64>,
    radii2: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let n = positions1.shape()[0];

    let p1 = positions1.as_array();
    let a1 = axes1.as_array();
    let l1 = lengths1.as_slice()?;
    let r1 = radii1.as_slice()?;

    let p2 = positions2.as_array();
    let a2 = axes2.as_array();
    let l2 = lengths2.as_slice()?;
    let r2 = radii2.as_slice()?;

    // Use Rayon for parallel computation
    use rayon::prelude::*;

    let results: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|i| {
            let c1 = Capsule::new(
                Vec3::new(p1[[i, 0]], p1[[i, 1]], p1[[i, 2]]),
                Vec3::new(a1[[i, 0]], a1[[i, 1]], a1[[i, 2]]),
                l1[i],
                r1[i],
            );
            let c2 = Capsule::new(
                Vec3::new(p2[[i, 0]], p2[[i, 1]], p2[[i, 2]]),
                Vec3::new(a2[[i, 0]], a2[[i, 1]], a2[[i, 2]]),
                l2[i],
                r2[i],
            );
            collision::capsule::capsule_capsule_distance(&c1, &c2)
        })
        .collect();

    Ok(PyArray1::from_vec_bound(py, results).unbind())
}

/// Python module
#[pymodule]
fn pyroki_native(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(capsule_capsule_distance, m)?)?;
    m.add_function(wrap_pyfunction!(batch_capsule_capsule_distance, m)?)?;
    Ok(())
}
```

---

## Appendix C: Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Numerical differences vs JAX | Medium | High | Extensive testing, tolerance tuning |
| Build complexity for users | Medium | Medium | Pre-built wheels, fallback to pure Python |
| SIMD portability issues | Low | Low | Runtime feature detection, scalar fallback |
| Memory safety bugs | Low | High | Extensive testing, fuzzing, miri |
| Performance regression | Low | High | Continuous benchmarking in CI |
| JAX version incompatibility | Medium | Medium | Pin compatible versions, test matrix |

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-31 | Claude | Initial SPARC PRD |

---

*This PRD was generated using the SPARC framework for the PyRoki Rust/native extension refactoring project.*
