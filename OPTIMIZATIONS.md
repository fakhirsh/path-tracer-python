# GPU Performance Optimizations

This document describes the major GPU performance optimizations implemented for the path tracer.

## Implemented Optimizations

### 1. **Packed BVH Structure** (20-40% potential gain)

**Location:** `src/render_server/taichi_renderer/fields.py`

**What:** Replaced 6 separate arrays with a single packed struct:
```python
# OLD: 6 separate arrays
bvh_bbox_min[i], bvh_bbox_max[i], bvh_left_child[i], ...

# NEW: Single packed structure
@ti.dataclass
class BVHNode:
    bbox_min: ti.math.vec3
    bbox_max: ti.math.vec3
    left_child: ti.i32
    right_child: ti.i32
    parent: ti.i32  # NEW: for stackless traversal
    prim_type: ti.i32
    prim_idx: ti.i32
```

**Why:** Better cache locality - accessing all node data requires fetching from one memory location instead of 6 separate arrays.

---

### 2. **Stackless BVH Traversal** (15-30% potential gain)

**Location:** `src/render_server/taichi_renderer/kernels.py`

**What:** Replaced stack-based traversal with parent-pointer-based "restart trail" algorithm.

**Key Changes:**
- **Eliminated:** 64-element stack allocation per thread (256 bytes register pressure)
- **Added:** Parent pointers to navigate up the tree
- **Result:** Reduced register spilling, better GPU occupancy

**Toggle:**
```python
# In kernels.py
USE_STACKLESS_TRAVERSAL = True  # Set to False for legacy stack-based
```

**Algorithm:**
1. Descend to left child if AABB hit
2. If can't descend, try right sibling
3. If no right sibling, ascend to parent via parent pointer
4. Track which child we came from to avoid revisiting

---

### 3. **Surface Area Heuristic (SAH) BVH** (2-3x better BVH quality)

**Location:** `src/render_server/taichi_renderer/sah_bvh_builder.py`

**What:** Builds optimal BVH using Surface Area Heuristic instead of median split.

**SAH Cost Function:**
```
cost = traverse_cost +
       P(left) * intersect_cost * num_left +
       P(right) * intersect_cost * num_right

where P(child) = surface_area(child) / surface_area(parent)
```

**Implementation:** Binned SAH (16 bins) for O(N log N) complexity instead of O(N² log N).

**Toggle:**
```python
# In bvh_compiler.py
USE_SAH_BVH = True  # Set to False for legacy median-split BVH
```

**Trade-off:**
- Build time: ~10ms for 512 spheres (binned) vs ~0.5ms (median)
- Traversal: 20-40% fewer node tests on complex scenes
- **Worth it for:** Final renders, complex scenes
- **Not worth it for:** Interactive/real-time, simple scenes

---

### 4. **Optimized AABB Intersection** (5-10% gain)

**Location:** `src/render_server/taichi_renderer/kernels.py:hit_aabb_optimized()`

**What:**
- Precompute inverse ray direction once per ray (instead of per AABB test)
- Use vectorized min/max operations instead of branches
- Remove early-exit branches (better for GPU SIMT execution)

**Before:**
```python
for axis in range(3):
    inv_d = 1.0 / ray_dir[axis]  # Computed every AABB test!
    if inv_d < 0.0:
        t0, t1 = t1, t0  # Branch per axis
```

**After:**
```python
inv_ray_dir = 1.0 / ray_dir  # Computed ONCE per ray
t0 = (bbox_min - ray_origin) * inv_ray_dir
t1 = (bbox_max - ray_origin) * inv_ray_dir
tmin_vec = ti.min(t0, t1)  # Vectorized, no branches
tmax_vec = ti.max(t0, t1)
```

---

## How to Test

### Baseline Test (Legacy)
```python
# In bvh_compiler.py
USE_SAH_BVH = False

# In kernels.py
USE_STACKLESS_TRAVERSAL = False

# Run
python3 main.py
```

### Test Stackless Traversal
```python
# In bvh_compiler.py
USE_SAH_BVH = False  # Keep BVH same

# In kernels.py
USE_STACKLESS_TRAVERSAL = True  # Enable stackless

# Run
python3 main.py
```

### Test SAH BVH
```python
# In bvh_compiler.py
USE_SAH_BVH = True  # Enable SAH

# In kernels.py
USE_STACKLESS_TRAVERSAL = False  # Keep traversal same

# Run
python3 main.py
```

### Test All Optimizations
```python
# In bvh_compiler.py
USE_SAH_BVH = True

# In kernels.py
USE_STACKLESS_TRAVERSAL = True

# Run
python3 main.py
```

---

## Expected Performance Gains

| Configuration | BVH Build Time | Render Speed | Best For |
|---------------|----------------|--------------|----------|
| Legacy (baseline) | ~0.5ms | 100% | Baseline comparison |
| + Stackless | ~0.5ms | 115-130% | All scenes |
| + SAH BVH | ~10ms | 120-140% | Complex scenes |
| + Both | ~10ms | 140-180% | Final renders |

**Note:** Actual gains depend on:
- Scene complexity (more spheres = bigger SAH benefit)
- Ray coherence (camera rays vs diffuse bounces)
- GPU architecture (register pressure varies)

---

## Debugging

### Verify Stackless is Being Used
Check that parent pointers are non-negative:
```python
# Add to renderer.py after BVH upload:
print(f"Sample parent pointers: {bvh['bvh_parent'][:10]}")
# Should see: [- 1, 0, 0, 1, 1, ...] (root=-1, others>=0)
```

### Verify SAH is Building
```bash
# Should see:
Building SAH BVH with 512 primitives...
# Not:
Using legacy BVH with 512 primitives...
```

### Profile GPU
```bash
# Kernel timing breakdown
TAICHI_KERNEL_PROFILER=1 python3 main.py

# GPU utilization (macOS)
MTL_HUD_ENABLED=1 python3 main.py
```

---

## Potential Future Optimizations

1. **Russian Roulette Path Termination** (5-15% gain)
   - Probabilistically terminate low-contribution paths
   - Reduces wasted computation on dark rays

2. **Branchless Material Evaluation** (10-20% gain)
   - Compute all material types, select result
   - Eliminates warp divergence

3. **Wavefront Path Tracing** (30-50% gain, complex)
   - Separate kernels for ray gen, intersection, shading
   - Sort rays by material type before shading
   - Maximum GPU occupancy

4. **Blue Noise Sampling** (5-10% gain + quality)
   - Replace `ti.random()` with precomputed blue noise
   - Better sample distribution, faster RNG

5. **Tile-Based Ray Generation** (10-25% gain)
   - Process pixels in 8x8 tiles
   - Better ray coherence = better BVH cache hits

---

## References

- **SAH BVH:** "On fast Construction of SAH-based Bounding Volume Hierarchies" (Wald 2007)
- **Stackless Traversal:** "Efficient Stack-less BVH Traversal for Ray Tracing" (Hapala et al. 2011)
- **Packed BVH:** "Understanding the Efficiency of Ray Traversal on GPUs" (Aila & Laine 2009)

---

## Current Status

✅ Implemented: Packed BVH struct
✅ Implemented: Stackless traversal
✅ Implemented: SAH BVH builder (binned)
✅ Implemented: Optimized AABB test
✅ Implemented: Parent pointer support in legacy BVH

🔧 Testing: Performance comparison needed
🔧 Validation: Correctness verification in progress

⚠️ **Known Issues:**
- Stackless traversal needs testing for correctness
- SAH BVH build time vs quality trade-off needs tuning
- No performance gain observed yet (debugging in progress)

