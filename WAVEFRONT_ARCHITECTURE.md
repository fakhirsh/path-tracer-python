# Wavefront Path Tracing Architecture

## Overview

This document describes the wavefront path tracing implementation added to the Taichi renderer. The wavefront architecture is an alternative to the traditional megakernel approach, offering better GPU utilization through reduced thread divergence.

## Architecture Comparison

### Megakernel (Depth-First)
**Traditional approach where each thread traces a complete ray path:**
- Each GPU thread handles one pixel
- Thread traces ray through all bounces until termination
- High thread divergence (some rays terminate early, others continue to max depth)
- Uses implicit ray state (thread registers/stack)
- Simpler code structure

**Flow:**
```
Thread 1: Ray1 → Bounce1 → Bounce2 → ... → BounceN → Done
Thread 2: Ray2 → Bounce1 → Bounce2 → Done
Thread 3: Ray3 → Bounce1 → ... → BounceM → Done
(All threads diverge - poor GPU occupancy)
```

### Wavefront (Breadth-First)
**Process all rays at the same bounce depth together:**
- All rays are processed in waves by bounce depth
- Each wave executes the same operation on all active rays
- Lower thread divergence (all threads do same work)
- Uses explicit ray queues (GPU memory)
- More complex code but better GPU utilization

**Flow:**
```
Wave 1: Generate all camera rays
Wave 2: Intersect all rays (bounce 0)
Wave 3: Shade all rays (bounce 0)
Wave 4: Intersect all scattered rays (bounce 1)
Wave 5: Shade all rays (bounce 1)
...
(All threads in a wave do the same work - better occupancy)
```

## Implementation Details

### 1. Data Structures ([fields.py](src/render_server/taichi_renderer/fields.py))

Added explicit ray queue storage:

```python
# Ray queues (double buffered)
ray_origins              # Current wave origins
ray_directions           # Current wave directions
ray_throughput           # Accumulated color attenuation
ray_pixel_index          # Which pixel owns this ray
ray_depth               # Current bounce depth

next_ray_origins         # Next wave buffers (for ping-pong)
next_ray_directions
next_ray_throughput
next_ray_pixel_index
next_ray_depth

# Hit records (intersection results)
hit_valid               # Did ray hit geometry?
hit_t                   # Ray parameter at hit
hit_point               # World space position
hit_normal              # Surface normal
hit_prim_type           # Primitive type (sphere/triangle/quad)
hit_prim_idx            # Primitive index
hit_front_face          # Front or back face?

# Queue management
active_ray_count        # Number of rays in current wave
next_ray_count          # Number of rays in next wave
```

**Memory Usage:**
- For 800x600 image: ~38MB (using 2x multiplier instead of max_depth)
- Most rays terminate early via Russian Roulette, so conservative allocation is safe

### 2. Kernels ([kernels.py](src/render_server/taichi_renderer/kernels.py))

Split the monolithic `trace_ray()` function into separate kernels:

#### `generate_camera_rays(width, height)`
- Generates initial camera rays for all pixels
- Stores rays in queue with full throughput (1,1,1)
- Sets `active_ray_count = width * height`

#### `intersect_rays()`
- Intersects all active rays with scene using BVH traversal
- Stores hit information in `hit_*` fields
- Reuses existing `traverse_bvh()` function

#### `shade_miss_rays()`
- Processes rays that missed the scene
- Adds background color contribution to accumulation buffer
- Uses atomic operations for accumulation

#### `shade_and_scatter()`
- **Core shading kernel** that handles:
  - Emission evaluation
  - Material scattering (Lambertian, Metal, Dielectric, Emissive)
  - Constant medium (volume) support
  - Russian Roulette path termination
  - Generates scattered rays for next wave
- Uses atomic operations to build next ray queue

#### `swap_ray_buffers()`
- Copies next wave buffers to current wave
- Updates `active_ray_count`
- Enables ping-pong buffering

### 3. Render Loop ([renderer.py](src/render_server/taichi_renderer/renderer.py))

Added `render_wavefront()` method:

```python
def render_wavefront(self, enable_preview=True):
    for sample in range(samples_per_pixel):
        # Generate camera rays
        generate_camera_rays(width, height)

        # Wave loop - process rays by bounce depth
        for bounce in range(max_depth):
            if active_ray_count == 0:
                break

            # Intersect all active rays
            intersect_rays()

            # Shade misses (background)
            shade_miss_rays()

            # Shade hits and generate scattered rays
            shade_and_scatter()

            # Swap buffers for next iteration
            swap_ray_buffers()
```

Original `render()` method remains unchanged for comparison.

## Key Features Preserved

All features from the megakernel implementation are preserved:

✅ Multiple materials (Lambertian, Metal, Dielectric, Emissive, Isotropic)
✅ Textures (Solid, Checker, Image, Perlin noise)
✅ Constant mediums (volumes/smoke)
✅ Russian Roulette path termination
✅ Importance sampling (cosine-weighted for Lambertian)
✅ BVH acceleration structure
✅ Anti-aliasing with jitter
✅ Defocus blur (depth of field)

## Performance Characteristics

### Expected Benefits
- **Reduced divergence**: All threads in a wave do the same work
- **Better cache locality**: Sequential memory access patterns
- **No per-thread stack**: Saves memory bandwidth
- **Easier profiling**: Can measure each stage independently

### Potential Drawbacks
- **More memory**: Explicit ray storage (~38MB for 800x600)
- **More kernel launches**: Overhead from multiple kernel invocations
- **Atomic operations**: Queue management requires atomics (can be slow)

### Optimization Opportunities (Future Work)
1. **Material sorting**: Separate queues per material type (reduce divergence further)
2. **Stream compaction**: Remove dead rays each bounce (reduce memory)
3. **Tiled rendering**: Process 64x64 tiles at a time (reduce memory footprint)
4. **Persistent threads**: Keep kernel resident, pull work from queue (reduce launch overhead)

## Usage

### Running the Comparison
```bash
cd src
python main.py  # Runs wavefront_comparison() by default
```

This will:
1. Render the same scene with megakernel mode
2. Render the same scene with wavefront mode
3. Display performance comparison
4. Save both images for visual verification

### Using Wavefront in Your Code
```python
from render_server.renderer_factory import RendererFactory

# Create renderer
renderer = RendererFactory.create('taichi', world, cam, "output.png")

# Use wavefront mode
renderer.render_wavefront(enable_preview=True)

# Or use megakernel mode
renderer.render(enable_preview=True)
```

## Files Modified

1. **[src/render_server/taichi_renderer/fields.py](src/render_server/taichi_renderer/fields.py)**
   - Added ray queue data structures (lines 191-277)
   - Added `allocate_wavefront_fields()` function

2. **[src/render_server/taichi_renderer/kernels.py](src/render_server/taichi_renderer/kernels.py)**
   - Added 5 new kernels (lines 1212-1448):
     - `generate_camera_rays()`
     - `intersect_rays()`
     - `shade_miss_rays()`
     - `shade_and_scatter()`
     - `swap_ray_buffers()`

3. **[src/render_server/taichi_renderer/renderer.py](src/render_server/taichi_renderer/renderer.py)**
   - Added wavefront field allocation in `__init__()` (line 75)
   - Added `render_wavefront()` method (lines 249-354)
   - Updated `render()` docstring to indicate megakernel mode

4. **[src/scenes.py](src/scenes.py)**
   - Added `wavefront_comparison()` function (lines 1132-1244)

5. **[src/main.py](src/main.py)**
   - Changed default scene to `wavefront_comparison()`

## Technical Notes

### Why Double Buffering?
The wavefront architecture uses ping-pong buffers (`ray_*` and `next_ray_*`) to avoid copying data. The `shade_and_scatter()` kernel reads from current buffers and writes to next buffers, then `swap_ray_buffers()` makes next buffers current.

### Atomic Operations
Atomic operations are used for:
- Building next ray queue (`ti.atomic_add(next_ray_count)`)
- Accumulating color to pixels (`ti.atomic_add(accum_buffer)`)

These are necessary for thread safety but can impact performance on some GPUs.

### Russian Roulette
Both architectures use the same Russian Roulette parameters:
- Start at depth 5
- Survival probability = min(max(throughput), 0.95)
- Boost throughput by 1/probability to maintain unbiased result

### Volume Rendering
Constant mediums (volumes/smoke) are fully supported in wavefront mode:
- Volume passthrough creates continuation ray (same depth)
- Volume scattering creates scattered ray (depth++)
- Fallback to surface shading if volume exit fails

## Debugging

Enable kernel profiling:
```bash
export TAICHI_KERNEL_PROFILER=1
python src/main.py
```

Check active ray counts:
```python
# In Python after kernel execution
print(f"Active rays: {fields.active_ray_count[None]}")
print(f"Next rays: {fields.next_ray_count[None]}")
```

## References

- **Wavefront Path Tracing**: Laine et al., "Megakernels Considered Harmful: Wavefront Path Tracing on GPUs"
- **Original Megakernel**: Peter Shirley, "Ray Tracing in One Weekend" series
- **Taichi Documentation**: https://docs.taichi-lang.org/

## Future Improvements

1. **Material sorting**: Split rays by material type before shading
2. **Stream compaction**: Remove terminated rays from queue
3. **Work stealing**: Balance load across GPU cores
4. **Hybrid approach**: Use megakernel for simple scenes, wavefront for complex ones
5. **Adaptive depth**: Early termination based on contribution
