# Wavefront vs Megakernel Testing Guide

## Available Test Scenes

You now have two comparison functions to test the performance characteristics:

### 1. Simple Scene Comparison (`wavefront_comparison`)
**Scene:** 41 spheres, simple materials
**Expected Result:** Megakernel wins (6x faster)
**Why:** Not enough complexity to offset wavefront overhead

```python
# In main.py
def main():
    wavefront_comparison()
```

**Results:**
- Megakernel: ~7ms/sample (50 Mpix/s)
- Wavefront: ~42ms/sample (8 Mpix/s)
- **Winner: Megakernel (6x faster)**

### 2. Complex Scene Comparison (`vol2_final_scene_comparison`)
**Scene:** 1000+ objects, volumes, complex materials, textures
**Expected Result:** Should show wavefront benefits
**Why:** High BVH complexity causes megakernel divergence

```python
# In main.py
def main():
    vol2_final_scene_comparison()
```

**Scene Details:**
- 400 ground boxes (quads)
- 1000 small spheres in cluster
- Glass spheres with refraction
- Volumes (constant mediums)
- Textured spheres (image, Perlin noise)
- Emissive light sources

## Running the Tests

```bash
cd src
python main.py
```

The script will:
1. Build the scene once
2. Render with megakernel mode → save to `../temp/vol2_final_megakernel.png`
3. Render with wavefront mode → save to `../temp/vol2_final_wavefront.png`
4. Print performance comparison

## What to Expect

### Simple Scene (41 spheres)
```
Megakernel Time:  12.78s
Wavefront Time:   13.97s
Slowdown:         1.09x (megakernel still faster)
```
Megakernel wins because kernel launch overhead dominates.

### Complex Scene (1000+ objects)
**Three possible outcomes:**

#### Outcome A: Wavefront wins (expected)
```
Megakernel Time:  180.5s
Wavefront Time:   120.3s
Speedup:          1.50x (wavefront is FASTER!)
```
BVH complexity causes enough divergence that wavefront's coherence wins.

#### Outcome B: Still close (possible)
```
Megakernel Time:  165.2s
Wavefront Time:   158.7s
Speedup:          1.04x (wavefront is slightly faster)
```
Kernel overhead still significant but complexity helps.

#### Outcome C: Megakernel still wins (possible)
```
Megakernel Time:  145.8s
Wavefront Time:   178.2s
Slowdown:         1.22x (megakernel still faster)
```
If this happens, it means even 1000 objects isn't enough for this GPU/scene.

## Why Results May Vary

### GPU Architecture Matters
- **Apple Silicon (M1/M2/M3)**: Unified memory, kernel launch is cheap → megakernel often wins
- **NVIDIA/AMD**: Discrete GPU, high memory latency → wavefront may win sooner

### Scene Characteristics
Wavefront needs **all** of these:
1. ✅ Many objects (>500)
2. ✅ Deep BVH traversal
3. ✅ Complex materials (glass, volumes)
4. ✅ Long ray paths (10+ bounces)
5. ✅ High resolution (more rays)

Missing any of these favors megakernel.

## Tuning Parameters

### For Faster Testing (Complex Scene)
Edit `vol2_final_scene_comparison()` in `scenes.py`:

```python
# Reduce resolution
cam.img_width = 400  # Instead of 600

# Reduce samples
cam.samples_per_pixel = 20  # Instead of 50

# Reduce sphere count
ns = 500  # Instead of 1000
```

### For More Dramatic Differences
Edit `vol2_final_scene_comparison()`:

```python
# Increase complexity
cam.img_width = 800
cam.samples_per_pixel = 100
ns = 2000  # 2000 small spheres!
```

## Understanding the Output

### Megakernel Output
```
Rendering Progress:
  10/50 ( 20.0%) │  45.2ms │ Elapsed:   0.5s │ Throughput: 7.95M pix/s │ ETA:  1.8s
```
- Consistent sample times (~45ms each)
- High throughput
- Predictable ETA

### Wavefront Output
```
Rendering Progress:
  10/50 ( 20.0%) │  38.1ms │ Elapsed:   0.4s │ Throughput: 9.45M pix/s │ ETA:  1.5s
```
- May have more variance in sample times
- Can be faster or slower depending on scene
- Early samples may be slower (JIT warmup)

## Performance Analysis

After running, compare:

1. **Total render time**: Which is actually faster?
2. **Sample time variance**: Megakernel should be more consistent
3. **Throughput**: Mpix/s (higher is better)
4. **Image quality**: Both should produce identical images

## Common Issues

### Out of Memory
If wavefront runs out of memory on complex scene:

```python
# In fields.py, line 254
max_rays = width * height * 1  # Reduce from 2 to 1
```

### Scene Too Simple
If megakernel still wins on complex scene, the scene isn't complex enough.
Try:
- Increase sphere count (`ns = 2000`)
- Add more glass materials (long paths)
- Increase resolution

### Scene Too Complex
If both are very slow:
- Reduce `cam.samples_per_pixel`
- Reduce `cam.img_width`
- Reduce `ns` (sphere count)

## Conclusion

The complex scene test should finally show wavefront's benefits. If megakernel still wins, it means:
1. Your GPU architecture favors megakernel (Apple Silicon likely)
2. Kernel launch overhead is just too high for wavefront to overcome
3. The scene needs to be even more complex (5000+ objects)

Either way, both implementations are correct and produce identical images - use whichever is faster for your hardware!
