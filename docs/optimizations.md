# Path Tracer Features & Architecture

Comprehensive list of all features, enhancements, and architectural components implemented in this path tracer.

## RENDERING TECHNIQUES & ALGORITHMS
- Path tracing (recursive Monte Carlo)
- Megakernel approach (depth-first)
- Wavefront path tracing (breadth-first)
- Russian Roulette path termination
- Importance sampling (cosine-weighted)
- Monte Carlo integration
- Tone mapping with gamma correction
- Anti-aliasing (stochastic pixel sampling)
- Defocus blur / Depth of field
- Motion blur support

## MATERIALS
- Lambertian (diffuse)
- Metal (specular reflection + fuzz)
- Dielectric (Fresnel reflection, Schlick approximation)
- Diffuse light / Emissive
- Isotropic (volumetric scattering)
- Subsurface scattering (displacement-based)
- Subsurface volumetric (Henyey-Greenstein phase function)

## TEXTURES
- Solid color
- Checkerboard (3D-aware)
- Image texture mapping
- Perlin noise with turbulence
- Texture composition

## GEOMETRIC PRIMITIVES
- Spheres (stationary + moving)
- Quads (parametric intersection)
- Triangles (Muller-Trumbore algorithm)
- Meshes (OBJ loading via PyWavefront)
- Klein bottle (parametric surface)

## ACCELERATION STRUCTURES
- BVH with median split
- SAH BVH builder (optimal cost function)
- AABB tests
- BVH flattening for GPU
- Stackless BVH traversal
- Primitive type encoding

## LIGHTING & ILLUMINATION
- Direct lighting (emissive materials)
- Area lights (quad-based)
- Global illumination (recursive bounces)
- Background/sky rendering
- Next-event estimation

## CAMERA FEATURES
- Pinhole camera model
- Adjustable FOV
- Position/orientation controls (lookfrom, lookat, up)
- Sub-pixel jittering
- Circular aperture for DOF
- Focus distance control
- Interactive orbit controls (mouse rotation, spherical coordinates)

## SAMPLING METHODS
- Stratified sampling
- Cosine-weighted hemisphere
- Random unit vector (isotropic)
- Defocus disk sampling
- Henyey-Greenstein phase function (anisotropic)
- Russian Roulette with adaptive probability

## WAVEFRONT OPTIMIZATIONS
- Reduced thread divergence
- Direct writes for first bounce (no atomics)
- Eliminated shape queries in hot loops
- Streamlined emission checks
- Explicit reset kernels
- Early-return volume passthrough
- Atomic operation reduction (~50% first bounce)
- Register pressure reduction

## ARCHITECTURE PATTERNS
- Client-server separation (Python/GPU)
- Renderer factory (pluggable backends)
- Abstract base renderer
- GPU field management & pooling
- Double buffering (ping-pong queues)
- Queue-based ray processing
- Kernel composition (separate stages)
- Interactive viewer with async rendering

## FILE I/O & SCENE MANAGEMENT
- PPM image output
- PNG image output
- OBJ mesh loading
- Image texture loading (PIL/Pillow)
- Scene compilation to GPU format
- BVH compilation & flattening
- Material data packing
- Texture registry

## POST-PROCESSING
- Gamma correction
- Tone mapping
- Progressive rendering with accumulation
- Live preview with tile updates
- Color space conversion

## MATHEMATICAL TECHNIQUES
- Perlin noise (gradient vectors, permutation tables)
- Hermite interpolation
- Trilinear interpolation
- Hemispherical sampling (orthonormal basis)
- Spherical to Cartesian conversion
- Cross product normals
- Determinant-based ray-triangle tests
- Barycentric coordinates
- Matrix-free rotation

## VOLUMETRIC RENDERING
- Constant medium (fog/smoke, exponential sampling)
- Volume extinction & scattering
- Isotropic phase function
- Absorption coefficient modeling
- Ray marching

## PERFORMANCE ANALYSIS
- Path depth statistics
- Render time estimation (windowed averaging)
- Performance profiling (cProfile)
- Kernel profiling (Taichi)
- Timing breakdown by stage
- Active ray count monitoring
- Sample count tracking

## RENDERING BACKENDS
- Taichi GPU (Metal on macOS)
- CPU rendering (pure Python)
- GPU rendering (legacy)
- Configuration-based selection

## DATA STRUCTURES
- Vec3 (full vector algebra)
- Point3, Color (typed vec3)
- Ray (origin, direction, time)
- Interval (1D range)
- AABB (3D bounding box)
- Hit record
- BVH node (internal/leaf)
- Material/Texture base classes (polymorphic)

## SCENE DEFINITIONS
- Multiple predefined scenes (vol1_sec9_5, vol1_sec14_1, vol2_sec2_6, etc.)
- Cornell Box
- Cornell Box with smoke
- Complex final scene (1000+ objects)
- Interactive variations
- Wavefront/megakernel comparison scenes
- Mesh loading demos

## INTERACTIVE FEATURES
- Live preview window (Tkinter)
- Real-time accumulation display
- Mouse-controlled camera
- Progressive refinement
- Throttled updates (50ms intervals)
- Sample count & progress display
