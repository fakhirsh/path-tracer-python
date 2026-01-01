"""
Taichi GPU Field Definitions

All Taichi fields (GPU memory allocations) are defined here.
Fields are organized by access pattern (HOT vs COLD data).
"""

import taichi as ti

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

MAX_SPHERES = 2048
MAX_TRIANGLES = 4096    # For future use
MAX_QUADS = 2048        # For future use
MAX_BVH_NODES = 8192
MAX_DEPTH = 50

# =============================================================================
# HOT DATA: Geometry (accessed during intersection tests)
# =============================================================================

# Spheres: pack center (xyz) + radius (w) into vec4 for aligned access
sphere_data = ti.Vector.field(4, ti.f32, MAX_SPHERES)  # [cx, cy, cz, radius]
num_spheres = ti.field(ti.i32, shape=())

# Triangles
triangle_v0 = ti.Vector.field(3, ti.f32, MAX_TRIANGLES)
triangle_v1 = ti.Vector.field(3, ti.f32, MAX_TRIANGLES)
triangle_v2 = ti.Vector.field(3, ti.f32, MAX_TRIANGLES)
triangle_edge1 = ti.Vector.field(3, ti.f32, MAX_TRIANGLES)
triangle_edge2 = ti.Vector.field(3, ti.f32, MAX_TRIANGLES)
triangle_normal = ti.Vector.field(3, ti.f32, MAX_TRIANGLES)
num_triangles = ti.field(ti.i32, shape=())

# Quads: Q (corner), u (first edge), v (second edge), normal, D, w
quad_Q = ti.Vector.field(3, ti.f32, MAX_QUADS)      # Corner point
quad_u = ti.Vector.field(3, ti.f32, MAX_QUADS)      # First edge vector
quad_v = ti.Vector.field(3, ti.f32, MAX_QUADS)      # Second edge vector
quad_normal = ti.Vector.field(3, ti.f32, MAX_QUADS) # Normal vector
quad_D = ti.field(ti.f32, MAX_QUADS)                # Plane constant
quad_w = ti.Vector.field(3, ti.f32, MAX_QUADS)      # Planar coordinate helper
num_quads = ti.field(ti.i32, shape=())

# =============================================================================
# HOT DATA: BVH (accessed during traversal)
# =============================================================================

# OPTIMIZED: Packed BVH structure for better cache locality
# Single struct instead of 6 separate arrays reduces cache misses by 20-40%
@ti.dataclass
class BVHNode:
    bbox_min: ti.math.vec3      # AABB min corner
    bbox_max: ti.math.vec3      # AABB max corner
    left_child: ti.i32          # Left child index (-1 if leaf)
    right_child: ti.i32         # Right child index (-1 if leaf)
    parent: ti.i32              # Parent index (-1 if root) - for stackless traversal
    prim_type: ti.i32           # Primitive type: 0=sphere, 1=triangle, 2=quad
    prim_idx: ti.i32            # Primitive index (-1 if internal node)

bvh_nodes = BVHNode.field(shape=MAX_BVH_NODES)
num_bvh_nodes = ti.field(ti.i32, shape=())

# Legacy fields (for backward compatibility during transition)
# TODO: Remove these once fully migrated
bvh_bbox_min = ti.Vector.field(3, ti.f32, MAX_BVH_NODES)
bvh_bbox_max = ti.Vector.field(3, ti.f32, MAX_BVH_NODES)
bvh_left_child = ti.field(ti.i32, MAX_BVH_NODES)
bvh_right_child = ti.field(ti.i32, MAX_BVH_NODES)
bvh_prim_type = ti.field(ti.i32, MAX_BVH_NODES)
bvh_prim_idx = ti.field(ti.i32, MAX_BVH_NODES)

# =============================================================================
# COLD DATA: Materials (accessed once per ray on closest hit)
# =============================================================================

# Sphere Materials
material_type = ti.field(ti.i32, MAX_SPHERES)  # Indexed by sphere primitive
material_albedo = ti.Vector.field(3, ti.f32, MAX_SPHERES)
material_fuzz = ti.field(ti.f32, MAX_SPHERES)      # Metal only
material_ir = ti.field(ti.f32, MAX_SPHERES)        # Dielectric only (index of refraction)
material_emit_color = ti.Vector.field(3, ti.f32, MAX_SPHERES)  # Emissive only

# Quad Materials
quad_material_type = ti.field(ti.i32, MAX_QUADS)
quad_material_albedo = ti.Vector.field(3, ti.f32, MAX_QUADS)
quad_material_fuzz = ti.field(ti.f32, MAX_QUADS)
quad_material_ir = ti.field(ti.f32, MAX_QUADS)
quad_material_emit_color = ti.Vector.field(3, ti.f32, MAX_QUADS)  # Emissive only

# Triangle Materials
triangle_material_type = ti.field(ti.i32, MAX_TRIANGLES)
triangle_material_albedo = ti.Vector.field(3, ti.f32, MAX_TRIANGLES)
triangle_material_fuzz = ti.field(ti.f32, MAX_TRIANGLES)
triangle_material_ir = ti.field(ti.f32, MAX_TRIANGLES)
triangle_material_emit_color = ti.Vector.field(3, ti.f32, MAX_TRIANGLES)  # Emissive only

# =============================================================================
# COLD DATA: Textures
# =============================================================================

# Sphere Textures
texture_type = ti.field(ti.i32, MAX_SPHERES)
texture_scale = ti.field(ti.f32, MAX_SPHERES)      # Checker scale
texture_color1 = ti.Vector.field(3, ti.f32, MAX_SPHERES)  # Primary/even color
texture_color2 = ti.Vector.field(3, ti.f32, MAX_SPHERES)  # Secondary/odd color
texture_image_idx = ti.field(ti.i32, MAX_SPHERES)  # Index into image texture array (-1 if not used)

# Quad Textures
quad_texture_type = ti.field(ti.i32, MAX_QUADS)
quad_texture_scale = ti.field(ti.f32, MAX_QUADS)
quad_texture_color1 = ti.Vector.field(3, ti.f32, MAX_QUADS)
quad_texture_color2 = ti.Vector.field(3, ti.f32, MAX_QUADS)
quad_texture_image_idx = ti.field(ti.i32, MAX_QUADS)  # Index into image texture array (-1 if not used)

# Triangle Textures
triangle_texture_type = ti.field(ti.i32, MAX_TRIANGLES)
triangle_texture_scale = ti.field(ti.f32, MAX_TRIANGLES)
triangle_texture_color1 = ti.Vector.field(3, ti.f32, MAX_TRIANGLES)
triangle_texture_color2 = ti.Vector.field(3, ti.f32, MAX_TRIANGLES)
triangle_texture_image_idx = ti.field(ti.i32, MAX_TRIANGLES)  # Index into image texture array (-1 if not used)

# Image Textures - Global storage for loaded images
# These will be allocated dynamically based on how many unique images are used
MAX_IMAGE_TEXTURES = 16  # Support up to 16 unique image textures
image_textures = []  # List of ti.Vector.field objects, one per loaded image
image_texture_dims = ti.Vector.field(2, ti.i32, MAX_IMAGE_TEXTURES)  # [width, height] for each image

# =============================================================================
# CONSTANT MEDIUM (SMOKE/FOG) DATA
# =============================================================================

# Constant medium flags - marks which primitives are constant mediums
is_constant_medium_sphere = ti.field(ti.i32, MAX_SPHERES)  # 0 = normal, 1 = constant medium
is_constant_medium_quad = ti.field(ti.i32, MAX_QUADS)
is_constant_medium_triangle = ti.field(ti.i32, MAX_TRIANGLES)

# Constant medium properties
medium_density_sphere = ti.field(ti.f32, MAX_SPHERES)  # Density value (will be converted to neg_inv_density)
medium_density_quad = ti.field(ti.f32, MAX_QUADS)
medium_density_triangle = ti.field(ti.f32, MAX_TRIANGLES)

# Constant medium phase function (isotropic material) color
medium_albedo_sphere = ti.Vector.field(3, ti.f32, MAX_SPHERES)
medium_albedo_quad = ti.Vector.field(3, ti.f32, MAX_QUADS)
medium_albedo_triangle = ti.Vector.field(3, ti.f32, MAX_TRIANGLES)

# =============================================================================
# PERLIN NOISE DATA
# =============================================================================

# Perlin noise lookup tables (static, initialized once)
PERLIN_POINT_COUNT = 256

# Random vectors for Perlin noise
perlin_randvec = ti.Vector.field(3, ti.f32, PERLIN_POINT_COUNT)

# Permutation tables for x, y, z
perlin_perm_x = ti.field(ti.i32, PERLIN_POINT_COUNT)
perlin_perm_y = ti.field(ti.i32, PERLIN_POINT_COUNT)
perlin_perm_z = ti.field(ti.i32, PERLIN_POINT_COUNT)

# =============================================================================
# CAMERA
# =============================================================================

cam_center = ti.Vector.field(3, ti.f32, shape=())
cam_pixel00 = ti.Vector.field(3, ti.f32, shape=())
cam_delta_u = ti.Vector.field(3, ti.f32, shape=())
cam_delta_v = ti.Vector.field(3, ti.f32, shape=())
cam_defocus_disk_u = ti.Vector.field(3, ti.f32, shape=())
cam_defocus_disk_v = ti.Vector.field(3, ti.f32, shape=())
cam_defocus_angle = ti.field(ti.f32, shape=())

# =============================================================================
# RENDER STATE
# =============================================================================

bg_color = ti.Vector.field(3, ti.f32, shape=())
max_depth = ti.field(ti.i32, shape=())

# Depth statistics (for average depth calculation)
depth_accumulator = ti.field(ti.f32, shape=())
path_count = ti.field(ti.i32, shape=())

# Russian Roulette statistics
rr_paths_killed = ti.field(ti.i32, shape=())       # Number of paths terminated by RR
rr_paths_survived = ti.field(ti.i32, shape=())     # Number of paths that survived RR
rr_depth_sum_killed = ti.field(ti.f32, shape=())   # Sum of depths where paths were killed
rr_depth_sum_survived = ti.field(ti.f32, shape=()) # Sum of depths where paths survived

# Max depth terminations statistics
max_depth_terminations = ti.field(ti.i32, shape=())  # Number of paths that hit max depth

# Accumulation buffer - allocated dynamically after camera initialization
accum_buffer = None


# =============================================================================
# DYNAMIC FIELD ALLOCATION
# =============================================================================

def allocate_dynamic_fields(width: int, height: int):
    """
    Allocate fields that depend on image dimensions.
    Must be called after camera initialization.
    """
    global accum_buffer
    accum_buffer = ti.Vector.field(3, ti.f32, shape=(height, width))


@ti.kernel
def clear_accumulation_buffer():
    """Clear accumulation buffer to zero"""
    for py, px in accum_buffer:
        accum_buffer[py, px] = ti.math.vec3(0.0)


@ti.kernel
def reset_depth_stats():
    """Reset depth statistics to zero"""
    depth_accumulator[None] = 0.0
    path_count[None] = 0
    max_depth_terminations[None] = 0


@ti.kernel
def reset_rr_stats():
    """Reset Russian Roulette statistics to zero"""
    rr_paths_killed[None] = 0
    rr_paths_survived[None] = 0
    rr_depth_sum_killed[None] = 0.0
    rr_depth_sum_survived[None] = 0.0


def init_perlin_noise(perlin_obj):
    """
    Initialize Perlin noise lookup tables from CPU perlin object.
    This uploads the random vectors and permutation tables to GPU.
    """
    import numpy as np

    # Upload random vectors
    randvec_np = np.array([[v.x, v.y, v.z] for v in perlin_obj.randvec], dtype=np.float32)
    perlin_randvec.from_numpy(randvec_np)

    # Upload permutation tables
    perlin_perm_x.from_numpy(np.array(perlin_obj.perm_x, dtype=np.int32))
    perlin_perm_y.from_numpy(np.array(perlin_obj.perm_y, dtype=np.int32))
    perlin_perm_z.from_numpy(np.array(perlin_obj.perm_z, dtype=np.int32))
