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

# Quad Materials
quad_material_type = ti.field(ti.i32, MAX_QUADS)
quad_material_albedo = ti.Vector.field(3, ti.f32, MAX_QUADS)
quad_material_fuzz = ti.field(ti.f32, MAX_QUADS)
quad_material_ir = ti.field(ti.f32, MAX_QUADS)

# Triangle Materials
triangle_material_type = ti.field(ti.i32, MAX_TRIANGLES)
triangle_material_albedo = ti.Vector.field(3, ti.f32, MAX_TRIANGLES)
triangle_material_fuzz = ti.field(ti.f32, MAX_TRIANGLES)
triangle_material_ir = ti.field(ti.f32, MAX_TRIANGLES)

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
