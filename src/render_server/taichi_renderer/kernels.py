"""
Taichi GPU Kernels and Functions

All @ti.func and @ti.kernel definitions for GPU path tracing.
Must be in single file to avoid Taichi scope issues.
"""

import taichi as ti
import math
from . import fields

# =============================================================================
# RANDOM / UTILITY FUNCTIONS
# =============================================================================

@ti.func
def random_in_unit_disk() -> ti.math.vec3:
    """Generate random point in unit disk (for defocus blur)"""
    result = ti.math.vec3(0.0)
    while True:
        p = ti.math.vec3(ti.random() * 2.0 - 1.0, ti.random() * 2.0 - 1.0, 0.0)
        if p.dot(p) < 1.0:
            result = p
            break
    return result


@ti.func
def random_unit_vector() -> ti.math.vec3:
    """Generate random unit vector on sphere surface"""
    result = ti.math.vec3(0.0)
    while True:
        p = ti.math.vec3(ti.random() * 2.0 - 1.0, ti.random() * 2.0 - 1.0, ti.random() * 2.0 - 1.0)
        lensq = p.dot(p)
        if lensq < 1.0 and lensq > 1e-20:
            result = p.normalized()
            break
    return result


@ti.func
def random_cosine_direction(normal: ti.math.vec3) -> ti.math.vec3:
    """
    Generate cosine-weighted random direction in hemisphere around normal.
    This is importance sampling for Lambertian materials.
    """
    r1 = ti.random()
    r2 = ti.random()

    # Cosine-weighted sampling
    z = ti.sqrt(1.0 - r2)
    phi = 2.0 * math.pi * r1
    sin_theta = ti.sqrt(r2)
    x = ti.cos(phi) * sin_theta
    y = ti.sin(phi) * sin_theta

    # Build orthonormal basis around normal
    w = normal.normalized()
    a = ti.math.vec3(0.0, 1.0, 0.0) if ti.abs(w.x) > 0.9 else ti.math.vec3(1.0, 0.0, 0.0)
    v = w.cross(a).normalized()
    u = w.cross(v)

    # Transform to world space
    return (x * u + y * v + z * w).normalized()


# =============================================================================
# RAY GENERATION
# =============================================================================

@ti.func
def get_ray(px: ti.i32, py: ti.i32) -> tuple:
    """
    Generate camera ray for pixel (px, py) with random jitter for AA.
    Returns: (origin, direction)
    Note: No motion blur - removed per requirements.
    """
    # Random offset within pixel for anti-aliasing
    offset_x = ti.random() - 0.5
    offset_y = ti.random() - 0.5

    # Calculate pixel sample position
    pixel_sample = (fields.cam_pixel00[None] +
                   (px + offset_x) * fields.cam_delta_u[None] +
                   (py + offset_y) * fields.cam_delta_v[None])

    # Ray origin (with defocus blur if enabled)
    ray_origin = fields.cam_center[None]
    if fields.cam_defocus_angle[None] > 0.0:
        # Sample from defocus disk
        p = random_in_unit_disk()
        ray_origin = fields.cam_center[None] + p.x * fields.cam_defocus_disk_u[None] + p.y * fields.cam_defocus_disk_v[None]

    ray_direction = pixel_sample - ray_origin

    return ray_origin, ray_direction


# =============================================================================
# INTERSECTION FUNCTIONS
# =============================================================================

@ti.func
def hit_aabb(node_idx: ti.i32, ray_origin: ti.math.vec3, ray_dir: ti.math.vec3,
             t_min: ti.f32, t_max: ti.f32) -> bool:
    """Test ray against AABB of BVH node"""
    bbox_min = fields.bvh_bbox_min[node_idx]
    bbox_max = fields.bvh_bbox_max[node_idx]

    # Use local copies to track interval
    t_min_local = t_min
    t_max_local = t_max
    hit = True

    # Test all three axes (no early return allowed in Taichi)
    for axis in ti.static(range(3)):
        inv_d = 1.0 / ray_dir[axis]
        t0 = (bbox_min[axis] - ray_origin[axis]) * inv_d
        t1 = (bbox_max[axis] - ray_origin[axis]) * inv_d

        if inv_d < 0.0:
            t0, t1 = t1, t0

        t_min_local = ti.max(t0, t_min_local)
        t_max_local = ti.min(t1, t_max_local)

        if t_max_local <= t_min_local:
            hit = False

    return hit


@ti.func
def hit_sphere(sphere_idx: ti.i32, ray_origin: ti.math.vec3, ray_dir: ti.math.vec3,
               t_min: ti.f32, t_max: ti.f32) -> tuple:
    """
    Test ray-sphere intersection.
    Returns: (hit, t, hit_point, normal)
    Note: Uses packed sphere_data[idx] = (cx, cy, cz, radius)
    """
    # Unpack sphere data
    sphere_vec4 = fields.sphere_data[sphere_idx]
    center = ti.math.vec3(sphere_vec4[0], sphere_vec4[1], sphere_vec4[2])
    radius = sphere_vec4[3]

    # Ray-sphere intersection math
    oc = center - ray_origin
    a = ray_dir.dot(ray_dir)
    h = ray_dir.dot(oc)
    c = oc.dot(oc) - radius * radius

    discriminant = h * h - a * c

    hit = False
    t = 0.0
    hit_point = ti.math.vec3(0.0)
    normal = ti.math.vec3(0.0)

    if discriminant >= 0.0:
        sqrtd = ti.sqrt(discriminant)

        # Find nearest root in acceptable range
        root = (h - sqrtd) / a
        if root < t_min or root > t_max:
            root = (h + sqrtd) / a

        if root >= t_min and root <= t_max:
            hit = True
            t = root
            hit_point = ray_origin + t * ray_dir
            normal = (hit_point - center) / radius

    return hit, t, hit_point, normal


@ti.func
def hit_triangle(tri_idx: ti.i32, ray_origin: ti.math.vec3, ray_dir: ti.math.vec3,
                 t_min: ti.f32, t_max: ti.f32) -> tuple:
    """
    Test ray-triangle intersection using Möller–Trumbore algorithm.
    Returns: (hit, t, hit_point, normal)
    FOR FUTURE USE - implement skeleton now.
    """
    # Return no-hit for now
    return False, 0.0, ti.math.vec3(0.0), ti.math.vec3(0.0)


@ti.func
def hit_quad(quad_idx: ti.i32, ray_origin: ti.math.vec3, ray_dir: ti.math.vec3,
             t_min: ti.f32, t_max: ti.f32) -> tuple:
    """
    Test ray-quad intersection.
    Returns: (hit, t, hit_point, normal)
    FOR FUTURE USE - implement skeleton now.
    """
    # Return no-hit for now
    return False, 0.0, ti.math.vec3(0.0), ti.math.vec3(0.0)


@ti.func
def traverse_bvh(ray_origin: ti.math.vec3, ray_dir: ti.math.vec3,
                 t_min: ti.f32, t_max: ti.f32) -> tuple:
    """
    Traverse BVH and find closest intersection.
    Uses iterative stack-based traversal (no recursion on GPU).
    Returns: (hit, t, hit_point, normal, primitive_idx)

    CRITICAL: Check bvh_prim_type to dispatch to correct intersection function.
    """
    hit_anything = False
    closest_t = t_max
    closest_hit_point = ti.math.vec3(0.0)
    closest_normal = ti.math.vec3(0.0)
    closest_prim_idx = 0

    # Stack-based iterative BVH traversal (no recursion on GPU)
    # Stack stores node indices to visit
    stack = ti.Vector([0 for _ in range(64)], dt=ti.i32)  # Max depth 64
    stack_ptr = 0
    stack[stack_ptr] = 0  # Start with root node
    stack_ptr += 1

    while stack_ptr > 0:
        # Pop node from stack
        stack_ptr -= 1
        node_idx = stack[stack_ptr]

        # Skip invalid nodes
        if node_idx < 0 or node_idx >= fields.num_bvh_nodes[None]:
            continue

        # Test ray against node's bounding box
        if not hit_aabb(node_idx, ray_origin, ray_dir, t_min, closest_t):
            continue  # Ray misses this node's bbox, skip entire subtree

        # Check if this is a leaf node (contains a primitive)
        prim_idx = fields.bvh_prim_idx[node_idx]
        if prim_idx >= 0:
            # Leaf node - test primitive intersection based on type
            prim_type = fields.bvh_prim_type[node_idx]
            hit = False
            t = 0.0
            hit_point = ti.math.vec3(0.0)
            normal = ti.math.vec3(0.0)

            if prim_type == 0:  # PRIM_SPHERE
                hit, t, hit_point, normal = hit_sphere(
                    prim_idx, ray_origin, ray_dir, t_min, closest_t
                )
            elif prim_type == 1:  # PRIM_TRIANGLE
                hit, t, hit_point, normal = hit_triangle(
                    prim_idx, ray_origin, ray_dir, t_min, closest_t
                )
            elif prim_type == 2:  # PRIM_QUAD
                hit, t, hit_point, normal = hit_quad(
                    prim_idx, ray_origin, ray_dir, t_min, closest_t
                )

            if hit and t < closest_t:
                hit_anything = True
                closest_t = t
                closest_hit_point = hit_point
                closest_normal = normal
                closest_prim_idx = prim_idx
        else:
            # Internal node - add children to stack
            left_child = fields.bvh_left_child[node_idx]
            right_child = fields.bvh_right_child[node_idx]

            if right_child >= 0 and stack_ptr < 64:
                stack[stack_ptr] = right_child
                stack_ptr += 1

            if left_child >= 0 and stack_ptr < 64:
                stack[stack_ptr] = left_child
                stack_ptr += 1

    return hit_anything, closest_t, closest_hit_point, closest_normal, closest_prim_idx


# =============================================================================
# MATERIAL FUNCTIONS
# =============================================================================

@ti.func
def reflect(v: ti.math.vec3, n: ti.math.vec3) -> ti.math.vec3:
    """Reflect vector v around normal n"""
    return v - 2.0 * v.dot(n) * n


@ti.func
def refract(uv: ti.math.vec3, n: ti.math.vec3, etai_over_etat: ti.f32) -> ti.math.vec3:
    """Snell's law refraction"""
    cos_theta = ti.min(-uv.dot(n), 1.0)
    r_out_perp = etai_over_etat * (uv + cos_theta * n)
    r_out_parallel = -ti.sqrt(ti.abs(1.0 - r_out_perp.dot(r_out_perp))) * n
    return r_out_perp + r_out_parallel


@ti.func
def reflectance(cosine: ti.f32, ref_idx: ti.f32) -> ti.f32:
    """Schlick's approximation for Fresnel reflectance"""
    r0 = (1.0 - ref_idx) / (1.0 + ref_idx)
    r0 = r0 * r0
    return r0 + (1.0 - r0) * ti.pow(1.0 - cosine, 5.0)


@ti.func
def scatter(ray_dir: ti.math.vec3, hit_point: ti.math.vec3, normal: ti.math.vec3,
            prim_idx: ti.i32) -> tuple:
    """
    Compute scattered ray based on material type.
    Returns: (did_scatter, scatter_direction, attenuation)

    Dispatches to Lambertian, Metal, or Dielectric based on material_type[prim_idx].
    """
    mat_type = fields.material_type[prim_idx]
    scatter_dir = ti.math.vec3(0.0)
    attenuation = ti.math.vec3(1.0)
    scattered = False

    # LAMBERTIAN material
    if mat_type == 0:
        # Evaluate texture at hit point to get albedo
        albedo = eval_texture(prim_idx, hit_point)
        scatter_dir = random_cosine_direction(normal)
        attenuation = albedo
        scattered = True

    # METAL material
    elif mat_type == 1:
        albedo = fields.material_albedo[prim_idx]
        reflected = reflect(ray_dir.normalized(), normal)
        fuzz = fields.material_fuzz[prim_idx]
        scatter_dir = reflected + fuzz * random_unit_vector()

        # Only scatter if ray is reflected outward
        if scatter_dir.dot(normal) > 0.0:
            attenuation = albedo
            scattered = True

    # DIELECTRIC material (glass)
    elif mat_type == 2:
        ir = fields.material_ir[prim_idx]

        # Determine if ray is entering or exiting
        front_face = ray_dir.dot(normal) < 0.0
        normal_facing = normal if front_face else -normal
        refraction_ratio = (1.0 / ir) if front_face else ir

        unit_dir = ray_dir.normalized()
        cos_theta = ti.min(-unit_dir.dot(normal_facing), 1.0)
        sin_theta = ti.sqrt(1.0 - cos_theta * cos_theta)

        cannot_refract = refraction_ratio * sin_theta > 1.0

        # Choose reflection or refraction
        if cannot_refract or reflectance(cos_theta, refraction_ratio) > ti.random():
            scatter_dir = reflect(unit_dir, normal_facing)
        else:
            scatter_dir = refract(unit_dir, normal_facing, refraction_ratio)

        # Dielectric doesn't attenuate (always white)
        attenuation = ti.math.vec3(1.0)
        scattered = True

    return scattered, scatter_dir, attenuation


# =============================================================================
# TEXTURE FUNCTIONS
# =============================================================================

@ti.func
def eval_texture(prim_idx: ti.i32, hit_point: ti.math.vec3) -> ti.math.vec3:
    """
    Evaluate texture at hit point.
    Returns: RGB color

    Handles: solid color, checker pattern.
    Uses texture_type[prim_idx] to dispatch.
    """
    tex_type = fields.texture_type[prim_idx]
    result = ti.math.vec3(1.0, 1.0, 1.0)

    if tex_type == 0:  # TEX_SOLID
        result = fields.texture_color1[prim_idx]
    elif tex_type == 1:  # TEX_CHECKER
        # Checker pattern based on 3D position
        inv_scale = 1.0 / fields.texture_scale[prim_idx]
        x_int = ti.floor(inv_scale * hit_point.x)
        y_int = ti.floor(inv_scale * hit_point.y)
        z_int = ti.floor(inv_scale * hit_point.z)

        is_even = (ti.cast(x_int, ti.i32) + ti.cast(y_int, ti.i32) + ti.cast(z_int, ti.i32)) % 2 == 0

        if is_even:
            result = fields.texture_color1[prim_idx]
        else:
            result = fields.texture_color2[prim_idx]

    return result


# =============================================================================
# PATH TRACING INTEGRATOR
# =============================================================================

@ti.func
def trace_ray(ray_origin: ti.math.vec3, ray_dir: ti.math.vec3) -> ti.math.vec3:
    """
    Trace a single ray through the scene.
    Iterative implementation (no recursion).
    Returns: final color for this ray path.

    Loop structure:
    1. Traverse BVH to find closest hit
    2. If miss: return background color * throughput
    3. If hit: scatter based on material, update throughput
    4. Repeat until max depth or ray absorbed
    """
    color = ti.math.vec3(0.0)
    throughput = ti.math.vec3(1.0, 1.0, 1.0)

    current_origin = ray_origin
    current_dir = ray_dir.normalized()

    # Iterative path tracing loop (instead of recursion)
    for depth in range(fields.max_depth[None]):
        # Test for intersection with scene
        hit, t, hit_point, normal, prim_idx = traverse_bvh(
            current_origin, current_dir, 0.001, 1e10
        )

        if hit:
            # Material scattering
            scattered, scatter_dir, attenuation = scatter(
                current_dir, hit_point, normal, prim_idx
            )

            if scattered:
                # Update ray for next bounce
                current_origin = hit_point
                current_dir = scatter_dir
                throughput = throughput * attenuation
            else:
                # Ray was absorbed
                break

        else:
            # Ray missed - add sky color weighted by throughput
            # Sky gradient: blue to white based on y direction
            unit_dir = current_dir.normalized()
            a = 0.5 * (unit_dir.y + 1.0)
            sky = (1.0 - a) * ti.math.vec3(1.0, 1.0, 1.0) + a * fields.bg_color[None]
            color = color + throughput * sky
            break

    return color


# =============================================================================
# MAIN RENDER KERNEL
# =============================================================================

@ti.kernel
def render_sample():
    """
    Render one sample per pixel.
    Each thread handles one pixel independently.
    Accumulates result into accum_buffer.
    """
    for py, px in fields.accum_buffer:
        ray_origin, ray_dir = get_ray(px, py)
        color = trace_ray(ray_origin, ray_dir)
        fields.accum_buffer[py, px] += color


@ti.kernel
def clear_accum_buffer():
    """Clear accumulation buffer to zero"""
    for py, px in fields.accum_buffer:
        fields.accum_buffer[py, px] = ti.math.vec3(0.0)
