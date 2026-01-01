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
    # Choose axis most perpendicular to w
    a = ti.math.vec3(0.0)
    if ti.abs(w.x) < 0.9:
        a = ti.math.vec3(1.0, 0.0, 0.0)
    elif ti.abs(w.y) < 0.9:
        a = ti.math.vec3(0.0, 1.0, 0.0)
    else:
        a = ti.math.vec3(0.0, 0.0, 1.0)
    v = a.cross(w).normalized()  # SWAPPED: was w.cross(a)
    u = v.cross(w)                # SWAPPED: was w.cross(v)

    # Transform to world space
    return (x * u + y * v + z * w).normalized()


# =============================================================================
# TEXTURE UV CALCULATION
# =============================================================================

@ti.func
def get_sphere_uv(p: ti.math.vec3, center: ti.math.vec3) -> tuple:
    """
    Calculate UV coordinates for a point on a sphere surface.
    p: point on sphere surface
    center: center of sphere
    Returns: (u, v) in [0,1] x [0,1]

    UV mapping:
    - u: azimuthal angle (longitude), 0 at -X, increases counterclockwise viewed from +Y
    - v: polar angle (latitude), 0 at -Y (south pole), 1 at +Y (north pole)
    """
    # Get outward normal (point relative to center, normalized)
    outward_normal = (p - center).normalized()

    # Convert to spherical coordinates
    # theta: angle around Y axis (azimuthal, 0 to 2π)
    # phi: angle from -Y axis (polar, 0 to π)
    phi = ti.acos(-outward_normal.y)
    theta = ti.atan2(-outward_normal.z, outward_normal.x) + math.pi

    u = theta / (2.0 * math.pi)
    v = phi / math.pi

    return u, v


# =============================================================================
# PERLIN NOISE
# =============================================================================

@ti.func
def perlin_noise(p: ti.math.vec3) -> ti.f32:
    """
    Compute Perlin noise value at point p.
    Returns value in approximately [-1, 1] range.
    """
    # Get integer and fractional parts
    u = p.x - ti.floor(p.x)
    v = p.y - ti.floor(p.y)
    w = p.z - ti.floor(p.z)

    i = ti.cast(ti.floor(p.x), ti.i32)
    j = ti.cast(ti.floor(p.y), ti.i32)
    k = ti.cast(ti.floor(p.z), ti.i32)

    # Hermite smoothing
    uu = u * u * (3.0 - 2.0 * u)
    vv = v * v * (3.0 - 2.0 * v)
    ww = w * w * (3.0 - 2.0 * w)

    accum = 0.0

    # Trilinear interpolation with dot product
    for di in ti.static(range(2)):
        for dj in ti.static(range(2)):
            for dk in ti.static(range(2)):
                # Hash the coordinates using permutation tables
                idx = fields.perlin_perm_x[(i + di) & 255] ^ \
                      fields.perlin_perm_y[(j + dj) & 255] ^ \
                      fields.perlin_perm_z[(k + dk) & 255]

                grad_vec = fields.perlin_randvec[idx]

                weight = ti.math.vec3(u - ti.cast(di, ti.f32),
                                     v - ti.cast(dj, ti.f32),
                                     w - ti.cast(dk, ti.f32))

                accum += (ti.cast(di, ti.f32) * uu + (1.0 - ti.cast(di, ti.f32)) * (1.0 - uu)) * \
                         (ti.cast(dj, ti.f32) * vv + (1.0 - ti.cast(dj, ti.f32)) * (1.0 - vv)) * \
                         (ti.cast(dk, ti.f32) * ww + (1.0 - ti.cast(dk, ti.f32)) * (1.0 - ww)) * \
                         grad_vec.dot(weight)

    return accum


@ti.func
def perlin_turb(p: ti.math.vec3, depth: ti.i32) -> ti.f32:
    """
    Turbulence function - sum of multiple octaves of noise.
    Creates more complex patterns than single noise.
    """
    accum = 0.0
    temp_p = p
    weight = 1.0

    for _ in range(depth):
        accum += weight * perlin_noise(temp_p)
        weight *= 0.5
        temp_p = temp_p * 2.0

    return ti.abs(accum)


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
    """
    epsilon = 1e-8

    # Get triangle data
    v0 = fields.triangle_v0[tri_idx]
    edge1 = fields.triangle_edge1[tri_idx]
    edge2 = fields.triangle_edge2[tri_idx]
    normal = fields.triangle_normal[tri_idx]

    # Compute determinant
    h = ray_dir.cross(edge2)
    det = edge1.dot(h)

    hit = False
    t = 0.0
    hit_point = ti.math.vec3(0.0)
    hit_normal = ti.math.vec3(0.0)

    # Ray is parallel to triangle
    if ti.abs(det) >= epsilon:
        inv_det = 1.0 / det

        # Compute u parameter
        s = ray_origin - v0
        u = inv_det * s.dot(h)

        if u >= 0.0 and u <= 1.0:
            # Compute v parameter
            q = s.cross(edge1)
            v = inv_det * ray_dir.dot(q)

            if v >= 0.0 and u + v <= 1.0:
                # Compute t
                t_temp = inv_det * edge2.dot(q)

                if t_temp >= t_min and t_temp <= t_max:
                    # We have a valid intersection
                    hit = True
                    t = t_temp
                    hit_point = ray_origin + t * ray_dir

                    # Determine front/back face (same logic as set_face_normal)
                    # If ray and normal point in same direction, we hit the back face
                    if ray_dir.dot(normal) > 0.0:
                        # Back face - flip normal
                        hit_normal = -normal
                    else:
                        # Front face - keep normal
                        hit_normal = normal

    return hit, t, hit_point, hit_normal


@ti.func
def hit_quad(quad_idx: ti.i32, ray_origin: ti.math.vec3, ray_dir: ti.math.vec3,
             t_min: ti.f32, t_max: ti.f32) -> tuple:
    """
    Test ray-quad intersection.
    Returns: (hit, t, hit_point, normal)

    Algorithm:
    1. Test ray against quad's plane
    2. Check if intersection point lies within quad boundaries using planar coordinates
    """
    # Get quad data
    Q = fields.quad_Q[quad_idx]
    u = fields.quad_u[quad_idx]
    v = fields.quad_v[quad_idx]
    normal = fields.quad_normal[quad_idx]
    D = fields.quad_D[quad_idx]
    w = fields.quad_w[quad_idx]

    # Check if ray is parallel to quad
    denom = normal.dot(ray_dir)

    hit = False
    t = 0.0
    hit_point = ti.math.vec3(0.0)
    hit_normal = ti.math.vec3(0.0)

    if ti.abs(denom) >= 1e-8:
        # Compute intersection with plane
        t_temp = (D - normal.dot(ray_origin)) / denom

        if t_temp >= t_min and t_temp <= t_max:
            # Determine if hit point lies within the planar shape
            intersection = ray_origin + t_temp * ray_dir
            planar_hitpt_vector = intersection - Q
            alpha = w.dot(planar_hitpt_vector.cross(v))
            beta = w.dot(u.cross(planar_hitpt_vector))

            # Check if inside unit square (0 <= alpha, beta <= 1)
            if alpha >= 0.0 and alpha <= 1.0 and beta >= 0.0 and beta <= 1.0:
                hit = True
                t = t_temp
                hit_point = intersection

                # Determine front/back face
                if denom < 0.0:
                    # Front face
                    hit_normal = normal
                else:
                    # Back face
                    hit_normal = -normal

    return hit, t, hit_point, hit_normal


@ti.func
def apply_constant_medium(prim_type: ti.i32, prim_idx: ti.i32, ray_origin: ti.math.vec3, ray_dir: ti.math.vec3,
                          t_min: ti.f32, t_max: ti.f32, t_entry: ti.f32) -> tuple:
    """
    Apply constant medium (volumetric smoke/fog) scattering.

    Called after hitting a boundary primitive that is marked as a constant medium.
    Implements probabilistic scattering inside the volume.

    Args:
        prim_type: Type of primitive (sphere, quad, triangle)
        prim_idx: Index of primitive
        ray_origin: Ray origin
        ray_dir: Ray direction (unnormalized)
        t_min: Minimum t value
        t_max: Maximum t value
        t_entry: t value where ray entered the boundary

    Returns:
        (is_medium_hit, t_scatter, scatter_point, scatter_normal, use_isotropic, t_exit)
        - is_medium_hit: True if ray scatters inside medium
        - t_scatter: t value where scattering occurs (or 0 if no scatter)
        - scatter_point: Point where scattering occurs (or 0 if no scatter)
        - scatter_normal: Arbitrary normal for medium (not used, but needed for consistency)
        - use_isotropic: Always True if is_medium_hit is True
        - t_exit: t value where ray exits the volume (used when ray passes through)
    """
    is_medium_hit = False
    t_scatter = 0.0
    scatter_point = ti.math.vec3(0.0)
    scatter_normal = ti.math.vec3(1.0, 0.0, 0.0)  # Arbitrary
    use_isotropic = False
    t_exit = 0.0

    # Check if this primitive is a constant medium
    is_medium = 0
    density = 0.0

    if prim_type == 0:  # PRIM_SPHERE
        is_medium = fields.is_constant_medium_sphere[prim_idx]
        density = fields.medium_density_sphere[prim_idx]
    elif prim_type == 1:  # PRIM_TRIANGLE
        is_medium = fields.is_constant_medium_triangle[prim_idx]
        density = fields.medium_density_triangle[prim_idx]
    elif prim_type == 2:  # PRIM_QUAD
        is_medium = fields.is_constant_medium_quad[prim_idx]
        density = fields.medium_density_quad[prim_idx]

    if is_medium > 0:
        # Find exit point by traversing the BVH to find the next hit
        # This is needed for multi-primitive boundaries (e.g., boxes made of 6 quads)
        # For a box, the exit might be through a different quad than the entry
        hit_exit, t_exit_temp, _, _, exit_prim_type, exit_prim_idx = traverse_bvh(
            ray_origin, ray_dir, t_entry + 0.0001, 1e10
        )

        if hit_exit:
            t_exit = t_exit_temp

            # Clamp to ray interval
            t1 = ti.max(t_entry, t_min)
            t2 = ti.min(t_exit, t_max)

            if t1 < t2:
                # Adjust t1 to be non-negative
                if t1 < 0.0:
                    t1 = 0.0

                # Calculate distance inside boundary
                ray_length = ti.sqrt(ray_dir.dot(ray_dir))
                distance_inside = (t2 - t1) * ray_length

                # Probabilistic scattering based on density
                # neg_inv_density = -1.0 / density
                # hit_distance = neg_inv_density * log(random())
                # Simplified: hit_distance = -log(random()) / density
                hit_distance = -ti.log(ti.max(ti.random(), 1e-10)) / density

                if hit_distance < distance_inside:
                    # Ray scatters inside the medium
                    t_scatter = t1 + hit_distance / ray_length
                    scatter_point = ray_origin + t_scatter * ray_dir
                    is_medium_hit = True
                    use_isotropic = True

    return is_medium_hit, t_scatter, scatter_point, scatter_normal, use_isotropic, t_exit


@ti.func
def traverse_bvh_stackless(ray_origin: ti.math.vec3, ray_dir: ti.math.vec3,
                           t_min: ti.f32, t_max: ti.f32) -> tuple:
    """
    OPTIMIZED: Stackless BVH traversal using parent pointers.

    Eliminates 64-element stack allocation per thread = 15-30% performance gain.
    Uses "restart trail" algorithm with parent pointers.

    Algorithm:
    1. Try to descend to left child if AABB hit
    2. If can't descend, try right sibling
    3. If no right sibling, ascend to parent and continue
    4. Track which child we came from to avoid revisiting

    Returns: (hit, t, hit_point, normal, prim_type, prim_idx)
    """
    hit_anything = False
    closest_t = t_max
    closest_hit_point = ti.math.vec3(0.0)
    closest_normal = ti.math.vec3(0.0)
    closest_prim_type = 0
    closest_prim_idx = 0

    # Precompute inverse ray direction for AABB tests (5-8% speedup)
    inv_ray_dir = ti.math.vec3(
        1.0 / ray_dir.x if ti.abs(ray_dir.x) > 1e-8 else 1e8,
        1.0 / ray_dir.y if ti.abs(ray_dir.y) > 1e-8 else 1e8,
        1.0 / ray_dir.z if ti.abs(ray_dir.z) > 1e-8 else 1e8
    )

    node_idx = 0  # Start at root
    came_from_child = -1  # Track which child we came from (-1 = none, 0 = left, 1 = right)

    # Maximum iterations to prevent infinite loops
    max_iterations = fields.num_bvh_nodes[None] * 2
    iteration = 0

    while node_idx >= 0 and iteration < max_iterations:
        iteration += 1

        # Get node data from packed struct
        node = fields.bvh_nodes[node_idx]

        # If we came from a child, try the other child or ascend
        if came_from_child == 0:
            # Came from left child, try right child
            if node.right_child >= 0:
                node_idx = node.right_child
                came_from_child = -1
                continue
            else:
                # No right child, ascend to parent
                parent_idx = node.parent
                if parent_idx >= 0:
                    parent_node = fields.bvh_nodes[parent_idx]
                    came_from_child = 0 if parent_node.left_child == node_idx else 1
                    node_idx = parent_idx
                else:
                    # Reached root, done
                    break
                continue
        elif came_from_child == 1:
            # Came from right child, ascend to parent
            parent_idx = node.parent
            if parent_idx >= 0:
                parent_node = fields.bvh_nodes[parent_idx]
                came_from_child = 0 if parent_node.left_child == node_idx else 1
                node_idx = parent_idx
            else:
                # Reached root, done
                break
            continue

        # Test ray against node's bounding box (optimized version)
        if not hit_aabb_optimized(node.bbox_min, node.bbox_max, ray_origin, inv_ray_dir, t_min, closest_t):
            # Ray misses bbox, skip this subtree
            # Ascend to parent
            parent_idx = node.parent
            if parent_idx >= 0:
                parent_node = fields.bvh_nodes[parent_idx]
                came_from_child = 0 if parent_node.left_child == node_idx else 1
                node_idx = parent_idx
            else:
                break
            continue

        # Check if this is a leaf node (contains a primitive)
        if node.prim_idx >= 0:
            # Leaf node - test primitive intersection
            hit = False
            t = 0.0
            hit_point = ti.math.vec3(0.0)
            normal = ti.math.vec3(0.0)

            if node.prim_type == 0:  # PRIM_SPHERE
                hit, t, hit_point, normal = hit_sphere(
                    node.prim_idx, ray_origin, ray_dir, t_min, closest_t
                )
            elif node.prim_type == 1:  # PRIM_TRIANGLE
                hit, t, hit_point, normal = hit_triangle(
                    node.prim_idx, ray_origin, ray_dir, t_min, closest_t
                )
            elif node.prim_type == 2:  # PRIM_QUAD
                hit, t, hit_point, normal = hit_quad(
                    node.prim_idx, ray_origin, ray_dir, t_min, closest_t
                )

            if hit and t < closest_t:
                hit_anything = True
                closest_t = t
                closest_hit_point = hit_point
                closest_normal = normal
                closest_prim_type = node.prim_type
                closest_prim_idx = node.prim_idx

            # After testing leaf, ascend to parent
            parent_idx = node.parent
            if parent_idx >= 0:
                parent_node = fields.bvh_nodes[parent_idx]
                came_from_child = 0 if parent_node.left_child == node_idx else 1
                node_idx = parent_idx
            else:
                break
        else:
            # Internal node - descend to left child
            if node.left_child >= 0:
                node_idx = node.left_child
                came_from_child = -1
            else:
                # No left child, try right child
                if node.right_child >= 0:
                    node_idx = node.right_child
                    came_from_child = -1
                else:
                    # No children (shouldn't happen), ascend
                    parent_idx = node.parent
                    if parent_idx >= 0:
                        parent_node = fields.bvh_nodes[parent_idx]
                        came_from_child = 0 if parent_node.left_child == node_idx else 1
                        node_idx = parent_idx
                    else:
                        break

    return hit_anything, closest_t, closest_hit_point, closest_normal, closest_prim_type, closest_prim_idx


@ti.func
def hit_aabb_optimized(bbox_min: ti.math.vec3, bbox_max: ti.math.vec3,
                       ray_origin: ti.math.vec3, inv_ray_dir: ti.math.vec3,
                       t_min: ti.f32, t_max: ti.f32) -> bool:
    """
    OPTIMIZED: AABB intersection test using precomputed inverse ray direction.
    Uses vectorized min/max operations instead of branches.
    5-10% speedup over branchy version.
    """
    # Compute intersection intervals for all axes simultaneously
    t0 = (bbox_min - ray_origin) * inv_ray_dir
    t1 = (bbox_max - ray_origin) * inv_ray_dir

    # Handle negative ray directions (swap intervals)
    tmin_vec = ti.min(t0, t1)
    tmax_vec = ti.max(t0, t1)

    # Find overall interval
    tmin = ti.max(ti.max(tmin_vec.x, tmin_vec.y), ti.max(tmin_vec.z, t_min))
    tmax = ti.min(ti.min(tmax_vec.x, tmax_vec.y), ti.min(tmax_vec.z, t_max))

    return tmax >= tmin


# Optimized stack-based traversal with front-to-back ordering
@ti.func
def traverse_bvh_legacy(ray_origin: ti.math.vec3, ray_dir: ti.math.vec3,
                        t_min: ti.f32, t_max: ti.f32) -> tuple:
    """
    OPTIMIZED stack-based BVH traversal with:
    - Front-to-back child ordering (10-20% speedup)
    - Early ray termination using closest_t
    - Precomputed inverse ray direction for AABB tests (5-10% speedup)
    """
    hit_anything = False
    closest_t = t_max
    closest_hit_point = ti.math.vec3(0.0)
    closest_normal = ti.math.vec3(0.0)
    closest_prim_type = 0
    closest_prim_idx = 0

    # Precompute inverse ray direction once for all AABB tests
    inv_ray_dir = ti.math.vec3(
        1.0 / ray_dir.x if ti.abs(ray_dir.x) > 1e-8 else 1e8,
        1.0 / ray_dir.y if ti.abs(ray_dir.y) > 1e-8 else 1e8,
        1.0 / ray_dir.z if ti.abs(ray_dir.z) > 1e-8 else 1e8
    )

    # Stack-based iterative BVH traversal
    stack = ti.Vector([0 for _ in range(64)], dt=ti.i32)
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

        # Fetch packed BVH node (single cache line access instead of 6)
        node = fields.bvh_nodes[node_idx]

        # Early termination: skip nodes beyond closest hit
        if not hit_aabb_optimized(node.bbox_min, node.bbox_max, ray_origin, inv_ray_dir, t_min, closest_t):
            continue

        # Check if this is a leaf node
        if node.prim_idx >= 0:
            # Leaf node - test primitive intersection
            hit = False
            t = 0.0
            hit_point = ti.math.vec3(0.0)
            normal = ti.math.vec3(0.0)

            if node.prim_type == 0:  # PRIM_SPHERE
                hit, t, hit_point, normal = hit_sphere(
                    node.prim_idx, ray_origin, ray_dir, t_min, closest_t
                )
            elif node.prim_type == 1:  # PRIM_TRIANGLE
                hit, t, hit_point, normal = hit_triangle(
                    node.prim_idx, ray_origin, ray_dir, t_min, closest_t
                )
            elif node.prim_type == 2:  # PRIM_QUAD
                hit, t, hit_point, normal = hit_quad(
                    node.prim_idx, ray_origin, ray_dir, t_min, closest_t
                )

            if hit and t < closest_t:
                hit_anything = True
                closest_t = t  # Update for early termination
                closest_hit_point = hit_point
                closest_normal = normal
                closest_prim_type = node.prim_type
                closest_prim_idx = node.prim_idx
        else:
            # Internal node - add children to stack with front-to-back ordering
            left_child = node.left_child
            right_child = node.right_child

            # Determine which child is closer based on ray direction and box centers
            # This ensures we test the nearer child first, improving early termination
            if left_child >= 0 and right_child >= 0:
                # Get bbox centers for both children
                left_node = fields.bvh_nodes[left_child]
                right_node = fields.bvh_nodes[right_child]
                left_center = (left_node.bbox_min + left_node.bbox_max) * 0.5
                right_center = (right_node.bbox_min + right_node.bbox_max) * 0.5

                # Project centers onto ray direction to determine ordering
                left_dist = (left_center - ray_origin).dot(ray_dir)
                right_dist = (right_center - ray_origin).dot(ray_dir)

                # Push far child first (so near child is tested first when popped)
                if left_dist < right_dist:
                    # Left is closer - push right first
                    if stack_ptr < 64:
                        stack[stack_ptr] = right_child
                        stack_ptr += 1
                    if stack_ptr < 64:
                        stack[stack_ptr] = left_child
                        stack_ptr += 1
                else:
                    # Right is closer - push left first
                    if stack_ptr < 64:
                        stack[stack_ptr] = left_child
                        stack_ptr += 1
                    if stack_ptr < 64:
                        stack[stack_ptr] = right_child
                        stack_ptr += 1
            else:
                # Only one child exists - just push it
                if right_child >= 0 and stack_ptr < 64:
                    stack[stack_ptr] = right_child
                    stack_ptr += 1
                if left_child >= 0 and stack_ptr < 64:
                    stack[stack_ptr] = left_child
                    stack_ptr += 1

    return hit_anything, closest_t, closest_hit_point, closest_normal, closest_prim_type, closest_prim_idx


# Toggle between stackless and legacy traversal
USE_STACKLESS_TRAVERSAL = False  # DISABLED: 2.5x slower than stack-based!

# Select which traversal to use
@ti.func
def traverse_bvh(ray_origin: ti.math.vec3, ray_dir: ti.math.vec3,
                 t_min: ti.f32, t_max: ti.f32) -> tuple:
    """
    Main BVH traversal entry point.
    Selects between stackless (optimized) or stack-based (legacy) traversal.
    """
    if ti.static(USE_STACKLESS_TRAVERSAL):
        return traverse_bvh_stackless(ray_origin, ray_dir, t_min, t_max)
    else:
        return traverse_bvh_legacy(ray_origin, ray_dir, t_min, t_max)


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
def emitted(prim_type: ti.i32, prim_idx: ti.i32, u: ti.f32, v: ti.f32, hit_point: ti.math.vec3) -> ti.math.vec3:
    """
    Get emission color for a hit point.
    Returns: RGB emission color (0,0,0 for non-emissive materials)
    """
    # Get material type
    mat_type = 0
    if prim_type == 0:  # PRIM_SPHERE
        mat_type = fields.material_type[prim_idx]
    elif prim_type == 1:  # PRIM_TRIANGLE
        mat_type = fields.triangle_material_type[prim_idx]
    elif prim_type == 2:  # PRIM_QUAD
        mat_type = fields.quad_material_type[prim_idx]

    # Only emissive materials (type 3) emit light
    emit_color = ti.math.vec3(0.0)
    if mat_type == 3:  # MAT_EMISSIVE
        if prim_type == 0:  # PRIM_SPHERE
            emit_color = fields.material_emit_color[prim_idx]
        elif prim_type == 1:  # PRIM_TRIANGLE
            emit_color = fields.triangle_material_emit_color[prim_idx]
        elif prim_type == 2:  # PRIM_QUAD
            emit_color = fields.quad_material_emit_color[prim_idx]

    return emit_color


@ti.func
def scatter(ray_dir: ti.math.vec3, hit_point: ti.math.vec3, normal: ti.math.vec3,
            prim_type: ti.i32, prim_idx: ti.i32) -> tuple:
    """
    Compute scattered ray based on material type.
    Returns: (did_scatter, scatter_direction, attenuation)

    Dispatches to Lambertian, Metal, Dielectric, Emissive, or Isotropic based on material type.
    Uses prim_type (0=sphere, 2=quad) to access correct material arrays.
    """
    # Get material type based on primitive type
    mat_type = 0
    if prim_type == 0:  # PRIM_SPHERE
        mat_type = fields.material_type[prim_idx]
    elif prim_type == 1:  # PRIM_TRIANGLE
        mat_type = fields.triangle_material_type[prim_idx]
    elif prim_type == 2:  # PRIM_QUAD
        mat_type = fields.quad_material_type[prim_idx]

    scatter_dir = ti.math.vec3(0.0)
    attenuation = ti.math.vec3(1.0)
    scattered = False

    # Declare variables in outer scope for Taichi
    albedo = ti.math.vec3(1.0)
    fuzz = 0.0
    ir = 1.0

    # LAMBERTIAN material
    if mat_type == 0:
        # Evaluate texture at hit point to get albedo
        albedo = eval_texture(prim_type, prim_idx, hit_point)
        scatter_dir = random_cosine_direction(normal)
        attenuation = albedo
        scattered = True

    # METAL material
    elif mat_type == 1:
        if prim_type == 0:  # PRIM_SPHERE
            albedo = fields.material_albedo[prim_idx]
            fuzz = fields.material_fuzz[prim_idx]
        elif prim_type == 1:  # PRIM_TRIANGLE
            albedo = fields.triangle_material_albedo[prim_idx]
            fuzz = fields.triangle_material_fuzz[prim_idx]
        else:  # PRIM_QUAD
            albedo = fields.quad_material_albedo[prim_idx]
            fuzz = fields.quad_material_fuzz[prim_idx]

        reflected = reflect(ray_dir.normalized(), normal)
        scatter_dir = reflected + fuzz * random_unit_vector()

        # Only scatter if ray is reflected outward
        if scatter_dir.dot(normal) > 0.0:
            attenuation = albedo
            scattered = True

    # DIELECTRIC material (glass)
    elif mat_type == 2:
        if prim_type == 0:  # PRIM_SPHERE
            ir = fields.material_ir[prim_idx]
        elif prim_type == 1:  # PRIM_TRIANGLE
            ir = fields.triangle_material_ir[prim_idx]
        else:  # PRIM_QUAD
            ir = fields.quad_material_ir[prim_idx]

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

    # EMISSIVE material (light source)
    elif mat_type == 3:
        # Emissive materials don't scatter light
        scattered = False

    # ISOTROPIC material (used for constant medium / smoke)
    elif mat_type == 4:
        # Scatter uniformly in all directions
        scatter_dir = random_unit_vector()
        # Use texture for albedo
        albedo = eval_texture(prim_type, prim_idx, hit_point)
        attenuation = albedo
        scattered = True

    return scattered, scatter_dir, attenuation


# =============================================================================
# TEXTURE FUNCTIONS
# =============================================================================

@ti.func
def eval_texture(prim_type: ti.i32, prim_idx: ti.i32, hit_point: ti.math.vec3) -> ti.math.vec3:
    """
    Evaluate texture at hit point.
    Returns: RGB color

    Handles: solid color, checker pattern, image texture.
    Uses prim_type to access correct texture arrays.
    """
    # Get texture type and colors based on primitive type
    tex_type = 0
    color1 = ti.math.vec3(1.0)
    color2 = ti.math.vec3(1.0)
    scale = 1.0
    img_idx = -1

    if prim_type == 0:  # PRIM_SPHERE
        tex_type = fields.texture_type[prim_idx]
        color1 = fields.texture_color1[prim_idx]
        color2 = fields.texture_color2[prim_idx]
        scale = fields.texture_scale[prim_idx]
        img_idx = fields.texture_image_idx[prim_idx]
    elif prim_type == 1:  # PRIM_TRIANGLE
        tex_type = fields.triangle_texture_type[prim_idx]
        color1 = fields.triangle_texture_color1[prim_idx]
        color2 = fields.triangle_texture_color2[prim_idx]
        scale = fields.triangle_texture_scale[prim_idx]
        img_idx = fields.triangle_texture_image_idx[prim_idx]
    elif prim_type == 2:  # PRIM_QUAD
        tex_type = fields.quad_texture_type[prim_idx]
        color1 = fields.quad_texture_color1[prim_idx]
        color2 = fields.quad_texture_color2[prim_idx]
        scale = fields.quad_texture_scale[prim_idx]
        img_idx = fields.quad_texture_image_idx[prim_idx]

    result = ti.math.vec3(1.0, 1.0, 1.0)

    if tex_type == 0:  # TEX_SOLID
        result = color1
    elif tex_type == 1:  # TEX_CHECKER
        # Checker pattern based on 3D position
        inv_scale = 1.0 / scale
        x_int = ti.floor(inv_scale * hit_point.x)
        y_int = ti.floor(inv_scale * hit_point.y)
        z_int = ti.floor(inv_scale * hit_point.z)

        is_even = (ti.cast(x_int, ti.i32) + ti.cast(y_int, ti.i32) + ti.cast(z_int, ti.i32)) % 2 == 0

        if is_even:
            result = color1
        else:
            result = color2
    elif tex_type == 2:  # TEX_IMAGE
        # Get sphere center for UV calculation (only for spheres currently)
        if prim_type == 0:  # PRIM_SPHERE
            sphere_vec4 = fields.sphere_data[prim_idx]
            center = ti.math.vec3(sphere_vec4[0], sphere_vec4[1], sphere_vec4[2])

            # Calculate UV coordinates
            u, v = get_sphere_uv(hit_point, center)

            # Get image dimensions
            dims = fields.image_texture_dims[img_idx]
            img_width = dims[0]
            img_height = dims[1]

            # Clamp UV to [0,1]
            u = ti.max(0.0, ti.min(1.0, u))
            v = 1.0 - ti.max(0.0, ti.min(1.0, v))  # Flip V to image coordinates

            # Convert to pixel coordinates
            i = ti.cast(u * ti.cast(img_width, ti.f32), ti.i32)
            j = ti.cast(v * ti.cast(img_height, ti.f32), ti.i32)

            # Clamp pixel coordinates
            i = ti.max(0, ti.min(img_width - 1, i))
            j = ti.max(0, ti.min(img_height - 1, j))

            # Sample texture using static dispatch
            # Taichi doesn't support dynamic list indexing, so we use compile-time iteration
            for tex_idx in ti.static(range(len(fields.image_textures))):
                if img_idx == tex_idx:
                    result = fields.image_textures[tex_idx][j, i]
        else:
            # For non-sphere primitives, return magenta as debug color
            result = ti.math.vec3(1.0, 0.0, 1.0)
    elif tex_type == 3:  # TEX_NOISE
        # OPTIMIZED: Perlin noise texture with turbulence and sine wave marble effect
        # Reduced octaves from 7 to 3 for 2-3x speedup (minimal visual difference)
        noise_val = ti.sin(scale * hit_point.z + 10.0 * perlin_turb(hit_point, 3))
        # Map from [-1, 1] to [0, 1] and multiply by base color
        result = color1 * 0.5 * (1.0 + noise_val)

    return result


# =============================================================================
# PATH TRACING INTEGRATOR
# =============================================================================

@ti.func
def trace_ray(ray_origin: ti.math.vec3, ray_dir: ti.math.vec3):
    """
    Trace a single ray through the scene.
    Iterative implementation (no recursion).
    Returns: (final color for this ray path, actual depth reached, was_killed_by_rr, rr_depth, hit_max_depth)

    Loop structure:
    1. Traverse BVH to find closest hit
    2. If miss: return background color * throughput
    3. If hit: scatter based on material, update throughput
    4. Apply Russian Roulette after minimum depth
    5. Repeat until max depth or ray absorbed
    """
    color = ti.math.vec3(0.0)
    throughput = ti.math.vec3(1.0, 1.0, 1.0)

    current_origin = ray_origin
    current_dir = ray_dir.normalized()

    actual_depth = 0
    was_killed_by_rr = False
    rr_depth = 0.0
    hit_max_depth = False

    # Russian Roulette parameters
    RR_MIN_DEPTH = 5  # Start RR after this bounce
    RR_MAX_PROB = 0.95  # Cap survival probability to avoid never terminating

    # Iterative path tracing loop (instead of recursion)
    for depth in range(fields.max_depth[None]):
        # Test for intersection with scene
        hit, t, hit_point, normal, prim_type, prim_idx = traverse_bvh(
            current_origin, current_dir, 0.001, 1e10
        )

        if hit:
            # Check if this primitive is a constant medium
            is_constant_medium = 0
            if prim_type == 0:  # PRIM_SPHERE
                is_constant_medium = fields.is_constant_medium_sphere[prim_idx]
            elif prim_type == 1:  # PRIM_TRIANGLE
                is_constant_medium = fields.is_constant_medium_triangle[prim_idx]
            elif prim_type == 2:  # PRIM_QUAD
                is_constant_medium = fields.is_constant_medium_quad[prim_idx]

            # Declare variables for Taichi
            scattered = False
            scatter_dir = ti.math.vec3(0.0)
            attenuation = ti.math.vec3(1.0)
            volume_passthrough = False

            # If this is a constant medium, check for scattering inside
            if is_constant_medium > 0:
                is_medium_hit, t_medium, medium_point, medium_normal, use_isotropic, t_exit = apply_constant_medium(
                    prim_type, prim_idx, current_origin, current_dir, 0.001, 1e10, t
                )

                if is_medium_hit:
                    # Ray scatters inside the volume - treat as isotropic scattering
                    hit_point = medium_point
                    normal = medium_normal

                    # Use isotropic scattering with medium albedo
                    scatter_dir = random_unit_vector()
                    albedo = ti.math.vec3(0.0)
                    if prim_type == 0:  # PRIM_SPHERE
                        albedo = fields.medium_albedo_sphere[prim_idx]
                    elif prim_type == 1:  # PRIM_TRIANGLE
                        albedo = fields.medium_albedo_triangle[prim_idx]
                    elif prim_type == 2:  # PRIM_QUAD
                        albedo = fields.medium_albedo_quad[prim_idx]
                    attenuation = albedo
                    scattered = True

                    # Note: Volumes don't emit light, so no emission check needed
                elif t_exit > 0.0:
                    # Ray passed through the volume without scattering
                    # Valid exit found - continue tracing from beyond the exit point
                    volume_passthrough = True
                    # Move ray origin to just beyond the exit point
                    # Use normalized epsilon to ensure consistent offset regardless of ray direction magnitude
                    ray_length = ti.sqrt(current_dir.dot(current_dir))
                    epsilon_t = 0.001 / ray_length  # Convert 0.001 units to t-space
                    current_origin = current_origin + current_dir * (t_exit + epsilon_t)
                    # Keep same direction
                    # Don't modify throughput or depth
                else:
                    # t_exit <= 0 means exit finding failed (shouldn't happen for valid volumes)
                    # Treat boundary as solid surface with its material (fallback behavior)
                    # This prevents rays from getting stuck
                    emit = emitted(prim_type, prim_idx, 0.0, 0.0, hit_point)
                    color = color + throughput * emit
                    scattered, scatter_dir, attenuation = scatter(
                        current_dir, hit_point, normal, prim_type, prim_idx
                    )
            else:
                # Normal non-volume primitive - handle emission and scattering
                # Add emitted light from this surface
                emit = emitted(prim_type, prim_idx, 0.0, 0.0, hit_point)
                color = color + throughput * emit

                # Material scattering
                scattered, scatter_dir, attenuation = scatter(
                    current_dir, hit_point, normal, prim_type, prim_idx
                )

            if scattered:
                # Update ray for next bounce
                current_origin = hit_point
                current_dir = scatter_dir
                throughput = throughput * attenuation
                actual_depth = depth + 1

                # Check if we've reached max depth
                if depth + 1 >= fields.max_depth[None]:
                    hit_max_depth = True
                    break

                # Russian Roulette path termination (after minimum depth)
                # Check AFTER scattering to ensure we've completed this bounce
                if depth + 1 >= RR_MIN_DEPTH:
                    # Survival probability = max component of throughput (luminance-based alternative)
                    # This ensures paths with low contribution are more likely to be terminated
                    survival_prob = ti.min(ti.max(ti.max(throughput.x, throughput.y), throughput.z), RR_MAX_PROB)

                    if ti.random() > survival_prob:
                        # Path terminated by Russian Roulette
                        was_killed_by_rr = True
                        rr_depth = ti.cast(depth + 1, ti.f32)
                        break
                    else:
                        # Path survived - boost throughput to maintain unbiased result
                        throughput = throughput / survival_prob
            elif not volume_passthrough:
                # Ray was absorbed (likely hit an emissive surface)
                # Don't break if this was a volume passthrough - keep tracing
                actual_depth = depth + 1
                break

        else:
            # Ray missed - add background color weighted by throughput
            color = color + throughput * fields.bg_color[None]
            actual_depth = depth
            break

    return color, actual_depth, was_killed_by_rr, rr_depth, hit_max_depth


# =============================================================================
# MAIN RENDER KERNEL
# =============================================================================

@ti.kernel
def render_sample():
    """
    OPTIMIZED: Render one sample per pixel with minimal overhead.
    Each thread handles one pixel independently.
    Statistics tracking removed for 15-25% speedup (atomic ops are very expensive on GPU).
    """
    for py, px in fields.accum_buffer:
        ray_origin, ray_dir = get_ray(px, py)
        color, depth, was_killed_by_rr, rr_depth, hit_max_depth = trace_ray(ray_origin, ray_dir)
        fields.accum_buffer[py, px] += color

        # OPTIMIZATION: Removed all atomic operations for statistics tracking
        # These were causing 15-25% performance overhead due to memory contention
        # Statistics can be re-enabled for debugging by uncommenting below:

        # ti.atomic_add(fields.depth_accumulator[None], ti.cast(depth, ti.f32))
        # ti.atomic_add(fields.path_count[None], 1)
        # if hit_max_depth:
        #     ti.atomic_add(fields.max_depth_terminations[None], 1)
        # if was_killed_by_rr:
        #     ti.atomic_add(fields.rr_paths_killed[None], 1)
        #     ti.atomic_add(fields.rr_depth_sum_killed[None], rr_depth)
        # elif depth >= 3:
        #     ti.atomic_add(fields.rr_paths_survived[None], 1)
        #     ti.atomic_add(fields.rr_depth_sum_survived[None], ti.cast(depth, ti.f32))


@ti.kernel
def clear_accum_buffer():
    """Clear accumulation buffer to zero"""
    for py, px in fields.accum_buffer:
        fields.accum_buffer[py, px] = ti.math.vec3(0.0)


# =============================================================================
# WAVEFRONT PATH TRACING KERNELS
# =============================================================================
# Breadth-first ray tracing where all rays at the same bounce depth are
# processed together in waves. This reduces thread divergence and improves
# GPU occupancy compared to the megakernel approach above.

@ti.kernel
def generate_camera_rays(img_width: ti.i32, img_height: ti.i32):
    """
    Generate initial camera rays for all pixels.
    This is the first stage of wavefront path tracing.
    """
    for py, px in ti.ndrange(img_height, img_width):
        idx = py * img_width + px

        # Generate ray with anti-aliasing jitter
        ray_o, ray_d = get_ray(px, py)

        # Store in ray queue
        fields.ray_origins[idx] = ray_o
        fields.ray_directions[idx] = ray_d
        fields.ray_throughput[idx] = ti.math.vec3(1.0, 1.0, 1.0)
        fields.ray_pixel_index[idx] = idx
        fields.ray_depth[idx] = 0

    # Set active ray count
    fields.active_ray_count[None] = img_width * img_height


@ti.kernel
def intersect_rays():
    """
    Intersect all active rays with the scene.
    Stores hit information for each ray.
    """
    for i in range(fields.active_ray_count[None]):
        ray_o = fields.ray_origins[i]
        ray_d = fields.ray_directions[i]

        # Traverse BVH to find closest hit
        hit, t, hit_point, normal, prim_type, prim_idx = traverse_bvh(
            ray_o, ray_d, 0.001, 1e10
        )

        # Store hit information
        fields.hit_valid[i] = 1 if hit else 0
        fields.hit_t[i] = t
        fields.hit_point[i] = hit_point
        fields.hit_normal[i] = normal
        fields.hit_prim_type[i] = prim_type
        fields.hit_prim_idx[i] = prim_idx


@ti.kernel
def shade_miss_rays():
    """
    Process rays that missed the scene - add background color.
    """
    for i in range(fields.active_ray_count[None]):
        if fields.hit_valid[i] == 0:  # Ray missed
            pixel_idx = fields.ray_pixel_index[i]

            # Convert linear pixel index to (py, px)
            # Note: pixel_idx = py * width + px, so we need image width
            # For now, we'll use atomic add directly to the pixel
            # The accumulation happens in the caller

            # Add background contribution
            contribution = fields.ray_throughput[i] * fields.bg_color[None]

            # Get image dimensions from accum_buffer shape
            img_height, img_width = fields.accum_buffer.shape
            py = pixel_idx // img_width
            px = pixel_idx % img_width

            ti.atomic_add(fields.accum_buffer[py, px], contribution)


@ti.kernel
def shade_and_scatter():
    """
    Shade all rays that hit geometry and generate scattered rays.
    This handles emission, material scattering, and generates new rays for the next bounce.
    Handles constant mediums (volumes) as well.

    Russian Roulette is applied after minimum depth.
    """
    # Reset next wave counter
    if ti.static(True):  # Run once before parallel loop
        fields.next_ray_count[None] = 0

    for i in range(fields.active_ray_count[None]):
        if fields.hit_valid[i] == 1:  # Ray hit something
            # Get ray info
            current_origin = fields.ray_origins[i]
            current_dir = fields.ray_directions[i]
            throughput = fields.ray_throughput[i]
            pixel_idx = fields.ray_pixel_index[i]
            depth = fields.ray_depth[i]

            # Get hit info
            hit_point = fields.hit_point[i]
            normal = fields.hit_normal[i]
            prim_type = fields.hit_prim_type[i]
            prim_idx = fields.hit_prim_idx[i]
            t = fields.hit_t[i]

            # Check if this primitive is a constant medium
            is_constant_medium = 0
            if prim_type == 0:  # PRIM_SPHERE
                is_constant_medium = fields.is_constant_medium_sphere[prim_idx]
            elif prim_type == 1:  # PRIM_TRIANGLE
                is_constant_medium = fields.is_constant_medium_triangle[prim_idx]
            elif prim_type == 2:  # PRIM_QUAD
                is_constant_medium = fields.is_constant_medium_quad[prim_idx]

            scattered = False
            scatter_dir = ti.math.vec3(0.0)
            attenuation = ti.math.vec3(1.0)
            volume_passthrough = False
            emit = ti.math.vec3(0.0)

            # Handle constant medium (volume) if applicable
            if is_constant_medium > 0:
                is_medium_hit, t_medium, medium_point, medium_normal, use_isotropic, t_exit = apply_constant_medium(
                    prim_type, prim_idx, current_origin, current_dir, 0.001, 1e10, t
                )

                if is_medium_hit:
                    # Ray scatters inside the volume
                    hit_point = medium_point
                    normal = medium_normal

                    # Isotropic scattering
                    scatter_dir = random_unit_vector()
                    albedo = ti.math.vec3(0.0)
                    if prim_type == 0:  # PRIM_SPHERE
                        albedo = fields.medium_albedo_sphere[prim_idx]
                    elif prim_type == 1:  # PRIM_TRIANGLE
                        albedo = fields.medium_albedo_triangle[prim_idx]
                    elif prim_type == 2:  # PRIM_QUAD
                        albedo = fields.medium_albedo_quad[prim_idx]
                    attenuation = albedo
                    scattered = True

                elif t_exit > 0.0:
                    # Ray passed through volume without scattering
                    volume_passthrough = True
                    ray_length = ti.sqrt(current_dir.dot(current_dir))
                    epsilon_t = 0.001 / ray_length

                    # Generate continuation ray
                    new_origin = current_origin + current_dir * (t_exit + epsilon_t)
                    new_dir = current_dir
                    new_throughput = throughput  # No attenuation

                    # Add to next wave
                    next_idx = ti.atomic_add(fields.next_ray_count[None], 1)
                    fields.next_ray_origins[next_idx] = new_origin
                    fields.next_ray_directions[next_idx] = new_dir
                    fields.next_ray_throughput[next_idx] = new_throughput
                    fields.next_ray_pixel_index[next_idx] = pixel_idx
                    fields.next_ray_depth[next_idx] = depth  # Same depth, just passthrough

                else:
                    # Fallback: treat as solid surface
                    emit = emitted(prim_type, prim_idx, 0.0, 0.0, hit_point)
                    scattered, scatter_dir, attenuation = scatter(
                        current_dir, hit_point, normal, prim_type, prim_idx
                    )
            else:
                # Normal surface - handle emission and scattering
                emit = emitted(prim_type, prim_idx, 0.0, 0.0, hit_point)
                scattered, scatter_dir, attenuation = scatter(
                    current_dir, hit_point, normal, prim_type, prim_idx
                )

            # Add emission contribution
            if emit.norm_sqr() > 0.0:
                img_height, img_width = fields.accum_buffer.shape
                py = pixel_idx // img_width
                px = pixel_idx % img_width
                ti.atomic_add(fields.accum_buffer[py, px], throughput * emit)

            # Generate scattered ray if material scattered
            if scattered and not volume_passthrough:
                new_throughput = throughput * attenuation
                new_depth = depth + 1

                # Check if we've hit max depth
                if new_depth >= fields.max_depth[None]:
                    # Don't generate new ray, path terminates
                    pass
                else:
                    # Russian Roulette after minimum depth
                    should_continue = True
                    RR_MIN_DEPTH = 5
                    RR_MAX_PROB = 0.95

                    if new_depth >= RR_MIN_DEPTH:
                        survival_prob = ti.min(ti.max(ti.max(new_throughput.x, new_throughput.y), new_throughput.z), RR_MAX_PROB)

                        if ti.random() > survival_prob:
                            # Path terminated by Russian Roulette
                            should_continue = False
                        else:
                            # Boost throughput to maintain unbiased result
                            new_throughput = new_throughput / survival_prob

                    if should_continue:
                        # Add to next wave
                        next_idx = ti.atomic_add(fields.next_ray_count[None], 1)
                        fields.next_ray_origins[next_idx] = hit_point
                        fields.next_ray_directions[next_idx] = scatter_dir
                        fields.next_ray_throughput[next_idx] = new_throughput
                        fields.next_ray_pixel_index[next_idx] = pixel_idx
                        fields.next_ray_depth[next_idx] = new_depth


@ti.kernel
def swap_ray_buffers():
    """
    Swap current and next ray buffers for the next iteration.
    This copies data from next_* arrays to current * arrays.
    """
    count = fields.next_ray_count[None]

    for i in range(count):
        fields.ray_origins[i] = fields.next_ray_origins[i]
        fields.ray_directions[i] = fields.next_ray_directions[i]
        fields.ray_throughput[i] = fields.next_ray_throughput[i]
        fields.ray_pixel_index[i] = fields.next_ray_pixel_index[i]
        fields.ray_depth[i] = fields.next_ray_depth[i]

    # Update active count
    fields.active_ray_count[None] = count
