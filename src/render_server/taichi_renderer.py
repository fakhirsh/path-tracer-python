import taichi as ti
import numpy as np
import time
from core.hittable import hittable
from core.camera import camera
from core.sphere import Sphere
from core.hittable_list import hittable_list
from core.bvh_node import bvh_node
from render_server.base_renderer import BaseRenderer
from util.color import color
import math

# Initialize Taichi for GPU
ti.init(arch=ti.metal)  # Metal backend for Apple Silicon (M1/M2/M3)


@ti.data_oriented
class TaichiRenderer(BaseRenderer):
    """
    GPU-accelerated path tracer using Taichi.
    Supports: Spheres with Lambertian, Metal, and Dielectric materials.
    """

    # Material type constants
    MAT_LAMBERTIAN = 0
    MAT_METAL = 1
    MAT_DIELECTRIC = 2

    def __init__(self, world: hittable, cam: camera, img_path: str):
        # Configuration
        self.MAX_SPHERES = 2048
        self.MAX_DEPTH = 50
        self.MAX_BVH_NODES = 4096  # BVH tree can have up to 2*N-1 nodes

        # Timing statistics
        self.setup_time = 0.0
        self.sample_times = []

        setup_start = time.time()

        # Initialize Taichi fields BEFORE base class init
        # (because base class calls _compile() which needs these fields)
        self._pre_init_taichi_fields(cam)

        # Initialize base class (sets up camera, accum_buffer, etc.)
        super().__init__(world, cam, img_path)

        self.setup_time = time.time() - setup_start

    def _pre_init_taichi_fields(self, cam: camera):
        """
        Initialize Taichi GPU fields for scene data and rendering.
        Called BEFORE base class init.
        """
        # Need to initialize camera first to get image dimensions
        cam.initialize()

        # Scene data (AoSoA structure)
        self.num_spheres = ti.field(ti.i32, shape=())
        self.sphere_centers = ti.Vector.field(3, ti.f32, self.MAX_SPHERES)
        self.sphere_radii = ti.field(ti.f32, self.MAX_SPHERES)
        self.sphere_center_vel = ti.Vector.field(3, ti.f32, self.MAX_SPHERES)  # for moving spheres

        # Material data (separate fields for each material type)
        self.sphere_mat_type = ti.field(ti.i32, self.MAX_SPHERES)  # 0=lambertian, 1=metal, 2=dielectric
        self.sphere_mat_albedo = ti.Vector.field(3, ti.f32, self.MAX_SPHERES)  # color for lambertian/metal
        self.sphere_mat_fuzz = ti.field(ti.f32, self.MAX_SPHERES)  # fuzz for metal
        self.sphere_mat_ir = ti.field(ti.f32, self.MAX_SPHERES)  # index of refraction for dielectric

        # Camera data on GPU
        self.cam_center = ti.Vector.field(3, ti.f32, shape=())
        self.cam_pixel00 = ti.Vector.field(3, ti.f32, shape=())
        self.cam_delta_u = ti.Vector.field(3, ti.f32, shape=())
        self.cam_delta_v = ti.Vector.field(3, ti.f32, shape=())
        self.cam_defocus_disk_u = ti.Vector.field(3, ti.f32, shape=())
        self.cam_defocus_disk_v = ti.Vector.field(3, ti.f32, shape=())
        self.cam_defocus_angle = ti.field(ti.f32, shape=())

        # Background color
        self.bg_color = ti.Vector.field(3, ti.f32, shape=())

        # Rendering parameters
        self.render_max_depth = ti.field(ti.i32, shape=())

        # Accumulation buffer on GPU
        self.accum_buffer_gpu = ti.Vector.field(3, ti.f32, (cam.img_height, cam.img_width))

        # BVH acceleration structure
        self.num_bvh_nodes = ti.field(ti.i32, shape=())
        self.bvh_bbox_min = ti.Vector.field(3, ti.f32, self.MAX_BVH_NODES)
        self.bvh_bbox_max = ti.Vector.field(3, ti.f32, self.MAX_BVH_NODES)
        self.bvh_left_child = ti.field(ti.i32, self.MAX_BVH_NODES)  # -1 if leaf
        self.bvh_right_child = ti.field(ti.i32, self.MAX_BVH_NODES)  # -1 if leaf
        self.bvh_sphere_idx = ti.field(ti.i32, self.MAX_BVH_NODES)  # sphere index if leaf, -1 otherwise

        # Statistics
        self.total_rays = ti.field(ti.i32, shape=())

    def _compile(self, world: hittable) -> hittable:
        """
        Extract scene data from Python world and pack into GPU-friendly AoSoA format.
        This is called during __init__ before rendering starts.
        """
        print("Compiling scene for Taichi GPU rendering...")

        # Flatten the world hierarchy to extract all spheres
        spheres = self._extract_spheres(world)

        print(f"  Found {len(spheres)} spheres")

        if len(spheres) > self.MAX_SPHERES:
            raise ValueError(f"Scene has {len(spheres)} spheres, but MAX_SPHERES is {self.MAX_SPHERES}")

        # Copy sphere data to GPU
        self.num_spheres[None] = len(spheres)

        for i, sphere in enumerate(spheres):
            # Get center (handle both stationary and moving spheres)
            center1 = sphere.center.at(0.0)  # Center at t=0
            center2 = sphere.center.at(1.0)  # Center at t=1
            velocity = [center2.x - center1.x, center2.y - center1.y, center2.z - center1.z]

            self.sphere_centers[i] = [center1.x, center1.y, center1.z]
            self.sphere_radii[i] = sphere.radius
            self.sphere_center_vel[i] = velocity

            # Extract material type and parameters
            mat = sphere.material
            mat_type_name = type(mat).__name__

            if mat_type_name == 'lambertian':
                self.sphere_mat_type[i] = self.MAT_LAMBERTIAN
                try:
                    mat_color = mat.tex.value(0, 0, center1)
                    self.sphere_mat_albedo[i] = [mat_color.x, mat_color.y, mat_color.z]
                except:
                    self.sphere_mat_albedo[i] = [0.8, 0.8, 0.8]
                self.sphere_mat_fuzz[i] = 0.0
                self.sphere_mat_ir[i] = 1.0

            elif mat_type_name == 'metal':
                self.sphere_mat_type[i] = self.MAT_METAL
                self.sphere_mat_albedo[i] = [mat.albedo.x, mat.albedo.y, mat.albedo.z]
                self.sphere_mat_fuzz[i] = mat.fuzz
                self.sphere_mat_ir[i] = 1.0

            elif mat_type_name == 'dielectric':
                self.sphere_mat_type[i] = self.MAT_DIELECTRIC
                self.sphere_mat_albedo[i] = [1.0, 1.0, 1.0]  # Dielectric is always white
                self.sphere_mat_fuzz[i] = 0.0
                self.sphere_mat_ir[i] = mat.ir

            else:
                # Unsupported material - default to lambertian
                print(f"  Warning: Sphere {i} has unsupported material '{mat_type_name}', using lambertian")
                self.sphere_mat_type[i] = self.MAT_LAMBERTIAN
                self.sphere_mat_albedo[i] = [0.8, 0.8, 0.8]
                self.sphere_mat_fuzz[i] = 0.0
                self.sphere_mat_ir[i] = 1.0

        # Build and upload BVH tree
        self._build_bvh(world)

        print(f"✓ Scene compiled: {len(spheres)} spheres ready on GPU")

        return world  # Return original world (we keep a copy on GPU)

    def _extract_spheres(self, obj) -> list:
        """Recursively extract all Sphere objects from the world hierarchy"""
        spheres = []

        if isinstance(obj, Sphere):
            spheres.append(obj)
        elif isinstance(obj, hittable_list):
            for item in obj.objects:
                spheres.extend(self._extract_spheres(item))
        elif isinstance(obj, bvh_node):
            # BVH nodes contain left and right children
            if obj.left is not None:
                spheres.extend(self._extract_spheres(obj.left))
            if obj.right is not None:
                spheres.extend(self._extract_spheres(obj.right))

        return spheres

    def _build_bvh(self, world: hittable):
        """
        Flatten BVH tree into GPU-friendly arrays.
        Uses depth-first traversal to convert tree into linear arrays.
        """
        print("  Building BVH acceleration structure...")

        # Extract the actual BVH root node from the world
        bvh_root = world
        if isinstance(world, hittable_list) and len(world.objects) > 0:
            # World is a hittable_list containing a BVH node
            bvh_root = world.objects[0]

        # Flatten the BVH tree into arrays
        bvh_nodes = []
        sphere_mapping = {}  # Map sphere objects to their indices

        # Create sphere mapping first
        spheres = self._extract_spheres(world)
        for i, sphere in enumerate(spheres):
            sphere_mapping[id(sphere)] = i

        def flatten_bvh(node, nodes_list):
            """Recursively flatten BVH tree into array format"""
            if node is None:
                return -1

            # Create current node entry
            current_idx = len(nodes_list)

            # Get bounding box
            bbox = node.bounding_box()

            # Check if this is a leaf node (contains a sphere)
            if isinstance(node, Sphere):
                # Leaf node - store sphere index
                sphere_idx = sphere_mapping.get(id(node), -1)
                nodes_list.append({
                    'bbox_min': [bbox.x.min, bbox.y.min, bbox.z.min],
                    'bbox_max': [bbox.x.max, bbox.y.max, bbox.z.max],
                    'left': -1,
                    'right': -1,
                    'sphere_idx': sphere_idx
                })
                return current_idx

            # Internal BVH node - placeholder for now, will update children indices
            nodes_list.append({
                'bbox_min': [bbox.x.min, bbox.y.min, bbox.z.min],
                'bbox_max': [bbox.x.max, bbox.y.max, bbox.z.max],
                'left': -1,
                'right': -1,
                'sphere_idx': -1
            })

            # Recursively flatten children
            left_idx = flatten_bvh(node.left, nodes_list) if hasattr(node, 'left') else -1
            right_idx = flatten_bvh(node.right, nodes_list) if hasattr(node, 'right') else -1

            # Update current node with child indices
            nodes_list[current_idx]['left'] = left_idx
            nodes_list[current_idx]['right'] = right_idx

            return current_idx

        # Flatten the tree starting from the actual BVH root
        flatten_bvh(bvh_root, bvh_nodes)

        if len(bvh_nodes) > self.MAX_BVH_NODES:
            raise ValueError(f"BVH has {len(bvh_nodes)} nodes, but MAX_BVH_NODES is {self.MAX_BVH_NODES}")

        # Upload to GPU
        self.num_bvh_nodes[None] = len(bvh_nodes)
        for i, node in enumerate(bvh_nodes):
            self.bvh_bbox_min[i] = node['bbox_min']
            self.bvh_bbox_max[i] = node['bbox_max']
            self.bvh_left_child[i] = node['left']
            self.bvh_right_child[i] = node['right']
            self.bvh_sphere_idx[i] = node['sphere_idx']

        print(f"  ✓ BVH built: {len(bvh_nodes)} nodes")

    def _upload_camera_to_gpu(self):
        """Copy camera parameters to GPU fields"""
        self.cam_center[None] = [self.cam.center.x, self.cam.center.y, self.cam.center.z]
        self.cam_pixel00[None] = [self.cam.pixel00_loc.x, self.cam.pixel00_loc.y, self.cam.pixel00_loc.z]
        self.cam_delta_u[None] = [self.cam.delta_u.x, self.cam.delta_u.y, self.cam.delta_u.z]
        self.cam_delta_v[None] = [self.cam.delta_v.x, self.cam.delta_v.y, self.cam.delta_v.z]
        self.cam_defocus_disk_u[None] = [self.cam.defocus_disk_u.x, self.cam.defocus_disk_u.y, self.cam.defocus_disk_u.z]
        self.cam_defocus_disk_v[None] = [self.cam.defocus_disk_v.x, self.cam.defocus_disk_v.y, self.cam.defocus_disk_v.z]
        self.cam_defocus_angle[None] = self.cam.defocus_angle

        # Background color
        self.bg_color[None] = [self.background_color.x, self.background_color.y, self.background_color.z]

        # Rendering parameters
        self.render_max_depth[None] = self.max_depth

    @ti.func
    def get_ray(self, px: ti.i32, py: ti.i32) -> tuple:
        """Generate a camera ray for pixel (px, py) with random sampling"""
        # Random offset within pixel for anti-aliasing
        offset_x = ti.random() - 0.5
        offset_y = ti.random() - 0.5

        # Calculate pixel sample position
        pixel_sample = (self.cam_pixel00[None] +
                       (px + offset_x) * self.cam_delta_u[None] +
                       (py + offset_y) * self.cam_delta_v[None])

        # Ray origin (with defocus blur if enabled)
        ray_origin = self.cam_center[None]
        if self.cam_defocus_angle[None] > 0.0:
            # Sample from defocus disk
            p = self.random_in_unit_disk()
            ray_origin = self.cam_center[None] + p.x * self.cam_defocus_disk_u[None] + p.y * self.cam_defocus_disk_v[None]

        ray_direction = pixel_sample - ray_origin
        ray_time = ti.random()  # Random time for motion blur

        return ray_origin, ray_direction, ray_time

    @ti.func
    def random_in_unit_disk(self) -> ti.math.vec3:
        """Generate random point in unit disk (for defocus blur)"""
        result = ti.math.vec3(0.0)
        while True:
            p = ti.math.vec3(ti.random() * 2.0 - 1.0, ti.random() * 2.0 - 1.0, 0.0)
            if p.dot(p) < 1.0:
                result = p
                break
        return result

    @ti.func
    def random_unit_vector(self) -> ti.math.vec3:
        """Generate random unit vector"""
        result = ti.math.vec3(0.0)
        while True:
            p = ti.math.vec3(ti.random() * 2.0 - 1.0, ti.random() * 2.0 - 1.0, ti.random() * 2.0 - 1.0)
            lensq = p.dot(p)
            if lensq < 1.0 and lensq > 1e-20:
                result = p.normalized()
                break
        return result

    @ti.func
    def random_cosine_direction(self, normal: ti.math.vec3) -> ti.math.vec3:
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

    @ti.func
    def reflect(self, v: ti.math.vec3, n: ti.math.vec3) -> ti.math.vec3:
        """Reflect vector v around normal n"""
        return v - 2.0 * v.dot(n) * n

    @ti.func
    def refract(self, uv: ti.math.vec3, n: ti.math.vec3, etai_over_etat: ti.f32) -> ti.math.vec3:
        """Refract vector uv with normal n and ratio etai_over_etat"""
        cos_theta = ti.min(-uv.dot(n), 1.0)
        r_out_perp = etai_over_etat * (uv + cos_theta * n)
        r_out_parallel = -ti.sqrt(ti.abs(1.0 - r_out_perp.dot(r_out_perp))) * n
        return r_out_perp + r_out_parallel

    @ti.func
    def reflectance(self, cosine: ti.f32, ref_idx: ti.f32) -> ti.f32:
        """Schlick's approximation for reflectance"""
        r0 = (1.0 - ref_idx) / (1.0 + ref_idx)
        r0 = r0 * r0
        return r0 + (1.0 - r0) * ti.pow(1.0 - cosine, 5.0)

    @ti.func
    def hit_aabb(self, node_idx: ti.i32, ray_origin: ti.math.vec3, ray_dir: ti.math.vec3,
                 t_min: ti.f32, t_max: ti.f32) -> bool:
        """
        Test if ray hits the AABB of a BVH node.
        Returns: True if hit, False otherwise
        """
        bbox_min = self.bvh_bbox_min[node_idx]
        bbox_max = self.bvh_bbox_max[node_idx]

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
    def hit_sphere(self, sphere_idx: ti.i32, ray_origin: ti.math.vec3, ray_dir: ti.math.vec3,
                   ray_time: ti.f32, t_min: ti.f32, t_max: ti.f32) -> tuple:
        """
        Test ray-sphere intersection.
        Returns: (hit: bool, t: float, hit_point: vec3, normal: vec3)
        """
        # Get sphere center at ray time (for motion blur)
        center = self.sphere_centers[sphere_idx] + ray_time * self.sphere_center_vel[sphere_idx]
        radius = self.sphere_radii[sphere_idx]

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
    def hit_world(self, ray_origin: ti.math.vec3, ray_dir: ti.math.vec3, ray_time: ti.f32,
                  t_min: ti.f32, t_max: ti.f32) -> tuple:
        """
        Test ray against scene using BVH acceleration structure.
        Returns: (hit: bool, t: float, hit_point: vec3, normal: vec3, material_idx: int)
        """
        hit_anything = False
        closest_t = t_max
        closest_hit_point = ti.math.vec3(0.0)
        closest_normal = ti.math.vec3(0.0)
        closest_mat_idx = 0

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
            if node_idx < 0 or node_idx >= self.num_bvh_nodes[None]:
                continue

            # Test ray against node's bounding box
            if not self.hit_aabb(node_idx, ray_origin, ray_dir, t_min, closest_t):
                continue  # Ray misses this node's bbox, skip entire subtree

            # Check if this is a leaf node (contains a sphere)
            sphere_idx = self.bvh_sphere_idx[node_idx]
            if sphere_idx >= 0:
                # Leaf node - test sphere intersection
                hit, t, hit_point, normal = self.hit_sphere(
                    sphere_idx, ray_origin, ray_dir, ray_time, t_min, closest_t
                )

                if hit and t < closest_t:
                    hit_anything = True
                    closest_t = t
                    closest_hit_point = hit_point
                    closest_normal = normal
                    closest_mat_idx = sphere_idx
            else:
                # Internal node - add children to stack
                left_child = self.bvh_left_child[node_idx]
                right_child = self.bvh_right_child[node_idx]

                if right_child >= 0 and stack_ptr < 64:
                    stack[stack_ptr] = right_child
                    stack_ptr += 1

                if left_child >= 0 and stack_ptr < 64:
                    stack[stack_ptr] = left_child
                    stack_ptr += 1

        return hit_anything, closest_t, closest_hit_point, closest_normal, closest_mat_idx

    @ti.func
    def trace_ray(self, ray_origin: ti.math.vec3, ray_dir: ti.math.vec3, ray_time: ti.f32) -> ti.math.vec3:
        """
        Trace a ray through the scene using iterative path tracing.
        Returns the color for this ray.
        """
        color = ti.math.vec3(0.0)
        throughput = ti.math.vec3(1.0, 1.0, 1.0)

        current_origin = ray_origin
        current_dir = ray_dir.normalized()
        current_time = ray_time

        # Iterative path tracing loop (instead of recursion)
        for depth in range(self.render_max_depth[None]):
            # Test for intersection with scene
            hit, t, hit_point, normal, mat_idx = self.hit_world(
                current_origin, current_dir, current_time, 0.001, 1e10
            )

            if hit:
                # Material scattering based on type
                mat_type = self.sphere_mat_type[mat_idx]
                albedo = self.sphere_mat_albedo[mat_idx]
                scatter_dir = ti.math.vec3(0.0)
                scattered = False

                # LAMBERTIAN material
                if mat_type == 0:
                    scatter_dir = self.random_cosine_direction(normal)
                    throughput = throughput * albedo
                    scattered = True

                # METAL material
                elif mat_type == 1:
                    reflected = self.reflect(current_dir.normalized(), normal)
                    fuzz = self.sphere_mat_fuzz[mat_idx]
                    scatter_dir = reflected + fuzz * self.random_unit_vector()

                    # Only scatter if ray is reflected outward
                    if scatter_dir.dot(normal) > 0.0:
                        throughput = throughput * albedo
                        scattered = True
                    else:
                        # Absorbed by metal
                        break

                # DIELECTRIC material (glass)
                elif mat_type == 2:
                    ir = self.sphere_mat_ir[mat_idx]

                    # Determine if ray is entering or exiting
                    # (Check if ray and normal point in similar direction)
                    front_face = current_dir.dot(normal) < 0.0
                    normal_facing = normal if front_face else -normal
                    refraction_ratio = (1.0 / ir) if front_face else ir

                    unit_dir = current_dir.normalized()
                    cos_theta = ti.min(-unit_dir.dot(normal_facing), 1.0)
                    sin_theta = ti.sqrt(1.0 - cos_theta * cos_theta)

                    cannot_refract = refraction_ratio * sin_theta > 1.0

                    # Choose reflection or refraction
                    if cannot_refract or self.reflectance(cos_theta, refraction_ratio) > ti.random():
                        scatter_dir = self.reflect(unit_dir, normal_facing)
                    else:
                        scatter_dir = self.refract(unit_dir, normal_facing, refraction_ratio)

                    # Dielectric doesn't attenuate (always white)
                    scattered = True

                if scattered:
                    # Update ray for next bounce
                    current_origin = hit_point
                    current_dir = scatter_dir
                else:
                    # Ray was absorbed
                    break

            else:
                # Ray missed - add sky color weighted by throughput
                # Sky gradient: blue to white based on y direction
                unit_dir = current_dir.normalized()
                a = 0.5 * (unit_dir.y + 1.0)
                sky = (1.0 - a) * ti.math.vec3(1.0, 1.0, 1.0) + a * self.bg_color[None]
                color = color + throughput * sky
                break

        return color

    @ti.kernel
    def render_sample(self, sample: ti.i32):
        """
        Render one sample per pixel (executed in parallel on GPU).
        Each thread handles one pixel.
        """
        for py, px in self.accum_buffer_gpu:
            # Generate ray for this pixel
            ray_origin, ray_dir, ray_time = self.get_ray(px, py)

            # Trace the ray
            pixel_color = self.trace_ray(ray_origin, ray_dir, ray_time)

            # Accumulate color
            self.accum_buffer_gpu[py, px] += pixel_color

    @ti.kernel
    def clear_accumulation_buffer(self):
        """Clear the GPU accumulation buffer"""
        for py, px in self.accum_buffer_gpu:
            self.accum_buffer_gpu[py, px] = ti.math.vec3(0.0)

    def render(self, enable_preview=True):
        """
        Main rendering loop - launches GPU kernels for each sample.
        """
        print(f"Rendering with {self.__class__.__name__}...")
        print(f"  Resolution: {self.cam.img_width}x{self.cam.img_height}")
        print(f"  Samples: {self.cam.samples_per_pixel}")
        print(f"  Max depth: {self.max_depth}")
        print(f"  Spheres: {self.num_spheres[None]}")

        # Upload camera parameters to GPU
        self._upload_camera_to_gpu()

        # Clear accumulation buffer
        self.clear_accumulation_buffer()

        # Setup live preview if enabled (20 fps = 50ms update interval)
        if enable_preview:
            self.setup_live_preview(update_interval_ms=50)

        # Render loop - launch GPU kernel for each sample
        print(f"\nRendering samples:")
        for sample in range(self.cam.samples_per_pixel):
            self.current_sample = sample + 1

            # Time this sample
            sample_start = time.time()

            # Launch GPU kernel (massively parallel!)
            self.render_sample(sample)

            # Wait for GPU to finish (synchronize)
            ti.sync()

            sample_time = time.time() - sample_start
            self.sample_times.append(sample_time)

            # Print progress every 10 samples
            if (sample + 1) % 10 == 0 or sample == 0:
                avg_time = sum(self.sample_times) / len(self.sample_times)
                print(f"  Sample {sample + 1}/{self.cam.samples_per_pixel}: "
                      f"{sample_time*1000:.2f}ms (avg: {avg_time*1000:.2f}ms)")

            # Process preview window events (no GPU sync needed - preview callback handles it)
            self.update_preview_if_needed()

        # Final sync from GPU to CPU
        self._sync_gpu_to_cpu()

        # Write final image and print stats
        self.write_image()
        self.print_statistics()

        # Print detailed timing statistics
        self._print_timing_stats()

        # Keep preview window open if enabled
        if enable_preview and self.preview_window is not None:
            print("\n✓ Preview window will stay open. Close it manually when done.")
            try:
                self.preview_window.mainloop()  # Keep window alive
            except:
                pass

    def _sync_gpu_to_cpu(self):
        """
        Copy GPU accumulation buffer back to CPU for final output.
        Only called at the end of rendering to write the image file.
        """
        gpu_data = self.accum_buffer_gpu.to_numpy()

        # Convert to Python color objects for compatibility with base class write_image()
        for h in range(self.cam.img_height):
            for w in range(self.cam.img_width):
                self.accum_buffer[h][w] = color(gpu_data[h, w, 0], gpu_data[h, w, 1], gpu_data[h, w, 2])

    def _accum_buffer_to_array(self):
        """
        OPTIMIZED: Convert GPU accumulation buffer directly to displayable numpy array.
        Bypasses the slow Python color object conversion entirely!
        """
        # Get data directly from GPU (this is fast!)
        gpu_data = self.accum_buffer_gpu.to_numpy()

        # Calculate scale based on current sample count
        scale = 1.0 / max(1, self.current_sample)

        # Apply scaling, gamma correction, and convert to uint8 in one go
        # This uses vectorized numpy operations (very fast!)
        scaled = gpu_data * scale

        # Gamma correction (gamma = 2.0)
        gamma_corrected = np.sqrt(np.maximum(0, scaled))

        # Convert to 0-255 range and clamp
        img_array = np.clip(gamma_corrected * 255.999, 0, 255).astype(np.uint8)

        return img_array

    def ray_color(self, r, depth: int, initial_depth: int = None):
        """
        Dummy implementation for BaseRenderer compatibility.
        Actual ray tracing happens in Taichi kernels.
        """
        return color(0, 0, 0)

    def _print_timing_stats(self):
        """Print detailed timing statistics"""
        if not self.sample_times:
            return

        total_render_time = sum(self.sample_times)
        avg_sample_time = total_render_time / len(self.sample_times)
        min_sample_time = min(self.sample_times)
        max_sample_time = max(self.sample_times)

        total_pixels = self.cam.img_width * self.cam.img_height
        pixels_per_sec = total_pixels / avg_sample_time if avg_sample_time > 0 else 0

        print(f"\n{'='*60}")
        print(f"PERFORMANCE STATISTICS")
        print(f"{'='*60}")
        print(f"Setup time:              {self.setup_time*1000:.2f} ms")
        print(f"Total render time:       {total_render_time:.3f} s")
        print(f"Average sample time:     {avg_sample_time*1000:.2f} ms")
        print(f"Min sample time:         {min_sample_time*1000:.2f} ms")
        print(f"Max sample time:         {max_sample_time*1000:.2f} ms")
        print(f"Throughput:              {pixels_per_sec:,.0f} pixels/sec")
        print(f"Rays per second:         {pixels_per_sec * self.max_depth:,.0f} rays/sec (approx)")
        print(f"{'='*60}")
