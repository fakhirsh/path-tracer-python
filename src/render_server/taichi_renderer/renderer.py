"""
TaichiRenderer: Main GPU path tracer orchestration.
Thin class that coordinates compilation, rendering, and output.
"""

import taichi as ti
import numpy as np
import time
import os
from PIL import Image
from typing import Optional

# Initialize Taichi ONCE at module load
# Set TAICHI_KERNEL_PROFILER=1 to enable kernel profiling
ENABLE_PROFILER = os.environ.get('TAICHI_KERNEL_PROFILER', '0') == '1'
ti.init(arch=ti.metal, kernel_profiler=ENABLE_PROFILER)

from . import fields
from . import kernels
from .scene_compiler import compile_scene
from .bvh_compiler import compile_bvh
from .preview import LivePreview


class TaichiRenderer:
    """
    GPU-accelerated path tracer using Taichi.

    Usage:
        renderer = TaichiRenderer(world, camera, "output.png")
        renderer.render(enable_preview=True)
    """

    def __init__(self, world, cam, img_path: str):
        """
        Initialize renderer and compile scene to GPU.

        Args:
            world: Scene object (hittable hierarchy with BVH)
            cam: Camera object
            img_path: Output image path (PNG format)
        """
        # Store timing breakdown
        self.timing = {
            'taichi_init': 0.0,  # Compatibility with old API
            'scene_compile': 0.0,
            'bvh_compile': 0.0,
            'bvh_flatten': 0.0,  # Alias for bvh_compile (compatibility)
            'camera_upload': 0.0,
            'gpu_upload': 0.0,
            'kernel_warmup': 0.0,
            'total_setup': 0.0,
        }
        self.sample_times = []

        # Rendering state (for InteractiveViewer compatibility)
        self.current_sample = 0
        self.render_start_time = 0.0

        setup_start = time.time()

        # Store config
        self.img_path = img_path
        self.cam = cam
        self.cam.initialize()

        # Render settings
        self.max_depth = 50
        self.background_color = (0.70, 0.80, 1.00)

        # Allocate dynamic fields
        fields.allocate_dynamic_fields(self.cam.img_width, self.cam.img_height)

        # Allocate wavefront fields (for wavefront path tracing mode)
        fields.allocate_wavefront_fields(self.cam.img_width, self.cam.img_height, self.max_depth)

        # Initialize Perlin noise (only needs to be done once)
        from core.perlin import perlin
        perlin_instance = perlin()
        fields.init_perlin_noise(perlin_instance)

        # Compile scene
        t0 = time.time()
        geometry_data, material_data, spheres, quad_geometry_data, quad_material_data, quads, triangle_geometry_data, triangle_material_data, triangles, image_texture_registry, image_texture_list = compile_scene(world)
        self.timing['scene_compile'] = time.time() - t0

        # Compile BVH
        t0 = time.time()
        bvh_data = compile_bvh(world, spheres, quads, triangles)
        self.timing['bvh_compile'] = time.time() - t0
        self.timing['bvh_flatten'] = self.timing['bvh_compile']  # Alias for compatibility

        # Upload to GPU
        t0 = time.time()
        self._upload_to_gpu(geometry_data, material_data, quad_geometry_data, quad_material_data, triangle_geometry_data, triangle_material_data, bvh_data, image_texture_list)
        self._upload_camera()
        self.timing['gpu_upload'] = time.time() - t0
        self.timing['camera_upload'] = self.timing['gpu_upload']  # Alias for compatibility

        self.timing['total_setup'] = time.time() - setup_start

    def _upload_to_gpu(self, geometry: dict, materials: dict, quad_geometry: dict, quad_materials: dict, triangle_geometry: dict, triangle_materials: dict, bvh: dict, image_textures: list):
        """Upload compiled numpy arrays to Taichi fields"""
        import taichi as ti
        import numpy as np

        # Sphere Geometry
        n = geometry['num_spheres']
        for i in range(n):
            fields.sphere_data[i] = geometry['sphere_data'][i]
        fields.num_spheres[None] = n

        # Sphere Materials
        for i in range(n):
            fields.material_type[i] = materials['material_type'][i]
            fields.material_albedo[i] = materials['material_albedo'][i]
            fields.material_fuzz[i] = materials['material_fuzz'][i]
            fields.material_ir[i] = materials['material_ir'][i]
            fields.material_emit_color[i] = materials['material_emit_color'][i]

        # Sphere Textures
        for i in range(n):
            fields.texture_type[i] = materials['texture_type'][i]
            fields.texture_scale[i] = materials['texture_scale'][i]
            fields.texture_color1[i] = materials['texture_color1'][i]
            fields.texture_color2[i] = materials['texture_color2'][i]
            fields.texture_image_idx[i] = materials['texture_image_idx'][i]

        # Sphere Constant Medium data
        for i in range(n):
            fields.is_constant_medium_sphere[i] = materials['is_constant_medium'][i]
            fields.medium_density_sphere[i] = materials['medium_density'][i]
            fields.medium_albedo_sphere[i] = materials['medium_albedo'][i]

        # Quad Geometry
        nq = quad_geometry['num_quads']
        for i in range(nq):
            fields.quad_Q[i] = quad_geometry['quad_Q'][i]
            fields.quad_u[i] = quad_geometry['quad_u'][i]
            fields.quad_v[i] = quad_geometry['quad_v'][i]
            fields.quad_normal[i] = quad_geometry['quad_normal'][i]
            fields.quad_D[i] = quad_geometry['quad_D'][i]
            fields.quad_w[i] = quad_geometry['quad_w'][i]
        fields.num_quads[None] = nq


        # Quad Materials (separate arrays from spheres)
        for i in range(nq):
            fields.quad_material_type[i] = quad_materials['material_type'][i]
            fields.quad_material_albedo[i] = quad_materials['material_albedo'][i]
            fields.quad_material_fuzz[i] = quad_materials['material_fuzz'][i]
            fields.quad_material_ir[i] = quad_materials['material_ir'][i]
            fields.quad_material_emit_color[i] = quad_materials['material_emit_color'][i]

        # Quad Textures
        for i in range(nq):
            fields.quad_texture_type[i] = quad_materials['texture_type'][i]
            fields.quad_texture_scale[i] = quad_materials['texture_scale'][i]
            fields.quad_texture_color1[i] = quad_materials['texture_color1'][i]
            fields.quad_texture_color2[i] = quad_materials['texture_color2'][i]
            fields.quad_texture_image_idx[i] = quad_materials['texture_image_idx'][i]

        # Quad Constant Medium data
        for i in range(nq):
            fields.is_constant_medium_quad[i] = quad_materials['is_constant_medium'][i]
            fields.medium_density_quad[i] = quad_materials['medium_density'][i]
            fields.medium_albedo_quad[i] = quad_materials['medium_albedo'][i]

        # Triangle Geometry
        nt = triangle_geometry['num_triangles']
        for i in range(nt):
            fields.triangle_v0[i] = triangle_geometry['triangle_v0'][i]
            fields.triangle_v1[i] = triangle_geometry['triangle_v1'][i]
            fields.triangle_v2[i] = triangle_geometry['triangle_v2'][i]
            fields.triangle_edge1[i] = triangle_geometry['triangle_edge1'][i]
            fields.triangle_edge2[i] = triangle_geometry['triangle_edge2'][i]
            fields.triangle_normal[i] = triangle_geometry['triangle_normal'][i]
        fields.num_triangles[None] = nt

        # Triangle Materials
        for i in range(nt):
            fields.triangle_material_type[i] = triangle_materials['material_type'][i]
            fields.triangle_material_albedo[i] = triangle_materials['material_albedo'][i]
            fields.triangle_material_fuzz[i] = triangle_materials['material_fuzz'][i]
            fields.triangle_material_ir[i] = triangle_materials['material_ir'][i]
            fields.triangle_material_emit_color[i] = triangle_materials['material_emit_color'][i]

        # Triangle Textures
        for i in range(nt):
            fields.triangle_texture_type[i] = triangle_materials['texture_type'][i]
            fields.triangle_texture_scale[i] = triangle_materials['texture_scale'][i]
            fields.triangle_texture_color1[i] = triangle_materials['texture_color1'][i]
            fields.triangle_texture_color2[i] = triangle_materials['texture_color2'][i]
            fields.triangle_texture_image_idx[i] = triangle_materials['texture_image_idx'][i]

        # Triangle Constant Medium data
        for i in range(nt):
            fields.is_constant_medium_triangle[i] = triangle_materials['is_constant_medium'][i]
            fields.medium_density_triangle[i] = triangle_materials['medium_density'][i]
            fields.medium_albedo_triangle[i] = triangle_materials['medium_albedo'][i]

        # Image Textures - Upload texture data to GPU
        fields.image_textures.clear()  # Clear previous textures
        for idx, img_tex in enumerate(image_textures):
            if img_tex.image.fdata is not None:
                height, width, _ = img_tex.image.fdata.shape
                # Create Taichi field for this texture
                tex_field = ti.Vector.field(3, ti.f32, shape=(height, width))
                # Upload image data
                tex_field.from_numpy(img_tex.image.fdata)
                # Store field in list
                fields.image_textures.append(tex_field)
                # Store dimensions
                fields.image_texture_dims[idx] = [width, height]

        # BVH - Upload to PACKED structure (optimized format)
        bvh_n = bvh['num_bvh_nodes']
        for i in range(bvh_n):
            node = fields.bvh_nodes[i]
            node.bbox_min = bvh['bvh_bbox_min'][i]
            node.bbox_max = bvh['bvh_bbox_max'][i]
            node.left_child = bvh['bvh_left_child'][i]
            node.right_child = bvh['bvh_right_child'][i]
            node.parent = bvh.get('bvh_parent', [-1] * bvh_n)[i]  # Parent pointer for stackless traversal
            node.prim_type = bvh['bvh_prim_type'][i]
            node.prim_idx = bvh['bvh_prim_idx'][i]

        fields.num_bvh_nodes[None] = bvh_n

    def _upload_camera(self):
        """Upload camera parameters to GPU fields"""
        fields.cam_center[None] = [self.cam.center.x, self.cam.center.y, self.cam.center.z]
        fields.cam_pixel00[None] = [self.cam.pixel00_loc.x, self.cam.pixel00_loc.y, self.cam.pixel00_loc.z]
        fields.cam_delta_u[None] = [self.cam.delta_u.x, self.cam.delta_u.y, self.cam.delta_u.z]
        fields.cam_delta_v[None] = [self.cam.delta_v.x, self.cam.delta_v.y, self.cam.delta_v.z]
        fields.cam_defocus_disk_u[None] = [self.cam.defocus_disk_u.x, self.cam.defocus_disk_u.y, self.cam.defocus_disk_u.z]
        fields.cam_defocus_disk_v[None] = [self.cam.defocus_disk_v.x, self.cam.defocus_disk_v.y, self.cam.defocus_disk_v.z]
        fields.cam_defocus_angle[None] = self.cam.defocus_angle

        # Background color (handle both vec3 and tuple)
        if hasattr(self.background_color, 'x'):
            fields.bg_color[None] = [self.background_color.x, self.background_color.y, self.background_color.z]
        else:
            fields.bg_color[None] = [self.background_color[0], self.background_color[1], self.background_color[2]]

        # Rendering parameters
        fields.max_depth[None] = self.max_depth

    def render_wavefront(self, enable_preview: bool = True):
        """
        Wavefront path tracing render loop.
        Uses breadth-first ray processing instead of depth-first (megakernel).

        This processes all rays at the same bounce depth together in waves,
        reducing thread divergence and improving GPU occupancy.

        Args:
            enable_preview: Show live preview window during rendering
        """
        # Re-upload camera and render settings
        self._upload_camera()

        self._print_setup_info()
        print("Render Mode: WAVEFRONT (breadth-first)")

        # Kernel warmup (JIT compile)
        print("\nWarming up GPU kernels...")
        t0 = time.time()
        # Warm up all wavefront kernels
        kernels.generate_camera_rays(self.cam.img_width, self.cam.img_height)
        ti.sync()
        kernels.intersect_rays()
        ti.sync()
        kernels.shade_miss_rays(self.cam.img_width)
        ti.sync()
        kernels.reset_next_ray_count()
        ti.sync()
        kernels.shade_and_scatter(self.cam.img_width)
        ti.sync()
        kernels.swap_ray_buffers()
        ti.sync()
        kernels.clear_accum_buffer()
        ti.sync()
        fields.reset_depth_stats()
        fields.reset_rr_stats()
        self.timing['kernel_warmup'] = time.time() - t0
        print(f"  Kernel Warmup: {self.timing['kernel_warmup']*1000:6.2f}ms (JIT compilation complete)")

        # Setup preview
        preview = None
        if enable_preview:
            preview = LivePreview(self.cam.img_width, self.cam.img_height)
            preview.start(update_interval_ms=50)

        # Render loop
        print(f"\nRendering Progress:")
        print(f"{'─' * 60}")

        render_start = time.time()

        # Calculate display interval (every 5%)
        display_interval = max(1, self.cam.samples_per_pixel // 20)  # 20 updates = every 5%
        total_pixels = self.cam.img_width * self.cam.img_height

        for sample in range(self.cam.samples_per_pixel):
            sample_start = time.time()

            # Wavefront rendering: process one sample per pixel in waves
            # 1. Generate camera rays for all pixels
            kernels.generate_camera_rays(self.cam.img_width, self.cam.img_height)

            # 2. Wave loop: process rays bounce by bounce
            for bounce in range(self.max_depth):
                # Check if any rays are still active
                active_count = fields.active_ray_count[None]
                if active_count == 0:
                    break

                # Intersect all active rays
                kernels.intersect_rays()

                # Shade misses (add background)
                kernels.shade_miss_rays(self.cam.img_width)

                # Reset next ray counter
                kernels.reset_next_ray_count()

                # Shade hits and generate scattered rays
                kernels.shade_and_scatter(self.cam.img_width)

                # Swap buffers for next iteration
                kernels.swap_ray_buffers()

            ti.sync()

            sample_time = time.time() - sample_start
            self.sample_times.append(sample_time)

            # Progress display (every 5%)
            should_display = (sample + 1) % display_interval == 0 or sample == 0 or sample == self.cam.samples_per_pixel - 1
            if should_display:
                self._print_progress(sample, render_start, total_pixels)

            # Update preview
            if preview:
                accum_np = fields.accum_buffer.to_numpy()
                preview.update(accum_np, sample + 1, self.cam.samples_per_pixel)
                preview.process_events()

        print(f"{'─' * 60}")

        # Write output
        self._write_image()
        self._print_stats()

        # Keep preview open
        if preview:
            print("\n✓ Preview window will stay open. Close it manually when done.")
            preview.finish()

    def render(self, enable_preview: bool = True):
        """
        Main render loop (megakernel mode).
        Uses depth-first ray processing where each thread traces a complete ray path.

        For wavefront (breadth-first) mode, use render_wavefront() instead.

        Args:
            enable_preview: Show live preview window during rendering
        """
        # Re-upload camera and render settings in case they were changed after __init__
        # This ensures background_color and max_depth changes are picked up
        self._upload_camera()

        self._print_setup_info()
        print("Render Mode: MEGAKERNEL (depth-first)")

        # Kernel warmup (JIT compile)
        print("\nWarming up GPU kernels...")
        t0 = time.time()
        kernels.render_sample()
        ti.sync()
        kernels.clear_accum_buffer()
        fields.reset_depth_stats()
        fields.reset_rr_stats()
        self.timing['kernel_warmup'] = time.time() - t0
        print(f"  Kernel Warmup: {self.timing['kernel_warmup']*1000:6.2f}ms (JIT compilation complete)")

        # Setup preview
        preview = None
        if enable_preview:
            preview = LivePreview(self.cam.img_width, self.cam.img_height)
            preview.start(update_interval_ms=50)

        # Render loop
        print(f"\nRendering Progress:")
        print(f"{'─' * 60}")

        render_start = time.time()

        # Calculate display interval (every 5%)
        display_interval = max(1, self.cam.samples_per_pixel // 20)  # 20 updates = every 5%
        total_pixels = self.cam.img_width * self.cam.img_height

        for sample in range(self.cam.samples_per_pixel):
            sample_start = time.time()

            kernels.render_sample()
            ti.sync()

            sample_time = time.time() - sample_start
            self.sample_times.append(sample_time)

            # Progress display (every 5%)
            should_display = (sample + 1) % display_interval == 0 or sample == 0 or sample == self.cam.samples_per_pixel - 1
            if should_display:
                self._print_progress(sample, render_start, total_pixels)

            # Update preview
            if preview:
                accum_np = fields.accum_buffer.to_numpy()
                preview.update(accum_np, sample + 1, self.cam.samples_per_pixel)
                preview.process_events()

        print(f"{'─' * 60}")

        # Write output
        self._write_image()
        self._print_stats()

        # Keep preview open
        if preview:
            print("\n✓ Preview window will stay open. Close it manually when done.")
            preview.finish()

    def _write_image(self):
        """Write final image to PNG file"""
        accum_np = fields.accum_buffer.to_numpy()
        img_array = LivePreview.buffer_to_image(accum_np, self.cam.samples_per_pixel)
        img = Image.fromarray(img_array, mode='RGB')
        img.save(self.img_path)
        print(f"\nImage saved to {self.img_path}")

    def _print_setup_info(self):
        """Print compact setup summary"""
        print(f"\n{self.__class__.__name__}")
        print(f"Resolution: {self.cam.img_width}x{self.cam.img_height} | Samples: {self.cam.samples_per_pixel} | Depth: {self.max_depth}")
        print(f"Spheres: {fields.num_spheres[None]} | BVH Nodes: {fields.num_bvh_nodes[None]}")
        print(f"\nSetup Timing:")
        print(f"  Scene Compile: {self.timing['scene_compile']*1000:6.2f}ms | BVH Compile: {self.timing['bvh_compile']*1000:6.2f}ms")
        print(f"  GPU Upload: {self.timing['gpu_upload']*1000:6.2f}ms | Total Setup: {self.timing['total_setup']*1000:6.2f}ms")

    def _print_progress(self, sample: int, render_start: float, total_pixels: int):
        """Print progress at 5% intervals"""
        elapsed = time.time() - render_start
        avg_time = sum(self.sample_times) / len(self.sample_times)
        throughput = total_pixels / avg_time if avg_time > 0 else 0

        # Calculate ETA
        remaining_samples = self.cam.samples_per_pixel - (sample + 1)
        eta = remaining_samples * avg_time

        # Progress percentage
        progress = (sample + 1) / self.cam.samples_per_pixel * 100

        sample_time = self.sample_times[-1]
        print(f"{sample + 1:4d}/{self.cam.samples_per_pixel} ({progress:5.1f}%) │ "
              f"{sample_time*1000:5.1f}ms │ "
              f"Elapsed: {elapsed:5.1f}s │ "
              f"Throughput: {throughput/1e6:5.2f}M pix/s │ "
              f"ETA: {eta:4.1f}s")

    def _get_average_depth(self):
        """Retrieve average depth from GPU fields"""
        total_depth = fields.depth_accumulator[None]
        total_paths = fields.path_count[None]
        if total_paths > 0:
            return total_depth / total_paths
        return 0.0

    def _get_rr_stats(self):
        """Retrieve Russian Roulette statistics from GPU fields"""
        killed = fields.rr_paths_killed[None]
        survived = fields.rr_paths_survived[None]
        depth_sum_killed = fields.rr_depth_sum_killed[None]
        depth_sum_survived = fields.rr_depth_sum_survived[None]

        total_rr_paths = killed + survived
        kill_rate = (killed / total_rr_paths * 100) if total_rr_paths > 0 else 0.0
        avg_depth_killed = (depth_sum_killed / killed) if killed > 0 else 0.0
        avg_depth_survived = (depth_sum_survived / survived) if survived > 0 else 0.0

        return {
            'killed': killed,
            'survived': survived,
            'total_rr_paths': total_rr_paths,
            'kill_rate': kill_rate,
            'avg_depth_killed': avg_depth_killed,
            'avg_depth_survived': avg_depth_survived
        }

    def _print_depth_stats(self):
        """Print depth statistics in the format similar to CPU renderer"""
        total_paths = fields.path_count[None]
        if total_paths == 0:
            return

        # Calculate average depth
        avg_depth = fields.depth_accumulator[None] / total_paths

        # Get max depth terminations
        max_depth_terms = fields.max_depth_terminations[None]
        max_depth_percentage = (max_depth_terms / total_paths * 100) if total_paths > 0 else 0.0

        # Get Russian Roulette terminations
        rr_terms = fields.rr_paths_killed[None]
        rr_percentage = (rr_terms / total_paths * 100) if total_paths > 0 else 0.0

        print(f"\nDepth Statistics:")
        print(f"  Average path depth: {avg_depth:.2f} (max: {self.max_depth})")
        print(f"  Russian Roulette terminations: {rr_terms:,} ({rr_percentage:.1f}%)")
        print(f"  Max depth terminations: {max_depth_terms:,} ({max_depth_percentage:.1f}%)")
        print(f"  Total paths traced: {total_paths:,}")

    def _print_stats(self):
        """Print final timing and depth statistics"""
        if not self.sample_times:
            return

        total_render_time = sum(self.sample_times)
        avg_sample_time = total_render_time / len(self.sample_times)
        min_sample_time = min(self.sample_times)
        max_sample_time = max(self.sample_times)

        total_pixels = self.cam.img_width * self.cam.img_height
        pixels_per_sec = total_pixels / avg_sample_time if avg_sample_time > 0 else 0
        rays_per_sec = pixels_per_sec * self.max_depth

        print(f"\nPERFORMANCE SUMMARY")
        print(f"Total Render Time: {total_render_time:6.2f}s")
        print(f"Sample Time: Avg {avg_sample_time*1000:5.2f}ms | Min {min_sample_time*1000:5.2f}ms | Max {max_sample_time*1000:5.2f}ms")
        print(f"Throughput: {pixels_per_sec/1e6:5.2f} Mpix/s ({rays_per_sec/1e6:5.2f} Mrays/s)")

        # Print depth statistics in CPU renderer format
        self._print_depth_stats()

    # =========================================================================
    # COMPATIBILITY METHODS (for InteractiveViewer and old API)
    # =========================================================================

    def render_sample(self, sample: int):
        """
        Render one sample (compatibility wrapper for InteractiveViewer).
        The sample parameter is ignored - kernel renders one sample.
        """
        kernels.render_sample()

    def clear_accumulation_buffer(self):
        """Clear accumulation buffer (compatibility wrapper)"""
        kernels.clear_accum_buffer()

    def _upload_camera_to_gpu(self):
        """Upload camera to GPU (already implemented above, kept for clarity)"""
        self._upload_camera()

    def _sync_gpu_to_cpu(self):
        """
        Copy GPU accumulation buffer back to CPU.
        For compatibility with BaseRenderer-based code that expects accum_buffer.
        """
        # This is only needed if code expects a CPU-side accum_buffer
        # Our new implementation works entirely with GPU buffers and numpy
        pass

    def write_image(self):
        """Write image (compatibility wrapper)"""
        self._write_image()

    def print_statistics(self):
        """Print statistics (compatibility wrapper)"""
        self._print_stats()

    @property
    def num_spheres(self):
        """Get number of spheres (compatibility property)"""
        return fields.num_spheres[None]

    @property
    def num_bvh_nodes(self):
        """Get number of BVH nodes (compatibility property)"""
        return fields.num_bvh_nodes[None]

    def setup_live_preview(self, update_interval_ms=500):
        """
        Setup live preview window (compatibility for InteractiveViewer).
        Creates a Tkinter window for displaying render progress.
        """
        import tkinter as tk
        from PIL import ImageTk

        self.update_interval_ms = update_interval_ms
        self.preview_window = tk.Tk()
        self.preview_window.title("Path Tracer - Live Preview")

        # Bring window to front
        self.preview_window.lift()
        self.preview_window.attributes('-topmost', True)
        self.preview_window.after_idle(self.preview_window.attributes, '-topmost', False)

        # Create label to hold the image
        self.preview_label = tk.Label(self.preview_window)
        self.preview_label.pack()

        # Track last update time
        self.last_preview_update = 0.0
        self.preview_photo = None

    def update_preview_if_needed(self):
        """
        Update preview window if enough time has elapsed (compatibility for InteractiveViewer).
        """
        if not hasattr(self, 'preview_window') or self.preview_window is None:
            return

        current_time = time.time()
        if current_time - self.last_preview_update < self.update_interval_ms / 1000.0:
            # Just process events without updating image
            try:
                self.preview_window.update()
            except:
                self.preview_window = None
            return

        self.last_preview_update = current_time

        # Update preview image
        accum_np = fields.accum_buffer.to_numpy()
        img_array = LivePreview.buffer_to_image(accum_np, max(1, self.current_sample))

        from PIL import Image, ImageTk
        img = Image.fromarray(img_array, mode='RGB')
        self.preview_photo = ImageTk.PhotoImage(img)
        self.preview_label.config(image=self.preview_photo)

        try:
            self.preview_window.update()
        except:
            self.preview_window = None
