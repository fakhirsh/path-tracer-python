from core.hittable import hittable, hit_record
from core.interval import interval
from util.color import color
from util.ray import Ray
from random import random
from render_server.base_renderer import BaseRenderer


class GpuRenderer(BaseRenderer):
    """GPU-accelerated renderer implementation"""

    def _compile(self, world: hittable) -> hittable:
        """GPU-specific world compilation (e.g., memory packing, BVH optimization)"""
        print("Compiling world for GPU rendering...")
        print("Returns a memory packed representation of the world.")
        # TODO: Implement actual GPU memory packing and BVH optimization
        # bvh = bvh_node.from_objects(world.objects, 0, len(world.objects))
        # packed_world = pack_for_gpu(bvh)
        return world

    def render(self, enable_preview=True):
        """
        GPU rendering implementation.
        TODO: Replace with actual GPU kernel launches (Taichi/CUDA).
        For now, uses the simple render loop.

        Args:
            enable_preview: Whether to show live preview window (default: True)
        """

        print(f"Rendering with {self.__class__.__name__}...")

        # Setup live preview if enabled
        if enable_preview:
            self.setup_live_preview()

        # Render loop - completely independent of preview updates
        for sample in range(self.cam.samples_per_pixel):
            self.current_sample = sample + 1  # Update progress for preview

            for h in range(self.cam.img_height):
                for w in range(self.cam.img_width):
                    r = self.cam.get_ray(w, h)
                    pcolor = self.ray_color(r, self.max_depth)
                    self.accum_buffer[h][w] += pcolor

            # Process GUI events to allow timer callbacks to execute
            self.update_preview_if_needed()

        # Close preview and write final image
        if enable_preview:
            self.close_preview()

        self.write_image()
        self.print_statistics()

    def ray_color(self, r: Ray, depth: int, initial_depth: int = None) -> color:
        """GPU-specific ray tracing implementation"""
        # Track initial depth for statistics
        if initial_depth is None:
            initial_depth = depth

        if depth <= 0:
            self.max_depth_terminations += 1
            self.total_path_depth += (initial_depth - depth)
            self.total_paths += 1
            return color(0, 0, 0)

        rec = hit_record()

        if not self.world.hit(r, interval.from_floats(0.001, float('inf')), rec):
            self.total_path_depth += (initial_depth - depth)
            self.total_paths += 1
            return self.background_color

        scattered = Ray(rec.p, rec.p)  # Will be modified by scatter
        attenuation = color(0, 0, 0)  # Will be modified by scatter
        color_from_emission = rec.material.emitted(rec.u, rec.v, rec.p)

        if not rec.material.scatter(r, rec, attenuation, scattered):
            self.total_path_depth += (initial_depth - depth)
            self.total_paths += 1
            return color_from_emission

        # Russian Roulette path termination
        if self.russian_roulette_enabled and depth < self.max_depth - self.russian_roulette_min_depth:
            # Calculate survival probability based on attenuation (throughput)
            # Use the maximum component to determine importance
            max_attenuation = max(attenuation.x, attenuation.y, attenuation.z)
            survival_prob = max(self.russian_roulette_threshold, max_attenuation)

            # Randomly terminate the path
            if random() > survival_prob:
                self.rr_terminations += 1
                self.total_path_depth += (initial_depth - depth)
                self.total_paths += 1
                return color_from_emission

            # Scale by inverse probability to maintain unbiased result
            attenuation = attenuation * (1.0 / survival_prob)

        color_from_scatter = attenuation * self.ray_color(scattered, depth - 1, initial_depth)
        return color_from_emission + color_from_scatter