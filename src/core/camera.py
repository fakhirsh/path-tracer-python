import math
import sys
import time
from core import hittable, hit_record, interval
from util import point3, vec3, color, write_color, Ray, degrees_to_radians, dot, cross, normalize, random_in_unit_disk
from random import random

def format_time(seconds: float) -> str:
    """Format time in seconds to a human-readable string (e.g., '1h 59m 26s' or '0m 56s')"""
    hours = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {mins}m {secs}s"
    else:
        return f"{mins}m {secs}s"

class camera:
    aspect_ratio = 1.0
    img_width = 100
    samples_per_pixel = 10
    max_depth = 10
    background = color(0.70, 0.80, 1.00)

    vfov = 90
    lookfrom = point3(0,0,0)
    lookat = point3(0,0,-1)
    vup = vec3(0,1,0)

    defocus_angle = 0.0
    focus_distance = 10.0

    # Russian Roulette parameters
    russian_roulette_enabled = False
    russian_roulette_min_depth = 3  # Start RR after this depth
    russian_roulette_threshold = 0.1  # Minimum survival probability

    def __init__(self):
        pass

    def initialize(self):
        # Calculate the image height, and ensure that it's at least 1.
        self.img_height = 1 if int(self.img_width / self.aspect_ratio) < 1 else int(self.img_width / self.aspect_ratio)
        self.center = point3(0, 0, 0)

        self.pixel_samples_scale = 1.0 / self.samples_per_pixel

        # Initialize depth tracking statistics
        self.total_path_depth = 0
        self.total_paths = 0
        self.rr_terminations = 0
        self.max_depth_terminations = 0

        self.center = self.lookfrom
        # focal_length = (self.lookfrom - self.lookat).length()
        # Viewport widths less than one are ok since they are real valued.
        theta = degrees_to_radians(self.vfov)
        h = math.tan(theta / 2)
        viewport_height = 2.0 * h * self.focus_distance
        viewport_width = viewport_height * (self.img_width / self.img_height)

        w = normalize(self.lookfrom - self.lookat)
        u = normalize(cross(self.vup, w))
        v = cross(w, u)

        # Calculate the vectors across the horizontal and down the vertical viewport edges.
        viewport_u = viewport_width * u
        viewport_v = viewport_height * -v

        # Calculate the horizontal and vertical delta vectors from pixel to pixel.
        self.delta_u = viewport_u / self.img_width
        self.delta_v = viewport_v / self.img_height
        # Calculate the location of the upper left pixel.
        viewport_upper_left = self.center - (self.focus_distance * w) - viewport_u / 2 - viewport_v / 2
        self.pixel00_loc = viewport_upper_left + 0.5 * (self.delta_u + self.delta_v)

        defocus_radius = self.focus_distance * math.tan(degrees_to_radians(self.defocus_angle) / 2)
        self.defocus_disk_u = defocus_radius * u
        self.defocus_disk_v = defocus_radius * v

    def ray_color(self, r: Ray, depth: int, world: hittable, initial_depth: int = None) -> color:
        # Track initial depth for statistics
        if initial_depth is None:
            initial_depth = depth

        if depth <= 0:
            self.max_depth_terminations += 1
            self.total_path_depth += (initial_depth - depth)
            self.total_paths += 1
            return color(0, 0, 0)

        rec = hit_record()

        if not world.hit(r, interval.from_floats(0.001, float('inf')), rec):
            self.total_path_depth += (initial_depth - depth)
            self.total_paths += 1
            return self.background

        scattered = Ray(point3(0,0,0), vec3(0,0,0))
        attenuation = color(0,0,0)
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

        color_from_scatter = attenuation * self.ray_color(scattered, depth - 1, world, initial_depth)
        return color_from_emission + color_from_scatter
    

    def sample_square(self) -> vec3:
        return vec3(random() - 0.5, random() - 0.5, 0)

    def defocus_disk_sample(self) -> point3:
        p = random_in_unit_disk()
        return self.center + p.x * self.defocus_disk_u + p.y * self.defocus_disk_v

    def get_ray(self, w: int, h: int) -> Ray:
        offset = self.sample_square()
        psample = self.pixel00_loc \
                    + (w + offset.x) * self.delta_u \
                    + (h + offset.y) * self.delta_v
        ray_origin = self.center if self.defocus_angle <= 0.0 else self.defocus_disk_sample()
        ray_direction = psample - ray_origin
        ray_time = random()  # Time can be used for motion blur; here we just use a random time in [0,1)
        return Ray(ray_origin, ray_direction, ray_time)

    def render(self, world: hittable, output_file: str = "image.ppm"):
        self.initialize()

        print(f"Starting render: {self.img_width}x{self.img_height} ({self.samples_per_pixel} samples/pixel, max depth {self.max_depth})", file=sys.stderr)

        start_time = time.time()
        last_time = start_time
        scanline_times = []  # Store recent scanline times
        window_size = 20  # Number of recent scanlines to average

        with open(output_file, 'w') as f:
            f.write(f"P3\n{self.img_width} {self.img_height}\n255\n")
            for h in range(self.img_height):
                for w in range(self.img_width):

                    pcolor = color(0,0,0)
                    for s in range(self.samples_per_pixel):
                        r = self.get_ray(w, h)
                        pcolor += self.ray_color(r, self.max_depth, world)

                    write_color(f, self.pixel_samples_scale * pcolor)

                # Calculate and display progress with windowed moving average
                current_time = time.time()
                scanline_time = current_time - last_time
                last_time = current_time

                # Maintain a sliding window of recent scanline times
                scanline_times.append(scanline_time)
                if len(scanline_times) > window_size:
                    scanline_times.pop(0)

                # Average only recent scanlines
                avg_scanline_time = sum(scanline_times) / len(scanline_times)

                elapsed = current_time - start_time
                scanlines_done = h + 1
                scanlines_remaining = self.img_height - scanlines_done
                estimated_remaining = avg_scanline_time * scanlines_remaining

                elapsed_str = format_time(elapsed)
                eta_str = format_time(estimated_remaining)

                sys.stderr.write(f"\rScanlines remaining: {scanlines_remaining} | Elapsed: {elapsed_str} | ETA: {eta_str}  ")
                sys.stderr.flush()

            # Clear the progress line and show completion message
            elapsed_total = time.time() - start_time
            sys.stderr.write("\r" + " " * 100 + "\r")
            sys.stderr.flush()

            total_str = format_time(elapsed_total)
            print(f"Done. Image saved to {output_file} (Total time: {total_str})", file=sys.stderr)

            # Display depth statistics
            if self.total_paths > 0:
                avg_depth = self.total_path_depth / self.total_paths
                rr_percentage = (self.rr_terminations / self.total_paths) * 100
                max_depth_percentage = (self.max_depth_terminations / self.total_paths) * 100
                print(f"\nDepth Statistics:", file=sys.stderr)
                print(f"  Average path depth: {avg_depth:.2f} (max: {self.max_depth})", file=sys.stderr)
                print(f"  Russian Roulette terminations: {self.rr_terminations:,} ({rr_percentage:.1f}%)", file=sys.stderr)
                print(f"  Max depth terminations: {self.max_depth_terminations:,} ({max_depth_percentage:.1f}%)", file=sys.stderr)
                print(f"  Total paths traced: {self.total_paths:,}", file=sys.stderr)
