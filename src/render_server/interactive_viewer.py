"""
Interactive Path Tracer Viewer with Mouse-Controlled Camera Rotation

This module provides an interactive viewer that extends TaichiRenderer with:
- Mouse-controlled camera rotation (orbit controls)
- Real-time accumulation buffer reset on camera changes
- Continuous rendering until window is closed
"""

import taichi as ti
import math
from render_server.taichi_renderer import TaichiRenderer
from core.camera import camera
from core.hittable import hittable
from util import vec3, point3


class InteractiveViewer(TaichiRenderer):
    """
    Interactive GPU-accelerated path tracer with orbit camera controls.

    Controls:
        - Left-click + drag: Rotate camera around the lookat point
        - Horizontal drag: Rotate around Y-axis (yaw)
        - Vertical drag: Rotate up/down (pitch)
    """

    def __init__(self, world: hittable, cam: camera, img_path: str):
        super().__init__(world, cam, img_path)

        # Mouse state
        self.mouse_down = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0

        # Throttle camera updates to reduce buffer clear frequency
        self.last_camera_update_time = 0
        self.camera_update_interval = 0.05  # 50ms = 20 updates/sec max

        # Rotation sensitivity: (horizontal_degrees_per_pixel, vertical_degrees_per_pixel)
        self.rotation_velocity = (0.3, 0.3)

        # Camera state in spherical coordinates (cached for smooth rotation)
        self._camera_radius = 0.0
        self._camera_theta = 0.0  # Azimuth angle (horizontal rotation)
        self._camera_phi = 0.0    # Elevation angle (vertical rotation)
        self._initialize_spherical_coords()

        # Rendering state
        self.is_rendering_active = True  # Flag to pause/resume rendering

    def _initialize_spherical_coords(self):
        """
        Convert current camera position to spherical coordinates.
        This allows smooth incremental rotation without accumulating errors.
        """
        # Vector from lookat to lookfrom
        offset = self.cam.lookfrom - self.cam.lookat

        self._camera_radius = math.sqrt(offset.x**2 + offset.y**2 + offset.z**2)

        # Azimuth (theta): angle in XZ plane from -Z axis
        # atan2(x, -z) gives angle from -Z axis, rotating around Y
        self._camera_theta = math.atan2(offset.x, -offset.z)

        # Elevation (phi): angle from XZ plane
        # asin(y / radius) gives angle from horizontal plane
        if self._camera_radius > 0:
            self._camera_phi = math.asin(offset.y / self._camera_radius)
        else:
            self._camera_phi = 0.0

    def _spherical_to_cartesian(self, radius, theta, phi):
        """
        Convert spherical coordinates to Cartesian offset vector.

        Args:
            radius: Distance from origin
            theta: Azimuth angle (horizontal rotation around Y)
            phi: Elevation angle (pitch up/down)

        Returns:
            vec3: Cartesian offset vector
        """
        # Standard spherical to Cartesian conversion
        # x = r * cos(phi) * sin(theta)
        # y = r * sin(phi)
        # z = -r * cos(phi) * cos(theta)  [negative Z is forward]

        cos_phi = math.cos(phi)
        sin_phi = math.sin(phi)
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)

        x = radius * cos_phi * sin_theta
        y = radius * sin_phi
        z = -radius * cos_phi * cos_theta

        return vec3(x, y, z)

    def rotate_camera(self, delta_x, delta_y):
        """
        Rotate camera around lookat point based on mouse movement.

        Args:
            delta_x: Horizontal mouse movement in pixels (positive = right)
            delta_y: Vertical mouse movement in pixels (positive = down)
        """
        # Convert pixel delta to angle delta (in radians)
        theta_delta = math.radians(delta_x * self.rotation_velocity[0])
        phi_delta = math.radians(delta_y * self.rotation_velocity[1])  # Invert Y for intuitive controls

        # Update spherical coordinates
        self._camera_theta += theta_delta
        self._camera_phi += phi_delta

        # Clamp phi to avoid gimbal lock (keep away from poles)
        max_phi = math.radians(89.0)  # Just below vertical
        self._camera_phi = max(-max_phi, min(max_phi, self._camera_phi))

        # Convert back to Cartesian and update camera
        offset = self._spherical_to_cartesian(self._camera_radius, self._camera_theta, self._camera_phi)
        self.cam.lookfrom = self.cam.lookat + offset

        # Reinitialize camera (recomputes u, v, w basis vectors)
        self.cam.initialize()

        # Upload new camera parameters to GPU
        self._upload_camera_to_gpu()

    def restart_rendering(self):
        """
        Reset accumulation buffer and sample counter.
        Called automatically when camera changes.
        The render loop will naturally continue from sample 0.
        """
        import time
        from render_server.taichi_renderer import fields
        self.current_sample = 0
        self.sample_times = []  # Reset timing stats
        self.render_start_time = time.time()  # Reset to current time
        self.clear_accumulation_buffer()
        fields.reset_depth_stats()
        self.is_rendering_active = True  # Resume rendering

        # Print restart notification
        print(f"\n{'─'*60}")
        print("Camera rotated - restarting render from sample 0")
        print(f"{'─'*60}")

    def _print_render_summary(self):
        """Print comprehensive rendering statistics with GPU performance indicators"""
        if not self.sample_times:
            return

        import time
        import statistics

        # Calculate timing statistics
        total_render_time = time.time() - self.render_start_time
        avg_sample_time = statistics.mean(self.sample_times)
        min_sample_time = min(self.sample_times)
        max_sample_time = max(self.sample_times)

        # NEW: Statistical analysis
        median_sample_time = statistics.median(self.sample_times)
        std_dev_sample_time = statistics.stdev(self.sample_times) if len(self.sample_times) > 1 else 0.0
        cv_sample_time = (std_dev_sample_time / avg_sample_time * 100) if avg_sample_time > 0 else 0.0

        # Calculate percentiles for outlier detection
        if len(self.sample_times) >= 20:
            p95_sample_time = statistics.quantiles(self.sample_times, n=20)[18]  # 95th percentile
        else:
            p95_sample_time = max_sample_time

        # Throughput metrics
        total_pixels = self.cam.img_width * self.cam.img_height
        pixels_per_sec = total_pixels / avg_sample_time if avg_sample_time > 0 else 0
        rays_per_sec = pixels_per_sec * self.max_depth
        samples_per_sec = 1.0 / avg_sample_time if avg_sample_time > 0 else 0

        # Get depth statistics
        avg_depth = self._get_average_depth()

        # Print summary
        print(f"\n{'═'*60}")
        print(f"RENDER SUMMARY")
        print(f"{'═'*60}")
        print(f"Resolution:       {self.cam.img_width} x {self.cam.img_height} ({total_pixels:,} pixels)")
        print(f"Samples:          {self.current_sample} / {self.cam.samples_per_pixel}")
        print(f"Max Ray Depth:    {self.max_depth}")
        print(f"Avg Path Depth:   {avg_depth:.2f}")
        print(f"Scene Complexity: {self.num_spheres} spheres, {self.num_bvh_nodes} BVH nodes")

        print(f"\n{'─'*60}")
        print(f"TIMING STATISTICS")
        print(f"{'─'*60}")
        print(f"Total Render Time:  {total_render_time:6.2f}s")
        print(f"Sample Time:")
        print(f"  Mean:             {avg_sample_time*1000:6.2f}ms")
        print(f"  Median:           {median_sample_time*1000:6.2f}ms")
        print(f"  Std Dev:          {std_dev_sample_time*1000:6.2f}ms")
        print(f"  Min:              {min_sample_time*1000:6.2f}ms")
        print(f"  Max:              {max_sample_time*1000:6.2f}ms")
        if len(self.sample_times) >= 20:
            print(f"  95th percentile:  {p95_sample_time*1000:6.2f}ms")

        print(f"\n{'─'*60}")
        print(f"THROUGHPUT")
        print(f"{'─'*60}")
        print(f"Samples/sec:      {samples_per_sec:6.2f} samp/s")
        print(f"Pixels/sec:       {pixels_per_sec/1e6:6.2f} Mpix/s")
        print(f"Rays/sec:         {rays_per_sec/1e6:6.2f} Mrays/s")

        # NEW: GPU Performance Indicators
        print(f"\n{'─'*60}")
        print(f"GPU PERFORMANCE INDICATORS")
        print(f"{'─'*60}")
        print(f"Timing Variability:")
        print(f"  Coefficient of Variation: {cv_sample_time:5.2f}%")

        # Interpret CV (variance measure)
        if cv_sample_time < 5:
            variance_status = "Low (consistent performance)"
        elif cv_sample_time < 15:
            variance_status = "Moderate (some variation)"
        else:
            variance_status = "High (possible divergence/throttling)"

        print(f"  Status: {variance_status}")

        # Detect potential issues
        if max_sample_time > avg_sample_time * 1.5:
            ratio = max_sample_time / avg_sample_time
            print(f"\n⚠️  WARNING: Detected outlier samples (max is {ratio:.1f}x mean)")
            print(f"  This may indicate:")
            print(f"    • Thermal throttling")
            print(f"    • Thread divergence in complex rays")
            print(f"    • System background tasks")

        # Print Taichi profiler info if enabled
        from render_server.taichi_renderer import ENABLE_PROFILER
        if ENABLE_PROFILER:
            print(f"\n{'─'*60}")
            print(f"TAICHI KERNEL PROFILER")
            print(f"{'─'*60}")
            ti.profiler.print_kernel_profiler_info('count')

        print(f"\n{'─'*60}")
        print(f"GPU PROFILING TOOLS")
        print(f"{'─'*60}")
        print(f"For detailed GPU metrics, use:")
        print(f"  1. Metal Performance HUD (live GPU utilization %):")
        print(f"     MTL_HUD_ENABLED=1 python3 main.py")
        print(f"  2. Instruments (detailed GPU profiling):")
        print(f"     instruments -t 'Metal System Trace' python3 main.py")
        print(f"  3. Taichi Profiler (kernel timing breakdown, ~5-10% overhead):")
        print(f"     TAICHI_KERNEL_PROFILER=1 python3 main.py")
        print(f"{'═'*60}")

    def on_mouse_down(self, event):
        """Handle mouse button press - start tracking drag"""
        self.mouse_down = True
        self.last_mouse_x = event.x
        self.last_mouse_y = event.y

    def on_mouse_up(self, event):
        """Handle mouse button release - stop tracking drag"""
        self.mouse_down = False

    def on_mouse_drag(self, event):
        """Handle mouse drag - rotate camera with throttling to reduce buffer clear frequency"""
        if not self.mouse_down:
            return

        # Throttle updates to reduce buffer clears (20 updates/sec instead of 60+)
        import time
        current_time = time.time()
        if current_time - self.last_camera_update_time < self.camera_update_interval:
            # Update mouse position but skip camera update
            self.last_mouse_x = event.x
            self.last_mouse_y = event.y
            return

        self.last_camera_update_time = current_time

        # Calculate mouse delta
        delta_x = event.x - self.last_mouse_x
        delta_y = event.y - self.last_mouse_y

        # Update last position for next delta
        self.last_mouse_x = event.x
        self.last_mouse_y = event.y

        # Only update if there's actual movement
        if delta_x == 0 and delta_y == 0:
            return

        # Rotate camera and restart rendering
        self.rotate_camera(delta_x, delta_y)
        self.restart_rendering()

    def setup_interactive_preview(self, update_interval_ms=16):
        """
        Setup live preview window with mouse controls.

        Args:
            update_interval_ms: Preview update frequency in milliseconds (default 16ms = ~60 FPS)
        """
        # Setup base preview window
        self.setup_live_preview(update_interval_ms=update_interval_ms)

        # Bind mouse events to the preview window
        self.preview_window.bind('<Button-1>', self.on_mouse_down)
        self.preview_window.bind('<B1-Motion>', self.on_mouse_drag)
        self.preview_window.bind('<ButtonRelease-1>', self.on_mouse_up)

        # Update window title to show controls
        self.preview_window.title("Path Tracer - Interactive View (Drag to Rotate)")

        print("Interactive controls enabled:")
        print("  - Left-click + drag to rotate camera (continuous rendering)")
        print(f"  - Camera update rate: {1.0/self.camera_update_interval:.0f} updates/sec (throttled)")
        print(f"  - Display refresh rate: ~{1000//update_interval_ms} FPS")
        print(f"  - Rotation sensitivity: {self.rotation_velocity}")

    def render_interactive(self):
        """
        Main interactive rendering loop.
        Renders up to cam.samples_per_pixel, then pauses but keeps window open.
        Mouse interaction restarts rendering from scratch.
        """
        # Print compact setup summary
        print(f"\nInteractiveViewer")
        print(f"Resolution: {self.cam.img_width}x{self.cam.img_height} | Max Samples: {self.cam.samples_per_pixel} | Depth: {self.max_depth}")
        print(f"Spheres: {self.num_spheres} | BVH Nodes: {self.num_bvh_nodes}")

        # Show profiler status if enabled
        from render_server.taichi_renderer import ENABLE_PROFILER
        if ENABLE_PROFILER:
            print(f"⚙️  Taichi Kernel Profiler: ENABLED (5-10% overhead)")
        print(f"\nSetup Timing:")
        print(f"  Taichi Init: {self.timing['taichi_init']*1000:6.2f}ms | Scene Compile: {self.timing['scene_compile']*1000:6.2f}ms")
        print(f"  BVH Flatten: {self.timing['bvh_flatten']*1000:6.2f}ms | Camera Upload: {self.timing['camera_upload']*1000:6.2f}ms")
        print(f"  Total Setup: {self.timing['total_setup']*1000:6.2f}ms")
        print(f"\nCamera Position:")
        print(f"  lookfrom: ({self.cam.lookfrom.x:6.2f}, {self.cam.lookfrom.y:6.2f}, {self.cam.lookfrom.z:6.2f})")
        print(f"  lookat:   ({self.cam.lookat.x:6.2f}, {self.cam.lookat.y:6.2f}, {self.cam.lookat.z:6.2f})")

        # Upload initial camera parameters to GPU
        self._upload_camera_to_gpu()

        # Warm up JIT compiler (force kernel compilation during setup)
        print("\nWarming up GPU kernels...")
        import time
        from render_server.taichi_renderer import fields
        warmup_start = time.time()
        self.render_sample(0)  # Compile all kernels on first call
        ti.sync()  # Wait for GPU to finish
        self.clear_accumulation_buffer()  # Clear the warmup sample
        fields.reset_depth_stats()  # Reset depth statistics
        warmup_time = time.time() - warmup_start
        print(f"  Kernel Warmup: {warmup_time*1000:6.2f}ms (JIT compilation complete)")

        # Setup preview window with mouse controls (16ms = ~60 FPS for smooth display)
        self.setup_interactive_preview(update_interval_ms=16)

        # Calculate display interval (every 5%)
        display_interval = max(1, self.cam.samples_per_pixel // 20)  # 20 updates = every 5%
        total_pixels = self.cam.img_width * self.cam.img_height

        print(f"\nRendering Progress:")
        print(f"{'─'*60}")

        # Continuous event loop - runs until window is closed
        import time

        # Initialize render start time
        self.render_start_time = time.time()

        try:
            while self.preview_window is not None:
                # Only render if we haven't reached max samples
                if self.is_rendering_active and self.current_sample < self.cam.samples_per_pixel:
                    # Time this sample
                    sample_start = time.time()

                    # Render one sample (massively parallel on GPU)
                    self.render_sample(self.current_sample)

                    # Wait for GPU to finish
                    ti.sync()

                    # Track sample time
                    sample_time = time.time() - sample_start
                    self.sample_times.append(sample_time)

                    # Increment sample counter
                    self.current_sample += 1

                    # Print progress at intervals (5% increments)
                    should_display = (self.current_sample % display_interval == 0 or
                                    self.current_sample == 1 or
                                    self.current_sample >= self.cam.samples_per_pixel)
                    if should_display:
                        elapsed = time.time() - self.render_start_time

                        avg_time = sum(self.sample_times) / len(self.sample_times)
                        throughput = total_pixels / avg_time if avg_time > 0 else 0

                        # Calculate ETA
                        remaining_samples = self.cam.samples_per_pixel - self.current_sample
                        eta = remaining_samples * avg_time

                        # Progress percentage
                        progress = self.current_sample / self.cam.samples_per_pixel * 100

                        print(f"{self.current_sample:4d}/{self.cam.samples_per_pixel} ({progress:5.1f}%) │ "
                              f"{sample_time*1000:5.1f}ms │ "
                              f"Elapsed: {elapsed:5.1f}s │ "
                              f"Throughput: {throughput/1e6:5.2f}M pix/s │ "
                              f"ETA: {eta:4.1f}s")

                    # Check if we've reached max samples
                    if self.current_sample >= self.cam.samples_per_pixel:
                        print(f"{'─'*60}")
                        print(f"✓ Reached max samples ({self.cam.samples_per_pixel})")
                        self._print_render_summary()
                        print("  Window remains open - drag mouse to rotate camera and restart rendering")
                        self.is_rendering_active = False

                # Always process Tkinter events (allows mouse handlers to fire)
                # This keeps the window responsive even when not rendering
                try:
                    self.update_preview_if_needed()
                except:
                    # Window was closed
                    break

        except KeyboardInterrupt:
            print("\nRendering interrupted by user")

        finally:
            # Final sync and write image
            print(f"\n{'─'*60}")
            print(f"Final sample count: {self.current_sample}")
            self._print_render_summary()
            print("\nSaving final image...")
            self._sync_gpu_to_cpu()
            self.write_image()
            print(f"✓ Image saved to {self.img_path}")
