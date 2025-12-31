from abc import ABC, abstractmethod
from core.hittable import hittable
from core.camera import camera
from util.color import color, write_color
from util.ray import Ray
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk


class BaseRenderer(ABC):
    """Abstract base class for all renderers (CPU, GPU, etc.)"""

    def __init__(self, world: hittable, cam: camera, img_path: str):
        self.world = self._compile(world)
        self.img_path = img_path

        self.cam = cam
        self.cam.initialize()

        # Rendering parameters
        self.max_depth = 5
        self.background_color = color(0.70, 0.80, 1.00)
        self.russian_roulette_enabled = False
        self.russian_roulette_min_depth = 3
        self.russian_roulette_threshold = 0.1

        # Accumulation buffer
        self.accum_buffer = [[color(0, 0, 0) for _ in range(self.cam.img_width)]
                             for _ in range(self.cam.img_height)]
        self.pixel_samples_scale = 1.0 / self.cam.samples_per_pixel

        # Initialize depth tracking statistics
        self.total_path_depth = 0
        self.total_paths = 0
        self.rr_terminations = 0
        self.max_depth_terminations = 0

        # Live preview state
        self.is_rendering = False
        self.preview_window = None
        self.preview_label = None
        self.preview_photo = None
        self.update_interval_ms = 500
        self.current_sample = 0

    def _compile(self, world: hittable) -> hittable:
        """
        Hook method for renderer-specific world compilation/optimization.
        Override in subclasses if needed (e.g., GPU memory packing, BVH building).
        """
        return world

    @abstractmethod
    def ray_color(self, r: Ray, depth: int, initial_depth: int = None) -> color:
        """
        Abstract method: compute the color for a given ray.
        Must be implemented by all concrete renderers.
        """
        pass

    @abstractmethod
    def render(self):
        """
        Abstract method: execute the rendering loop.
        GPU renderers launch kernels, CPU renderers use threads/multiprocessing.
        Implementations should:
        1. Accumulate samples into self.accum_buffer
        2. Call self.write_image() when done
        3. Optionally call self.print_statistics()
        """
        pass

    def _simple_render_loop(self):
        """
        Helper method: simple single-threaded render loop.
        Concrete renderers can call this if they don't need parallelization.
        """
        # print(f"Rendering with {self.__class__.__name__}...")

        # for sample in range(self.cam.samples_per_pixel):
        #     for h in range(self.cam.img_height):
        #         for w in range(self.cam.img_width):
        #             r = self.cam.get_ray(w, h)
        #             pcolor = self.ray_color(r, self.max_depth)
        #             self.accum_buffer[h][w] += pcolor

        # print(f"{self.__class__.__name__} completed.")
        pass

    def write_image(self):
        """Write the accumulated buffer to a PPM file"""
        print(f"Writing image to {self.img_path}...")
        with open(self.img_path, 'w') as f:
            f.write(f"P3\n{self.cam.img_width} {self.cam.img_height}\n255\n")
            for h in range(self.cam.img_height):
                for w in range(self.cam.img_width):
                    pcolor = self.accum_buffer[h][w]
                    write_color(f, self.pixel_samples_scale * pcolor)
        print("Image writing completed.")

    def get_statistics(self) -> dict:
        """Return rendering statistics as a dictionary"""
        if self.total_paths == 0:
            return {}

        avg_depth = self.total_path_depth / self.total_paths
        rr_percentage = (self.rr_terminations / self.total_paths) * 100
        max_depth_percentage = (self.max_depth_terminations / self.total_paths) * 100

        return {
            'average_path_depth': avg_depth,
            'max_depth': self.max_depth,
            'rr_terminations': self.rr_terminations,
            'rr_percentage': rr_percentage,
            'max_depth_terminations': self.max_depth_terminations,
            'max_depth_percentage': max_depth_percentage,
            'total_paths': self.total_paths
        }

    def print_statistics(self):
        """Print rendering statistics to console"""
        stats = self.get_statistics()
        if not stats:
            return

        print(f"\nDepth Statistics:")
        print(f"  Average path depth: {stats['average_path_depth']:.2f} (max: {stats['max_depth']})")
        print(f"  Russian Roulette terminations: {stats['rr_terminations']:,} ({stats['rr_percentage']:.1f}%)")
        print(f"  Max depth terminations: {stats['max_depth_terminations']:,} ({stats['max_depth_percentage']:.1f}%)")
        print(f"  Total paths traced: {stats['total_paths']:,}")

    def setup_live_preview(self, update_interval_ms=500):
        """
        Initialize live preview window with timer-based updates.

        Args:
            update_interval_ms: Update interval in milliseconds (default: 500ms)
        """
        self.update_interval_ms = update_interval_ms
        self.is_rendering = True

        # Create Tkinter window
        self.preview_window = tk.Tk()
        self.preview_window.title("Path Tracer - Live Preview")

        # Bring window to front
        self.preview_window.lift()
        self.preview_window.attributes('-topmost', True)
        self.preview_window.after_idle(self.preview_window.attributes, '-topmost', False)

        # Create label to hold the image
        self.preview_label = tk.Label(self.preview_window)
        self.preview_label.pack()

        # Add title showing render progress
        self.preview_title = tk.Label(
            self.preview_window,
            text="Initializing...",
            font=("Arial", 12)
        )
        self.preview_title.pack()

        # Create initial blank image to ensure window has content
        initial_img = Image.new('RGB', (self.cam.img_width, self.cam.img_height), color=(0, 0, 0))
        self.preview_photo = ImageTk.PhotoImage(initial_img)
        self.preview_label.config(image=self.preview_photo)

        # Force the window to appear immediately
        self.preview_window.update()

        # Start the periodic update
        self.preview_window.after(self.update_interval_ms, self._preview_update_callback)

        print(f"Live preview started (updating every {update_interval_ms}ms)")
        print(f"Window size: {self.cam.img_width}x{self.cam.img_height}")

    def _preview_update_callback(self):
        """Timer callback - periodically reads accum_buffer and updates display"""
        if not self.is_rendering or self.preview_window is None:
            return

        try:
            # Convert accumulation buffer to displayable image
            img_array = self._accum_buffer_to_array()

            # Convert to PIL Image
            img = Image.fromarray(img_array, mode='RGB')

            # Convert to PhotoImage for Tkinter
            self.preview_photo = ImageTk.PhotoImage(img)

            # Update the label
            self.preview_label.config(image=self.preview_photo)

            # Update title with current sample count
            samples_completed = self.current_sample
            # Use max_samples if available (InteractiveViewer), otherwise use cam.samples_per_pixel
            total_samples = getattr(self, 'max_samples', self.cam.samples_per_pixel)
            progress = (samples_completed / total_samples * 100) if total_samples > 0 else 0
            self.preview_title.config(
                text=f"Sample {samples_completed}/{total_samples} ({progress:.1f}%)"
            )

            # Process pending events to update the window
            self.preview_window.update()

            # Schedule next update if still rendering
            if self.is_rendering:
                self.preview_window.after(self.update_interval_ms, self._preview_update_callback)

        except Exception as e:
            print(f"Error updating preview: {e}")

    def _accum_buffer_to_array(self):
        """
        Convert accumulation buffer to numpy array for display.
        Applies tone mapping and gamma correction.

        Returns:
            numpy array of shape (height, width, 3) with uint8 values
        """
        height = self.cam.img_height
        width = self.cam.img_width

        # Calculate scale based on current sample count
        # This allows progressive preview as samples accumulate
        scale = 1.0 / max(1, self.current_sample)

        # Create numpy array
        img_array = np.zeros((height, width, 3), dtype=np.uint8)

        for h in range(height):
            for w in range(width):
                # Get accumulated color
                accum_color = self.accum_buffer[h][w]

                # Scale by number of samples taken so far
                r = accum_color.x * scale
                g = accum_color.y * scale
                b = accum_color.z * scale

                # Apply gamma correction (gamma = 2.0)
                r = np.sqrt(max(0, r))
                g = np.sqrt(max(0, g))
                b = np.sqrt(max(0, b))

                # Convert to 0-255 range and clamp
                img_array[h, w, 0] = int(np.clip(r * 255.999, 0, 255))
                img_array[h, w, 1] = int(np.clip(g * 255.999, 0, 255))
                img_array[h, w, 2] = int(np.clip(b * 255.999, 0, 255))

        return img_array

    def update_preview_if_needed(self):
        """
        Process pending Tkinter events if preview is enabled.
        Call this periodically during rendering to allow timer callbacks to execute.
        This is a lightweight operation that only processes events if preview is active.
        """
        if self.preview_window is not None and self.is_rendering:
            try:
                self.preview_window.update()
            except:
                pass

    def close_preview(self):
        """Stop live preview and close the window"""
        self.is_rendering = False

        if self.preview_window is not None:
            try:
                self.preview_window.destroy()
                print("Live preview closed")
            except:
                pass
            finally:
                self.preview_window = None
                self.preview_label = None
                self.preview_photo = None
