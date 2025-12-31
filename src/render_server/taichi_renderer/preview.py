"""
Live Preview: Tkinter-based window for displaying render progress.
Completely standalone - no Taichi dependencies.
"""

import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from typing import Callable, Optional


class LivePreview:
    """
    Live preview window that periodically displays render progress.

    Usage:
        preview = LivePreview(width, height)
        preview.start(update_interval_ms=50)

        # During rendering:
        preview.update(accum_buffer_numpy, current_sample)

        # After rendering:
        preview.finish()  # Keeps window open until user closes
    """

    def __init__(self, width: int, height: int, title: str = "Path Tracer - Live Preview"):
        self.width = width
        self.height = height
        self.title = title

        self.window = None
        self.canvas = None
        self.label = None
        self.photo = None
        self.update_interval_ms = 50
        self.is_finished = False

    def start(self, update_interval_ms: int = 50):
        """Initialize and show the preview window"""
        self.update_interval_ms = update_interval_ms

        # Create window
        self.window = tk.Tk()
        self.window.title(self.title)

        # Create canvas for image display
        self.canvas = tk.Canvas(self.window, width=self.width, height=self.height)
        self.canvas.pack()

        # Create label for status text
        self.label = tk.Label(self.window, text="Rendering...", font=("Courier", 10))
        self.label.pack()

        # Don't block - just update
        self.window.update()

    def update(self, accum_buffer: np.ndarray, current_sample: int, total_samples: int):
        """
        Update preview with current accumulation buffer state.

        Args:
            accum_buffer: numpy array of shape (H, W, 3), dtype float32
            current_sample: number of samples completed
            total_samples: total samples to render
        """
        if self.window is None or self.is_finished:
            return

        # Convert buffer to image
        img_array = self.buffer_to_image(accum_buffer, current_sample)

        # Create PIL image and Tkinter photo
        img = Image.fromarray(img_array, mode='RGB')
        self.photo = ImageTk.PhotoImage(img)

        # Update canvas
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

        # Update status label
        progress = (current_sample / total_samples) * 100
        status_text = f"Sample {current_sample}/{total_samples} ({progress:.1f}%)"
        self.label.config(text=status_text)

    def process_events(self):
        """Process pending Tkinter events. Call periodically during render."""
        if self.window is not None and not self.is_finished:
            try:
                self.window.update()
            except:
                # Window was closed
                self.window = None

    def finish(self):
        """Mark rendering complete and keep window open for viewing"""
        if self.window is None:
            return

        self.is_finished = True
        self.label.config(text="Rendering Complete - Close window when done")

        # Enter main loop to keep window open
        try:
            self.window.mainloop()
        except:
            pass

    def close(self):
        """Close the preview window"""
        if self.window is not None:
            try:
                self.window.destroy()
            except:
                pass
            self.window = None

    @staticmethod
    def buffer_to_image(accum_buffer: np.ndarray, sample_count: int) -> np.ndarray:
        """
        Convert accumulation buffer to displayable uint8 image.
        Applies: sample averaging, gamma correction (gamma=2.0), clamping.

        Args:
            accum_buffer: (H, W, 3) float32 accumulated colors
            sample_count: number of samples accumulated

        Returns: (H, W, 3) uint8 image ready for display
        """
        scale = 1.0 / max(1, sample_count)
        scaled = accum_buffer * scale
        gamma_corrected = np.sqrt(np.maximum(0, scaled))  # gamma = 2.0
        return np.clip(gamma_corrected * 255.999, 0, 255).astype(np.uint8)
