#!/usr/bin/env python3
"""
Test script for Taichi GPU renderer.
Renders a simple scene with a few spheres.
"""

import sys
sys.path.insert(0, '..')

from core.material import lambertian
from core.sphere import Sphere
from core.hittable_list import hittable_list
from core.camera import camera
from util import point3, vec3, color
from render_server.renderer_factory import RendererFactory


def test_taichi_simple():
    """Simple test scene with 3 spheres"""

    print("=" * 60)
    print("Testing Taichi GPU Renderer - Simple Scene")
    print("=" * 60)

    # Create world
    world = hittable_list()

    # Ground sphere
    ground_material = lambertian.from_color(color(0.5, 0.5, 0.5))
    world.add(Sphere.stationary(point3(0, -100.5, -1), 100, ground_material))

    # Center sphere (red)
    center_material = lambertian.from_color(color(0.7, 0.3, 0.3))
    world.add(Sphere.stationary(point3(0, 0, -1), 0.5, center_material))

    # Left sphere (blue)
    left_material = lambertian.from_color(color(0.3, 0.3, 0.7))
    world.add(Sphere.stationary(point3(-1, 0, -1), 0.5, left_material))

    # Right sphere (green)
    right_material = lambertian.from_color(color(0.3, 0.7, 0.3))
    world.add(Sphere.stationary(point3(1, 0, -1), 0.5, right_material))

    # Setup camera
    cam = camera()
    cam.aspect_ratio = 16.0 / 9.0
    cam.img_width = 400
    cam.samples_per_pixel = 50
    cam.vfov = 90
    cam.lookfrom = point3(0, 0, 0)
    cam.lookat = point3(0, 0, -1)
    cam.vup = vec3(0, 1, 0)
    cam.defocus_angle = 0.0

    # Create Taichi renderer
    renderer = RendererFactory.create(
        'taichi',
        world,
        cam,
        "../../temp/test_taichi_simple.ppm"
    )

    # Set rendering parameters
    renderer.max_depth = 10
    renderer.background_color = color(0.5, 0.7, 1.0)

    # Render!
    renderer.render(enable_preview=True)

    print("\n" + "=" * 60)
    print("✓ Rendering complete!")
    print(f"✓ Output saved to: ../../temp/test_taichi_simple.ppm")
    print("=" * 60)


if __name__ == "__main__":
    test_taichi_simple()
