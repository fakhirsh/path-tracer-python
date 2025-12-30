"""
Interactive Path Tracer Demo

Launch an interactive path tracer with mouse-controlled camera rotation.
Left-click and drag to rotate the camera around the scene.
"""

import sys
from pathlib import Path

# Add parent directory (src/) to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from render_server.interactive_viewer import InteractiveViewer
from core.camera import camera
from core.hittable_list import hittable_list
from core.sphere import Sphere
from core.material import lambertian, metal, dielectric
from core.bvh_node import bvh_node
from util import point3, vec3, color


def create_demo_scene():
    """Create a simple demo scene with various materials"""
    world = hittable_list()

    # Ground sphere
    ground_material = lambertian.from_color(color(0.5, 0.5, 0.5))
    world.add(Sphere.stationary(point3(0, -100.5, -1), 100, ground_material))

    # Center sphere - Lambertian (matte red)
    center_material = lambertian.from_color(color(0.7, 0.3, 0.3))
    world.add(Sphere.stationary(point3(0, 0, -1), 0.5, center_material))

    # Left sphere - Metal (gold)
    left_material = metal(color(0.8, 0.6, 0.2), 0.3)
    world.add(Sphere.stationary(point3(-1.0, 0, -1), 0.5, left_material))

    # Right sphere - Dielectric (glass)
    right_material = dielectric(1.5)
    world.add(Sphere.stationary(point3(1.0, 0, -1), 0.5, right_material))

    # Small spheres for detail
    small_mat1 = lambertian.from_color(color(0.2, 0.3, 0.8))
    world.add(Sphere.stationary(point3(-0.5, 0.2, -0.5), 0.2, small_mat1))

    small_mat2 = metal(color(0.9, 0.9, 0.9), 0.0)
    world.add(Sphere.stationary(point3(0.5, 0.2, -0.5), 0.2, small_mat2))

    # Build BVH for GPU acceleration
    bvh = bvh_node.from_objects(world.objects, 0, len(world.objects))
    world = hittable_list()
    world.add(bvh)

    return world


def create_three_sphere_scene():
    """Create the classic three-sphere scene"""
    world = hittable_list()

    # Ground
    ground_material = lambertian.from_color(color(0.8, 0.8, 0.0))
    world.add(Sphere.stationary(point3(0, -100.5, -1), 100, ground_material))

    # Center - Lambertian
    center_material = lambertian.from_color(color(0.1, 0.2, 0.5))
    world.add(Sphere.stationary(point3(0, 0, -1.2), 0.5, center_material))

    # Left - Metal
    left_material = dielectric(1.5)
    world.add(Sphere.stationary(point3(-1, 0, -1), 0.5, left_material))
    world.add(Sphere.stationary(point3(-1, 0, -1), 0.4, dielectric(1.0 / 1.5)))

    # Right - Glass
    right_material = metal(color(0.8, 0.6, 0.2), 0.0)
    world.add(Sphere.stationary(point3(1, 0, -1), 0.5, right_material))

    # Build BVH for GPU acceleration
    bvh = bvh_node.from_objects(world.objects, 0, len(world.objects))
    world = hittable_list()
    world.add(bvh)

    return world


def main():
    """Run the interactive viewer with a demo scene"""
    print("Interactive Path Tracer")
    print("=" * 60)
    print()

    # Choose which scene to render
    # world = create_demo_scene()
    world = create_three_sphere_scene()

    # Setup camera
    cam = camera()
    cam.aspect_ratio = 16.0 / 9.0
    cam.img_width = 800
    cam.samples_per_pixel = 1000  # Not used in interactive mode, but set for final save

    cam.vfov = 20
    cam.lookfrom = point3(0, 1, 5)  # Start position
    cam.lookat = point3(0, 0, -1)   # Look at center of scene
    cam.vup = vec3(0, 1, 0)         # Up direction

    # Optional: depth of field
    cam.defocus_angle = 0.0  # Set to 0.6 for depth of field effect
    cam.focus_distance = (cam.lookfrom - cam.lookat).length()

    # Create interactive viewer
    viewer = InteractiveViewer(
        world,
        cam,
        "../temp/interactive_output.ppm"
    )

    # Set rendering parameters
    viewer.max_depth = 50
    viewer.max_samples = 1000  # Stop rendering after this many samples (window stays open)
    viewer.background_color = color(0.70, 0.80, 1.00)

    # Adjust rotation sensitivity if needed
    # viewer.rotation_velocity = (0.5, 0.5)  # Faster rotation
    # viewer.rotation_velocity = (0.1, 0.1)  # Slower rotation

    print("Starting interactive viewer...")
    print("Controls:")
    print("  - Left-click and drag to rotate camera")
    print("  - Close window to exit and save final image")
    print()

    # Run interactive rendering
    viewer.render_interactive()


if __name__ == "__main__":
    main()
