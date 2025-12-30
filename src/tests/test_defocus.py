"""Test script to verify defocus implementation"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.material import *
from core import *
from util import *
from render_server.renderer_factory import RendererFactory

def test_defocus():
    """Simple scene to test defocus blur"""
    world = hittable_list()

    # Ground
    ground_material = lambertian.from_color(color(0.5, 0.5, 0.5))
    world.add(Sphere.stationary(point3(0, -1000, 0), 1000, ground_material))

    # Three spheres at different distances
    # Close sphere (red)
    world.add(Sphere.stationary(point3(-2, 0.5, 2), 0.5, lambertian.from_color(color(0.8, 0.2, 0.2))))

    # Middle sphere (green) - THIS SHOULD BE IN FOCUS
    world.add(Sphere.stationary(point3(0, 0.5, 5), 0.5, lambertian.from_color(color(0.2, 0.8, 0.2))))

    # Far sphere (blue)
    world.add(Sphere.stationary(point3(2, 0.5, 8), 0.5, lambertian.from_color(color(0.2, 0.2, 0.8))))

    # Build BVH
    bvh = bvh_node.from_objects(world.objects, 0, len(world.objects))
    world = hittable_list()
    world.add(bvh)

    cam = camera()
    cam.aspect_ratio = 16.0 / 9.0
    cam.img_width = 400
    cam.samples_per_pixel = 50
    cam.max_depth = 20

    cam.vfov = 40
    cam.lookfrom = point3(0, 1, 0)  # Camera at origin
    cam.lookat = point3(0, 0.5, 5)   # Looking at middle sphere
    cam.vup = vec3(0, 1, 0)

    # Enable defocus blur
    cam.defocus_angle = 2.0      # Wider aperture = more blur
    cam.focus_distance = 5.0     # Focus on the green sphere at distance 5

    cam.background = color(0.70, 0.80, 1.00)

    print("="*60)
    print("DEFOCUS TEST")
    print("="*60)
    print(f"Camera settings:")
    print(f"  defocus_angle: {cam.defocus_angle}")
    print(f"  focus_distance: {cam.focus_distance}")
    print(f"  lookfrom: {cam.lookfrom}")
    print(f"  lookat: {cam.lookat}")
    print(f"\nExpected result:")
    print(f"  - Green sphere (at distance 5) should be SHARP")
    print(f"  - Red sphere (at distance 2) should be BLURRY")
    print(f"  - Blue sphere (at distance 8) should be BLURRY")
    print("="*60)

    # Test with Taichi renderer
    renderer = RendererFactory.create(
        'taichi',
        world,
        cam,
        "../../temp/test_defocus_taichi.ppm"
    )
    renderer.render()

if __name__ == "__main__":
    test_defocus()
