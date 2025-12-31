"""
Test script to render a single quad to isolate the bug
"""
import sys
sys.path.insert(0, '/Users/fakhir/Work/path-tracer-python/src')

from core.hittable_list import hittable_list
from core.quad import quad
from core.bvh_node import bvh_node
from core.material.lambertian import lambertian
from util import *
from camera import camera
from render_server.renderer_factory import RendererFactory

def test_back_wall():
    """Test rendering just the back wall that's missing in GPU"""
    world = hittable_list()

    white = lambertian.from_color(color(0.73, 0.73, 0.73))

    # Back wall from cornell_box (line 849 in scenes.py)
    world.add(quad(point3(0, 0, 555), vec3(0, 555, 0), vec3(555, 0, 0), white))

    # Create BVH
    bvh = bvh_node.from_objects(world.objects, 0, len(world.objects))
    world = hittable_list()
    world.add(bvh)

    cam = camera()
    cam.aspect_ratio = 1.0
    cam.img_width = 200
    cam.samples_per_pixel = 10
    cam.vfov = 40
    cam.lookfrom = point3(278, 278, -800)
    cam.lookat = point3(278, 278, 0)
    cam.vup = vec3(0, 1, 0)
    cam.defocus_angle = 0

    print("Testing back wall with GPU renderer...")
    renderer = RendererFactory.create(
        'taichi',
        world,
        cam,
        "../temp/test_back_wall.ppm"
    )
    renderer.background_color = color(0, 0, 0)
    renderer.max_depth = 10
    renderer.render()
    print("✓ Done! Check ../temp/test_back_wall.ppm")

if __name__ == "__main__":
    print("="*60)
    print("Testing back wall quad to isolate GPU rendering bug")
    print("="*60)

    test_back_wall()

    print("\n" + "="*60)
    print("Test complete. Check ../temp/test_back_wall.ppm")
    print("="*60)
