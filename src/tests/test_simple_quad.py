"""
Minimal quad test - just one quad, no BVH
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.material import *
from core.quad import quad
from render_server.renderer_factory import RendererFactory
from util import *
from core import *

def test_simple_quad():
    """Absolute simplest quad test - one quad facing camera"""
    world = hittable_list()

    # Single quad directly facing camera
    mat = lambertian.from_color(color(0.8, 0.3, 0.3))
    q = quad(
        point3(-2, -2, 0),  # Q: bottom left corner at z=0
        vec3(4, 0, 0),       # u: 4 units right
        vec3(0, 4, 0),       # v: 4 units up
        mat
    )
    world.add(q)

    # Camera setup - looking straight at the quad
    cam = camera()
    cam.aspect_ratio = 1.0
    cam.img_width = 400
    cam.samples_per_pixel = 10
    cam.max_depth = 10

    cam.vfov = 90
    cam.lookfrom = point3(0, 0, -5)  # Camera 5 units back
    cam.lookat = point3(0, 0, 0)      # Looking at origin
    cam.vup = vec3(0, 1, 0)
    cam.defocus_angle = 0.0

    # Use Taichi renderer
    renderer = RendererFactory.create(
        'taichi',
        world,
        cam,
        "test_simple_quad.png"
    )
    renderer.background_color = color(0.1, 0.1, 0.1)  # Dark background
    renderer.render()

if __name__ == "__main__":
    test_simple_quad()
