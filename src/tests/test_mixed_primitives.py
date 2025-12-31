"""
Test scene with mixed primitives: spheres, quads, and triangles
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.material import *
from core.triangle import triangle
from core.quad import quad
from core.sphere import Sphere
from render_server.renderer_factory import RendererFactory
from util import *
from core import *

def test_mixed_primitives():
    """Test scene with spheres, quads, and triangles together"""
    world = hittable_list()

    # Ground plane (large quad)
    ground_mat = lambertian.from_color(color(0.5, 0.5, 0.5))
    world.add(quad(
        point3(-10, 0, -10),
        vec3(20, 0, 0),
        vec3(0, 0, 20),
        ground_mat
    ))

    # Spheres
    mat_sphere1 = metal(color(0.8, 0.8, 0.9), 0.05)
    world.add(Sphere.stationary(point3(-3, 1, 0), 1.0, mat_sphere1))

    mat_sphere2 = dielectric(1.5)
    world.add(Sphere.stationary(point3(0, 1, 0), 1.0, mat_sphere2))

    mat_sphere3 = lambertian.from_color(color(0.8, 0.3, 0.3))
    world.add(Sphere.stationary(point3(3, 1, 0), 1.0, mat_sphere3))

    # Triangles (forming a simple tent/roof shape)
    mat_tri_red = lambertian.from_color(color(0.9, 0.2, 0.2))
    mat_tri_blue = lambertian.from_color(color(0.2, 0.2, 0.9))

    # Tent over the middle sphere
    apex = point3(0, 4, 0)
    base_bl = point3(-2, 1.5, -2)
    base_br = point3(2, 1.5, -2)
    base_fr = point3(2, 1.5, 2)
    base_fl = point3(-2, 1.5, 2)

    world.add(triangle(base_bl, base_br, apex, mat_tri_red))  # back
    world.add(triangle(base_br, base_fr, apex, mat_tri_blue)) # right
    world.add(triangle(base_fr, base_fl, apex, mat_tri_red))  # front
    world.add(triangle(base_fl, base_bl, apex, mat_tri_blue)) # left

    # Vertical quad (backdrop)
    mat_quad = lambertian.from_color(color(0.3, 0.6, 0.8))
    world.add(quad(
        point3(-5, 0, -3),
        vec3(10, 0, 0),
        vec3(0, 5, 0),
        mat_quad
    ))

    # Camera setup
    cam = camera()
    cam.aspect_ratio = 16.0 / 9.0
    cam.img_width = 1200
    cam.samples_per_pixel = 100
    cam.max_depth = 50

    cam.vfov = 40
    cam.lookfrom = point3(8, 5, 12)
    cam.lookat = point3(0, 2, 0)
    cam.vup = vec3(0, 1, 0)
    cam.defocus_angle = 0.0

    # Use Taichi renderer
    renderer = RendererFactory.create(
        'taichi',
        world,
        cam,
        "test_mixed_primitives.png"
    )
    renderer.render()

if __name__ == "__main__":
    test_mixed_primitives()
