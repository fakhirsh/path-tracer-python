"""
Test scene for quad rendering
Simple scene with quad ground plane, quad wall, and some spheres
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.material import *
from core.texture import checker_texture
from core.quad import quad
from render_server.renderer_factory import RendererFactory
from util import *
from core import *

def test_quad():
    """Simple scene with a quad and spheres to test quad rendering"""
    world = hittable_list()

    # Ground plane as a large quad
    ground_mat = lambertian.from_color(color(0.5, 0.5, 0.5))
    ground_quad = quad(
        point3(-5, 0, -5),  # Q: corner point
        vec3(10, 0, 0),      # u: edge vector (10 units in x)
        vec3(0, 0, 10),      # v: edge vector (10 units in z)
        ground_mat
    )
    world.add(ground_quad)

    # Add a few spheres above the ground
    sphere_mat1 = lambertian.from_color(color(0.8, 0.3, 0.3))
    world.add(Sphere.stationary(point3(0, 0.5, 0), 0.5, sphere_mat1))

    sphere_mat2 = metal(color(0.7, 0.7, 0.7), 0.1)
    world.add(Sphere.stationary(point3(-1.5, 0.5, 0), 0.5, sphere_mat2))

    sphere_mat3 = dielectric(1.5)
    world.add(Sphere.stationary(point3(1.5, 0.5, 0), 0.5, sphere_mat3))

    # Back wall quad
    wall_mat = lambertian.from_color(color(0.3, 0.5, 0.8))
    wall_quad = quad(
        point3(-3, 0, 3),    # Q: corner point
        vec3(6, 0, 0),       # u: edge vector (6 units in x)
        vec3(0, 3, 0),       # v: edge vector (3 units up)
        wall_mat
    )
    world.add(wall_quad)

    # Create BVH
    bvh = bvh_node.from_objects(world.objects, 0, len(world.objects))
    world = hittable_list()
    world.add(bvh)

    # Camera setup
    cam = camera()
    cam.aspect_ratio = 16.0 / 9.0
    cam.img_width = 800
    cam.samples_per_pixel = 50
    cam.max_depth = 50

    cam.vfov = 60
    cam.lookfrom = point3(0, 2, -5)
    cam.lookat = point3(0, 0.5, 0)
    cam.vup = vec3(0, 1, 0)
    cam.defocus_angle = 0.0

    # Use Taichi renderer
    renderer = RendererFactory.create(
        'taichi',
        world,
        cam,
        "test_quad.png"
    )
    renderer.render()

if __name__ == "__main__":
    test_quad()
