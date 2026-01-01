from core.material import *
from core.texture import checker_texture, image_texture, noise_texture
from core import quad
from core.constant_medium import constant_medium
from render_server.renderer_factory import RendererFactory
from render_server.interactive_viewer import InteractiveViewer
from util import *
from core import *
from math import sqrt, cos, sin, pi, radians
import random
import time
import logging

#------------------------------------------------------------------------

def vol1_sec9_5():
    ground_material = lambertian.from_color(color(0.5, 0.5, 0.5))
    shpere_material = lambertian.from_color(color(0.8, 0.3, 0.3))
    world = hittable_list()
    obj = Sphere.stationary(point3(0,0,0), 0.5, shpere_material)
    world.add(obj)
    obj = Sphere.stationary(point3(0,-100.5,-1), 100, ground_material)
    world.add(obj)

    cam = camera()

    cam.aspect_ratio = 16.0 / 9.0
    cam.img_width = 800
    cam.samples_per_pixel = 100

    cam.vfov = 20
    cam.lookfrom = point3(0, 1, -5)
    cam.lookat = point3(0, 0, 0)
    cam.vup = vec3(0, 1, 0)
    cam.defocus_angle = 0.0

    # Use factory to create renderer (can easily switch between 'gpu' and 'cpu')
    renderer = RendererFactory.create(
        'taichi',  # Change to 'cpu' for CPU renderer
        world,
        cam,
        "../temp/vol1_sec9_5.ppm"
    )
    renderer.render()

#------------------------------------------------------------------------

def vol1_sec14_1():
    world = hittable_list()

    ground_material = lambertian.from_color(color(0.5, 0.5, 0.5))
    world.add(Sphere.stationary(point3(0, -1000, 0), 1000, ground_material))

    for a in range(-11, 11):
        for b in range(-11, 11):
            choose_mat = random.uniform(0, 1)
            center = point3(a + 0.9 * random.uniform(0, 1), 0.2, b + 0.9 * random.uniform(0, 1))

            if (center - point3(4, 0.2, 0)).length() > 0.9:
                sphere_material = None

                if choose_mat < 0.8:
                    # diffuse
                    albedo = color.random() * color.random()
                    sphere_material = lambertian.from_color(albedo)
                    center2 = center + vec3(0, random.uniform(0, 0.5), 0)
                    world.add(Sphere.stationary(center, 0.2, sphere_material))
                elif choose_mat < 0.95:
                    # metal
                    albedo = color.random(0.5, 1)
                    fuzz = random.uniform(0, 0.5)
                    sphere_material = metal(albedo, fuzz)
                    world.add(Sphere.stationary(center, 0.2, sphere_material))
                else:
                    # glass
                    sphere_material = dielectric(1.5)
                    world.add(Sphere.stationary(center, 0.2, sphere_material))
    material1 = dielectric(1.5)
    world.add(Sphere.stationary(point3(0, 1, 0), 1.0, material1))

    material2 = lambertian.from_color(color(0.4, 0.2, 0.1))
    world.add(Sphere.stationary(point3(-4, 1, 0), 1.0, material2))

    material3 = metal(color(0.7, 0.6, 0.5), 0.0)
    world.add(Sphere.stationary(point3(4, 1, 0), 1.0, material3))

    # Create BVH and wrap it
    bvh = bvh_node.from_objects(world.objects, 0, len(world.objects))
    world = hittable_list()
    world.add(bvh)

    cam = camera()

    cam.aspect_ratio = 16.0 / 9.0
    cam.img_width = 800
    cam.samples_per_pixel = 100
    cam.max_depth = 50

    cam.vfov = 20
    cam.lookfrom = point3(13, 2, 3)
    cam.lookat = point3(0, 0, 0)
    cam.vup = vec3(0, 1, 0)
    cam.defocus_angle = 0.0
    cam.background = color(0.70, 0.80, 1.00)

    # Use factory to create renderer (can easily switch between 'gpu' and 'cpu')
    renderer = RendererFactory.create(
        'taichi',  # Change to 'cpu' for CPU renderer
        world,
        cam,
        "../temp/vol1_sec9_5.ppm"
    )
    renderer.render()

#------------------------------------------------------------------------

def vol2_sec2_6():
    world = hittable_list()

    ground_material = lambertian.from_color(color(0.5, 0.5, 0.5))
    world.add(Sphere.stationary(point3(0, -1000, 0), 1000, ground_material))

    for a in range(-11, 11):
        for b in range(-11, 11):
            choose_mat = random.uniform(0, 1)
            center = point3(a + 0.9 * random.uniform(0, 1), 0.2, b + 0.9 * random.uniform(0, 1))

            if (center - point3(4, 0.2, 0)).length() > 0.9:
                sphere_material = None

                if choose_mat < 0.8:
                    # diffuse
                    albedo = color.random() * color.random()
                    sphere_material = lambertian.from_color(albedo)
                    center2 = center + vec3(0, random.uniform(0, 0.5), 0)
                    world.add(Sphere.moving(center, center2, 0.2, sphere_material))
                elif choose_mat < 0.95:
                    # metal
                    albedo = color.random(0.5, 1)
                    fuzz = random.uniform(0, 0.5)
                    sphere_material = metal(albedo, fuzz)
                    world.add(Sphere.stationary(center, 0.2, sphere_material))
                else:
                    # glass
                    sphere_material = dielectric(1.5)
                    world.add(Sphere.stationary(center, 0.2, sphere_material))

    material1 = dielectric(1.5)
    world.add(Sphere.stationary(point3(0, 1, 0), 1.0, material1))

    material2 = lambertian.from_color(color(0.4, 0.2, 0.1))
    world.add(Sphere.stationary(point3(-4, 1, 0), 1.0, material2))

    material3 = metal(color(0.7, 0.6, 0.5), 0.0)
    world.add(Sphere.stationary(point3(4, 1, 0), 1.0, material3))

    # Create BVH and wrap it
    bvh = bvh_node.from_objects(world.objects, 0, len(world.objects))
    world = hittable_list()
    world.add(bvh)

    cam = camera()

    cam.aspect_ratio = 16.0 / 9.0
    cam.img_width = 1280
    cam.samples_per_pixel = 100

    cam.vfov = 20
    cam.lookfrom = point3(13, 2, 3)
    cam.lookat = point3(0, 0, 0)
    cam.vup = vec3(0, 1, 0)

    # Defocus blur (depth of field)
    cam.defocus_angle = 0.6  # Aperture size (0 = no blur, larger = more blur)
    cam.focus_distance = 10.0  # Distance to focal plane (objects at this distance are sharp)

    # Use factory to create renderer (can easily switch between 'gpu' and 'cpu')
    renderer = RendererFactory.create(
        'taichi',  # Change to 'cpu' for CPU renderer
        world,
        cam,
        "../temp/vol2_sec2_6.ppm"
    )
    renderer.background_color = color(0.70, 0.80, 1.00)
    renderer.max_depth = 50
    renderer.render()


def vol2_sec2_6_interactive():
    """Interactive version of vol2_sec2_6 with mouse-controlled camera rotation"""
    world = hittable_list()

    checker = checker_texture.from_colors(0.32, color(0.2, 0.3, 0.1), color(0.9, 0.9, 0.9))
    ground_material = lambertian.from_texture(checker)
    # ground_material = lambertian.from_color(color(0.5, 0.5, 0.5))
    world.add(Sphere.stationary(point3(0, -1000, 0), 1000, ground_material))

    for a in range(-11, 11):
        for b in range(-11, 11):
            choose_mat = random.uniform(0, 1)
            center = point3(a + 0.9 * random.uniform(0, 1), 0.2, b + 0.9 * random.uniform(0, 1))

            if (center - point3(4, 0.2, 0)).length() > 0.9:
                sphere_material = None

                if choose_mat < 0.8:
                    # diffuse
                    albedo = color.random() * color.random()
                    sphere_material = lambertian.from_color(albedo)
                    center2 = center + vec3(0, random.uniform(0, 0.5), 0)
                    world.add(Sphere.moving(center, center2, 0.2, sphere_material))
                elif choose_mat < 0.95:
                    # metal
                    albedo = color.random(0.5, 1)
                    fuzz = random.uniform(0, 0.5)
                    sphere_material = metal(albedo, fuzz)
                    world.add(Sphere.stationary(center, 0.2, sphere_material))
                else:
                    # glass
                    sphere_material = dielectric(1.5)
                    world.add(Sphere.stationary(center, 0.2, sphere_material))

    material1 = dielectric(1.5)
    world.add(Sphere.stationary(point3(0, 1, 0), 1.0, material1))

    material2 = lambertian.from_color(color(0.4, 0.2, 0.1))
    world.add(Sphere.stationary(point3(-4, 1, 0), 1.0, material2))

    material3 = metal(color(0.7, 0.6, 0.5), 0.0)
    world.add(Sphere.stationary(point3(4, 1, 0), 1.0, material3))

    # Create BVH and wrap it
    bvh = bvh_node.from_objects(world.objects, 0, len(world.objects))
    world = hittable_list()
    world.add(bvh)

    cam = camera()

    cam.aspect_ratio = 16.0 / 9.0
    cam.img_width = 1280
    cam.samples_per_pixel = 1000

    cam.vfov = 20
    cam.lookfrom = point3(13, 2, 3)
    cam.lookat = point3(0, 0, 0)
    cam.vup = vec3(0, 1, 0)

    # Defocus blur (depth of field)
    cam.defocus_angle = 0.6  # Aperture size (0 = no blur, larger = more blur)
    cam.focus_distance = 10.0  # Distance to focal plane (objects at this distance are sharp)

    # Create interactive viewer
    viewer = InteractiveViewer(
        world,
        cam,
        "../temp/vol2_sec2_6_interactive.ppm"
    )
    viewer.background_color = color(0.70, 0.80, 1.00)
    viewer.max_depth = 50

    print("\nInteractive Controls:")
    print("  - Left-click and drag to rotate camera")
    print("  - Close window to exit and save final image")
    print()

    # Run interactive rendering
    viewer.render_interactive()


#------------------------------------------------------------------------

def vol2_sec42_scene_simple():
    world = hittable_list()

    # Ground
    checker = checker_texture.from_colors(0.32, color(0.2, 0.3, 0.1), color(0.9, 0.9, 0.9))
    ground_material = lambertian.from_texture(checker)
    world.add(Sphere.stationary(point3(0, -1000, 0), 1000, ground_material))

    # Moving diffuse sphere (left)
    moving_material = lambertian.from_color(color(0.8, 0.3, 0.3))
    center1 = point3(-2, 0.5, 0)
    center2 = center1 + vec3(0, 0.3, 0)
    world.add(Sphere.moving(center1, center2, 0.5, moving_material))

    # Static glass sphere (center)
    glass_material = dielectric(1.5)
    world.add(Sphere.stationary(point3(0, 0.5, 0), 0.5, glass_material))

    # Static metal sphere (right)
    metal_material = metal(color(0.7, 0.6, 0.5), 0.1)
    world.add(Sphere.stationary(point3(2, 0.5, 0), 0.5, metal_material))

    # Moving diffuse sphere (behind)
    moving_material2 = lambertian.from_color(color(0.3, 0.3, 0.8))
    center3 = point3(0, 0.3, -2)
    center4 = center3 + vec3(0, 0.4, 0)
    world.add(Sphere.moving(center3, center4, 0.3, moving_material2))

    # Additional moving diffuse sphere (front left)
    moving_material3 = lambertian.from_color(color(0.3, 0.8, 0.3))
    center5 = point3(-1, 0.3, 1)
    center6 = center5 + vec3(0, 0.4, 0)
    world.add(Sphere.moving(center5, center6, 0.3, moving_material3))

    # Additional moving diffuse sphere (front right)
    moving_material4 = lambertian.from_color(color(0.8, 0.8, 0.3))
    center7 = point3(1, 0.3, 1.5)
    center8 = center7 + vec3(0, 0.35, 0)
    world.add(Sphere.moving(center7, center8, 0.3, moving_material4))

    # Static glass sphere (smaller, right side)
    glass_material2 = dielectric(1.5)
    world.add(Sphere.stationary(point3(3, 0.3, -1), 0.3, glass_material2))

    # Static metal sphere (left side, shiny)
    metal_material2 = metal(color(0.9, 0.9, 0.9), 0.0)
    world.add(Sphere.stationary(point3(-3, 0.4, -0.5), 0.4, metal_material2))

    # Static metal sphere (back, bronze-ish)
    metal_material3 = metal(color(0.8, 0.5, 0.3), 0.3)
    world.add(Sphere.stationary(point3(0.5, 0.3, -3), 0.3, metal_material3))

    # Moving diffuse sphere (far left)
    moving_material5 = lambertian.from_color(color(0.7, 0.3, 0.7))
    center9 = point3(-3.5, 0.25, 1)
    center10 = center9 + vec3(0, 0.25, 0)
    world.add(Sphere.moving(center9, center10, 0.25, moving_material5))

    # Create BVH and wrap it
    bvh = bvh_node.from_objects(world.objects, 0, len(world.objects))
    world = hittable_list()
    world.add(bvh)

    cam = camera()

    cam.aspect_ratio = 16.0 / 9.0
    cam.img_width = 400
    cam.samples_per_pixel = 100
    cam.max_depth = 20

    cam.vfov = 20
    cam.lookfrom = point3(13, 2, 3)
    cam.lookat = point3(0, 0, 0)
    cam.vup = vec3(0, 1, 0)
    cam.defocus_angle = 0.0
    cam.background = color(0.70, 0.80, 1.00)

    cam.render(world, "../temp/vol2_sec42_scene_simple.ppm")


def vol2_sec4_3_simple():
    world = hittable_list()

    checker = checker_texture.from_colors(0.32, color(0.2, 0.3, 0.1), color(0.9, 0.9, 0.9))

    world.add(Sphere.stationary(point3(0, -10, 0), 10, lambertian.from_texture(checker)))
    world.add(Sphere.stationary(point3(0, 10, 0), 10, lambertian.from_texture(checker)))

    # Create BVH and wrap it
    bvh = bvh_node.from_objects(world.objects, 0, len(world.objects))
    world = hittable_list()
    world.add(bvh)

    cam = camera()

    cam.aspect_ratio = 16.0 / 9.0
    cam.img_width = 300
    cam.samples_per_pixel = 10
    cam.max_depth = 5

    cam.vfov = 20
    cam.lookfrom = point3(13, 2, 3)
    cam.lookat = point3(0, 0, 0)
    cam.vup = vec3(0, 1, 0)
    cam.defocus_angle = 0.0
    cam.background = color(0.70, 0.80, 1.00)

    cam.render(world, "../temp/vol2_sec4_3_simple.ppm")

#------------------------------------------------------------------------

def vol2_sec4_6():
    world = hittable_list()

    earth_texture = image_texture("assets/images/earthmap.jpg")
    earth_surface = lambertian.from_texture(earth_texture)
    globe = Sphere.stationary(point3(0, 0, 0), 2.0, earth_surface)

    world.add(globe)

    # Create BVH and wrap it
    bvh = bvh_node.from_objects(world.objects, 0, len(world.objects))
    world = hittable_list()
    world.add(bvh)

    cam = camera()

    cam.aspect_ratio = 16.0 / 9.0
    cam.img_width = 600
    cam.samples_per_pixel = 50
    cam.max_depth = 10

    cam.vfov = 20
    cam.lookfrom = point3(0,0,12)
    cam.lookat = point3(0, 0, 0)
    cam.vup = vec3(0, 1, 0)

    cam.defocus_angle = 0.0 # for perfectly sharp images, default: 0.6
    cam.background = color(0.70, 0.80, 1.00)

    cam.render(world, "../temp/vol2_sec4_6.ppm")

#------------------------------------------------------------------------

def vol2_sec4_6_ver2_cpu():
    world = hittable_list()

    # Ground plane (large sphere below)
    ground_material = lambertian.from_color(color(0.5, 0.5, 0.5))
    world.add(Sphere.stationary(point3(0, -1000, 0), 1000, ground_material))

    # LEFT: Solid texture sphere (red)
    red_material = lambertian.from_texture(solid_color.from_color(color(0.8, 0.3, 0.3)))
    world.add(Sphere.stationary(point3(-1, 0.5, 0), 0.5, red_material))

    # CENTER: Earth textured sphere
    earth_texture = image_texture("assets/images/earthmap.jpg")
    earth_material = lambertian.from_texture(earth_texture)
    world.add(Sphere.stationary(point3(0, 0.5, 0), 0.5, earth_material))

    # RIGHT: Solid color sphere (blue)

    blue_material = lambertian.from_texture(checker_texture.from_colors(0.2, color(0.2, 0.3, 0.8), color(0.9, 0.9, 0.9)))
    world.add(Sphere.stationary(point3(1, 0.5, 0), 0.5, blue_material))

    # Create BVH and wrap it
    bvh = bvh_node.from_objects(world.objects, 0, len(world.objects))
    world = hittable_list()
    world.add(bvh)

    cam = camera()

    cam.aspect_ratio = 16.0 / 9.0
    cam.img_width = 600
    cam.samples_per_pixel = 50
    cam.max_depth = 10

    cam.vfov = 20
    cam.lookfrom = point3(0, 1, -5)  # Looking from slightly above
    cam.lookat = point3(0, 0.5, 0)   # Looking at center sphere
    cam.vup = vec3(0, 1, 0)
    cam.defocus_angle = 0.0
    cam.background = color(0.70, 0.80, 1.00)

    cam.render(world, "../temp/vol2_sec4_6_ver2.ppm")

#------------------------------------------------------------------------

def vol2_sec4_6_ver2():
    """Earth texture test scene - Taichi GPU renderer version - Interactive"""
    world = hittable_list()

    # Ground plane (large sphere below)
    ground_material = lambertian.from_color(color(0.5, 0.5, 0.5))
    world.add(Sphere.stationary(point3(0, -1000, 0), 1000, ground_material))

    # LEFT: Solid texture sphere (red)
    red_material = lambertian.from_texture(solid_color.from_color(color(0.8, 0.3, 0.3)))
    world.add(Sphere.stationary(point3(-1, 0.5, 0), 0.5, red_material))

    # CENTER: Earth textured sphere
    earth_texture = image_texture("assets/images/earthmap.jpg")
    earth_material = lambertian.from_texture(earth_texture)
    world.add(Sphere.stationary(point3(0, 0.5, 0), 0.5, earth_material))

    # RIGHT: Checker texture sphere
    blue_material = lambertian.from_texture(checker_texture.from_colors(0.2, color(0.2, 0.3, 0.8), color(0.9, 0.9, 0.9)))
    world.add(Sphere.stationary(point3(1, 0.5, 0), 0.5, blue_material))

    # Create BVH and wrap it
    bvh = bvh_node.from_objects(world.objects, 0, len(world.objects))
    world = hittable_list()
    world.add(bvh)

    cam = camera()

    cam.aspect_ratio = 16.0 / 9.0
    cam.img_width = 800
    cam.samples_per_pixel = 1000
    cam.max_depth = 50

    cam.vfov = 20
    cam.lookfrom = point3(0, 1, -5)  # Looking from slightly above
    cam.lookat = point3(0, 0.5, 0)   # Looking at center sphere
    cam.vup = vec3(0, 1, 0)
    cam.defocus_angle = 0.0
    cam.background = color(0.70, 0.80, 1.00)

    # Use interactive viewer
    viewer = InteractiveViewer(
        world,
        cam,
        "../temp/vol2_sec4_6_ver2_taichi.ppm"
    )
    viewer.render_interactive()

#------------------------------------------------------------------------

def subsurface_scattering():
    world = hittable_list()

    difflight = diffuse_light.from_color(color(4, 4, 4))
    world.add(quad(point3(-1, 0, 3), vec3(2, 0, 0), vec3(0, 2, 0), difflight))

    # Ground plane (large sphere below)
    ground_material = lambertian.from_color(color(0.5, 0.5, 0.5))
    world.add(Sphere.stationary(point3(0, -1000, 0), 1000, ground_material))

    # Dark green wax sphere (center)
    wax_material = subsurface_volumetric(
                                        albedo=color(0.2, 0.5, 0.2),
                                        scatter_coeff=0.08,    # low = light travels far inside
                                        absorb_coeff=0.8,    # low = minimal absorption
                                        g=0.7                 # forward scattering for soft look
                                    )
    world.add(Sphere.stationary(point3(0, 0.5, 0), 0.5, wax_material))

    # Regular lambertian for comparison (left)
    matte_green = lambertian.from_color(color(0.1, 0.3, 0.1))
    world.add(Sphere.stationary(point3(-1, 0.5, 0), 0.5, matte_green))

    # Glass sphere (right)
    pretext = noise_texture(50.0)
    noise_material = lambertian.from_texture(pretext)
    world.add(Sphere.stationary(point3(1, 0.5, 0), 0.5, noise_material))
    
    # Create BVH and wrap it
    # bvh = bvh_node.from_objects(world.objects, 0, len(world.objects))
    # world = hittable_list()
    # world.add(bvh)

    cam = camera()

    cam.aspect_ratio = 16.0 / 9.0
    cam.img_width = 100
    cam.samples_per_pixel = 40
    cam.max_depth = 15

    cam.vfov = 20
    cam.lookfrom = point3(0, 1, -5)  # Looking from slightly above
    cam.lookat = point3(0, 0.5, 0)   # Looking at center sphere
    cam.vup = vec3(0, 1, 0)
    cam.defocus_angle = 0.0
    cam.background = color(0.70, 0.80, 1.00)

    cam.render(world, "../temp/subsurface_scattering.ppm")

#------------------------------------------------------------------------

def vol2_sec5():
    world = hittable_list()

    pretext = noise_texture(4.0)
    ground_material = lambertian.from_texture(pretext)
    world.add(Sphere.stationary(point3(0, -1000, 0), 1000, ground_material))

    noise_material = lambertian.from_texture(pretext)
    world.add(Sphere.stationary(point3(0, 2, 0), 2, noise_material))

    # Create BVH and wrap it
    bvh = bvh_node.from_objects(world.objects, 0, len(world.objects))
    world = hittable_list()
    world.add(bvh)

    cam = camera()

    cam.aspect_ratio = 16.0 / 9.0
    cam.img_width = 500
    cam.samples_per_pixel = 20
    cam.max_depth = 10
    cam.vfov = 20
    cam.lookfrom = point3(13,2,3)
    cam.lookat = point3(0, 0, 0)
    cam.vup = vec3(0, 1, 0)
    cam.defocus_angle = 0.0
    cam.background = color(0.70, 0.80, 1.00)

    # Use Taichi GPU renderer
    renderer = RendererFactory.create(
        'taichi',
        world,
        cam,
        "../temp/vol2_sec5.ppm"
    )
    renderer.background_color = color(0.70, 0.80, 1.00)
    renderer.max_depth = 10
    renderer.render()

#------------------------------------------------------------------------

def emmission():
    world = hittable_list()

    pretext = noise_texture()
    ground_material = lambertian.from_color(color(0.5, 0.5, 0.5))
    world.add(Sphere.stationary(point3(0, -1000, 0), 1000, ground_material))

    noise_material = lambertian.from_texture(pretext)
    world.add(Sphere.stationary(point3(0, 2, 0), 2, noise_material))
    
    # Create BVH and wrap it
    bvh = bvh_node.from_objects(world.objects, 0, len(world.objects))
    world = hittable_list()
    world.add(bvh)

    cam = camera()

    cam.aspect_ratio = 16.0 / 9.0
    cam.img_width = 400
    cam.samples_per_pixel = 20
    cam.max_depth = 10

    cam.vfov = 20
    cam.lookfrom = point3(13,2,3)
    cam.lookat = point3(0, 0, 0)
    cam.vup = vec3(0, 1, 0)
    cam.defocus_angle = 0.0
    cam.background = color(0.70, 0.80, 1.00)

    cam.render(world, "../temp/vol2_sec5.ppm")

#------------------------------------------------------------------------

def vol2_sec6():
    world = hittable_list()

    # Materials
    left_red     = lambertian.from_color(color(1.0, 0.2, 0.2))
    back_green   = lambertian.from_color(color(0.2, 1.0, 0.2))
    right_blue   = lambertian.from_color(color(0.2, 0.2, 1.0))
    upper_orange = lambertian.from_color(color(1.0, 0.5, 0.0))
    lower_teal   = lambertian.from_color(color(0.2, 0.8, 0.8))

    # Quads
    world.add(quad(point3(-3, -2, 5), vec3(0, 0, -4), vec3(0, 4, 0), left_red))
    world.add(quad(point3(-2, -2, 0), vec3(4, 0, 0), vec3(0, 4, 0), back_green))
    world.add(quad(point3( 3, -2, 1), vec3(0, 0, 4), vec3(0, 4, 0), right_blue))
    world.add(quad(point3(-2, 3, 1), vec3(4, 0, 0), vec3(0, 0, 4), upper_orange))
    world.add(quad(point3(-2, -3, 5), vec3(4, 0, 0), vec3(0, 0, -4), lower_teal))

    # Create BVH and wrap it
    bvh = bvh_node.from_objects(world.objects, 0, len(world.objects))
    world = hittable_list()
    world.add(bvh)

    cam = camera()

    cam.aspect_ratio      = 1.0
    cam.img_width         = 400
    cam.samples_per_pixel = 50
    cam.max_depth         = 10

    cam.vfov     = 80
    cam.lookfrom = point3(0, 0, 9)
    cam.lookat   = point3(0, 0, 0)
    cam.vup      = vec3(0, 1, 0)
    cam.defocus_angle = 0.0
    cam.background = color(0.70, 0.80, 1.00)

    cam.render(world, "../temp/vol2_sec6.ppm")

#------------------------------------------------------------------------

def triangles():
    world = hittable_list()

    # Ground (large sphere below)
    ground_material = lambertian.from_color(color(0.5, 0.5, 0.5))
    world.add(Sphere.stationary(point3(0, -1000, 0), 1000, ground_material))

    # Triangle 1: Solid color texture (red)
    red_texture = solid_color.from_color(color(0.9, 0.2, 0.2))
    red_material = lambertian.from_texture(red_texture)
    tri1 = triangle(
        point3(-2, 0, -1),
        point3(-1, 2, -1),
        point3(0, 0, -1),
        red_material
    )
    world.add(tri1)

    # Triangle 2: Earth texture
    earth_texture = image_texture("assets/images/earthmap.jpg")
    earth_material = lambertian.from_texture(earth_texture)
    tri2 = triangle(
        point3(0.5, 0, 0),
        point3(1.5, 2, 0),
        point3(2.5, 0, 0),
        earth_material
    )
    world.add(tri2)

    # Triangle 3: Perlin noise texture
    perlin_texture = noise_texture(24.0)
    perlin_material = lambertian.from_texture(perlin_texture)
    tri3 = triangle(
        point3(-0.5, 0, 1),
        point3(0.5, 2, 1),
        point3(1.5, 0, 1),
        perlin_material
    )
    world.add(tri3)

    # Create BVH
    bvh = bvh_node.from_objects(world.objects, 0, len(world.objects))
    world = hittable_list()
    world.add(bvh)

    cam = camera()

    cam.aspect_ratio      = 16.0 / 9.0
    cam.img_width         = 400
    cam.samples_per_pixel = 50
    cam.max_depth         = 10

    cam.vfov     = 50
    cam.lookfrom = point3(0, 1, 5)
    cam.lookat   = point3(0.5, 1, 0)
    cam.vup      = vec3(0, 1, 0)
    cam.defocus_angle = 0.0
    cam.background = color(0.70, 0.80, 1.00)

    cam.render(world, "../temp/triangles.ppm")

#------------------------------------------------------------------------

def test_mesh():
    """Load and render a mesh using the GPU renderer."""

    print("Loading mesh model...")
    start_time = time.time()

    # Create material for the mesh
    mesh_material = lambertian.from_color(color(0.6, 0.4, 0.2))

    # Load the mesh from the models folder
    # The mesh class will automatically find the .obj file inside
    # IMPORTANT: You need to have an OBJ file in assets/models/
    # Example: assets/models/house/house.obj or assets/models/teapot.obj
    model_mesh = mesh(
        model_path="assets/models",  # Directory containing .obj files
        obj_filename="teapot.obj",   # Specific .obj file to load
        mat=mesh_material,
        scale=0.1,  # Scale down to 10% of original size
        offset=point3(0, 0, 0)
    )

    load_time = time.time() - start_time
    print(f"✓ Loaded {model_mesh.triangle_count()} triangles in {load_time:.2f}s")
    print(f"  Bounding box: {model_mesh.bounding_box()}")

    # Build scene
    world = hittable_list()

    # Add ground plane
    ground_material = lambertian.from_color(color(0.5, 0.5, 0.5))
    from core import Sphere
    world.add(Sphere.stationary(point3(0, -1000, 0), 1000, ground_material))

    # Add the mesh (triangles will be automatically extracted by GPU compiler)
    world.add(model_mesh)

    # Optional: Add a light source for better visibility
    light = diffuse_light.from_color(color(4, 4, 4))
    world.add(Sphere.stationary(point3(10, 10, -10), 2, light))

    # Create BVH for the entire scene
    bvh = bvh_node.from_objects(world.objects, 0, len(world.objects))
    world = hittable_list()
    world.add(bvh)

    # Setup camera
    cam = camera()
    cam.aspect_ratio = 16.0 / 9.0
    cam.img_width = 800
    cam.samples_per_pixel = 100
    cam.max_depth = 10

    # Camera position - adjusted for scaled model
    cam.vfov = 40
    cam.lookfrom = point3(15, 5, 10)
    cam.lookat = point3(0, 1.5, 0)
    cam.vup = vec3(0, 1, 0)
    cam.defocus_angle = 0

    # Use GPU renderer for MAXIMUM speed!
    print("\nRendering with GPU...")
    renderer = RendererFactory.create(
        'taichi',  # GPU renderer
        world,
        cam,
        "../temp/test_mesh.ppm"
    )
    renderer.background_color = color(0, 0, 0)  # Black background
    renderer.max_depth = 10
    renderer.render()

    print(f"✓ Done! Rendered {model_mesh.triangle_count()} triangles on GPU")
    print("  Output: ../temp/test_mesh.ppm")


def test_mesh_interactive():
    """
    Load and render a mesh using the GPU renderer with interactive camera controls.

    Controls:
        - Left-click and drag: Rotate camera around the mesh
        - Close window: Exit and save final image
    """
    print("="*80)
    print("INTERACTIVE MESH VIEWER")
    print("="*80)
    print("Loading mesh model...")
    start_time = time.time()

    # Create material for the mesh
    mesh_material = lambertian.from_color(color(0.6, 0.4, 0.2))
    teapot_material = dielectric(1.5)

    # Load the mesh from the models folder
    # IMPORTANT: You need to have an OBJ file in assets/models/
    # Example: assets/models/house/house.obj or assets/models/teapot.obj
    model_mesh = mesh(
        model_path="assets/models",  # Directory containing .obj files
        obj_filename="teapot.obj",   # Specific .obj file to load
        mat=teapot_material,
        scale=0.03,  # Scale down to 10% of original size
        offset=point3(0, 1.3, 0)
    )

    load_time = time.time() - start_time
    print(f"✓ Loaded {model_mesh.triangle_count()} triangles in {load_time:.2f}s")
    print(f"  Bounding box: {model_mesh.bounding_box()}")

    # Build scene
    world = hittable_list()

    # Add ground plane with checker texture
    checker = checker_texture.from_colors(0.5, color(0.2, 0.3, 0.1), color(0.9, 0.9, 0.9))
    ground_material = lambertian.from_texture(checker)
    from core import Sphere
    world.add(Sphere.stationary(point3(0, -1000, 0), 1000, ground_material))

    # Add the mesh (triangles will be automatically extracted by GPU compiler)
    world.add(model_mesh)

    # Add quad diffuse light to the northwest, 5 units from teapot
    light_material = diffuse_light.from_color(color(7, 7, 7))
    world.add(quad(
        point3(-4, 4, -3),      # Corner position (northwest, 5 units away)
        vec3(2, 0, 0),          # Width (2 units wide)
        vec3(0, 0, 2),          # Height (2 units tall)
        light_material
    ))

    # Optional: Add some reference spheres to show scale
    ref_material = lambertian.from_color(color(0.3, 0.3, 0.8))
    world.add(Sphere.stationary(point3(-3, 0.5, 0), 0.5, ref_material))
    world.add(Sphere.stationary(point3(3, 0.5, 0), 0.5, ref_material))

    # Create BVH for the entire scene
    bvh = bvh_node.from_objects(world.objects, 0, len(world.objects))
    world = hittable_list()
    world.add(bvh)

    # Setup camera
    cam = camera()
    cam.aspect_ratio = 16.0 / 9.0
    cam.img_width = 1280  # Higher resolution for interactive viewing
    cam.samples_per_pixel = 1000  # High quality progressive rendering

    # Camera position - adjusted for scaled model
    cam.vfov = 40
    cam.lookfrom = point3(6, 3, 6)  # Position camera to view mesh
    cam.lookat = point3(0, 1.5, 0)    # Look at center of mesh
    cam.vup = vec3(0, 1, 0)
    cam.defocus_angle = 0

    # Use Interactive GPU viewer with mouse controls
    viewer = InteractiveViewer(
        world,
        cam,
        "../temp/test_mesh_interactive.ppm"
    )
    viewer.background_color = color(0.1, 0.05, 0.05)  # Dark gray background
    # viewer.background_color = color(0.70, 0.80, 1.00)  # Light blue background
    viewer.max_depth = 50

    print("\n" + "="*80)
    print("Interactive Controls:")
    print("  • Left-click + drag: Rotate camera around the mesh")
    print("  • Horizontal drag: Rotate left/right (yaw)")
    print("  • Vertical drag: Rotate up/down (pitch)")
    print("  • Close window: Save final image and exit")
    print("="*80)
    print(f"\nRendering {model_mesh.triangle_count()} triangles on GPU...")
    print("Progressive refinement will continue until window is closed.\n")

    # Start interactive rendering
    viewer.render_interactive()

    print(f"\n✓ Done! Final image saved to: ../temp/test_mesh_interactive.ppm")
    print(f"  Rendered {model_mesh.triangle_count()} triangles with GPU acceleration")

#------------------------------------------------------------------------

def simple_light():
    world = hittable_list()

    pertext = noise_texture(4)
    world.add(Sphere.stationary(point3(0, -1000, 0), 1000, lambertian.from_texture(pertext)))
    world.add(Sphere.stationary(point3(0, 2, 0), 2, lambertian.from_texture(pertext)))

    difflight = diffuse_light.from_color(color(4, 4, 4))
    world.add(Sphere.stationary(point3(0, 7, 0), 2, difflight))
    world.add(quad(point3(3, 1, -2), vec3(2, 0, 0), vec3(0, 2, 0), difflight))

    # Create BVH and wrap it
    bvh = bvh_node.from_objects(world.objects, 0, len(world.objects))
    world = hittable_list()
    world.add(bvh)

    cam = camera()

    cam.aspect_ratio = 16.0 / 9.0
    cam.img_width = 800
    cam.samples_per_pixel = 200
    cam.max_depth = 50

    cam.vfov = 20
    cam.lookfrom = point3(26, 3, 6)
    cam.lookat = point3(0, 2, 0)
    cam.vup = vec3(0, 1, 0)

    cam.defocus_angle = 0

    # Use Taichi GPU renderer for MAXIMUM efficiency!
    renderer = RendererFactory.create(
        'taichi',
        world,
        cam,
        "../temp/simple_light.ppm"
    )
    renderer.background_color = color(0, 0, 0)
    renderer.max_depth = 50
    renderer.render()

#------------------------------------------------------------------------

def box(a: point3, b: point3, mat: material, angle: float = 0.0) -> hittable_list:
    """Create an axis-aligned box defined by two corner points.

    Args:
        a: First corner point
        b: Opposite corner point
        mat: Material for all box faces
        angle: Rotation angle in degrees around Y axis (default: 0.0)
    """
    sides = hittable_list()

    # Construct the two opposite vertices with the minimum and maximum coordinates.
    min_pt = vec3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z))
    max_pt = vec3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z))

    dx = vec3(max_pt.x - min_pt.x, 0, 0)
    dy = vec3(0, max_pt.y - min_pt.y, 0)
    dz = vec3(0, 0, max_pt.z - min_pt.z)

    # Helper function to rotate a vector around Y axis
    def rotate_y(v: vec3, theta_rad: float) -> vec3:
        cos_theta = cos(theta_rad)
        sin_theta = sin(theta_rad)
        return vec3(
            cos_theta * v.x + sin_theta * v.z,
            v.y,
            -sin_theta * v.x + cos_theta * v.z
        )

    # Apply rotation if angle is non-zero
    if angle != 0.0:
        theta = radians(angle)

        # Rotate corner positions around the center of the box
        center = (min_pt + max_pt) * 0.5

        # Create vertices relative to center, rotate them, then translate back
        def rotate_point(p: vec3) -> vec3:
            p_centered = p - center
            p_rotated = rotate_y(p_centered, theta)
            return p_rotated + center

        # Rotate the delta vectors
        dx = rotate_y(dx, theta)
        dy_rot = rotate_y(dy, theta)
        dz = rotate_y(dz, theta)

        # Rotate and add the quads
        sides.add(quad(rotate_point(vec3(min_pt.x, min_pt.y, max_pt.z)),  dx,  dy_rot, mat))  # front
        sides.add(quad(rotate_point(vec3(max_pt.x, min_pt.y, max_pt.z)), -dz,  dy_rot, mat))  # right
        sides.add(quad(rotate_point(vec3(max_pt.x, min_pt.y, min_pt.z)), -dx,  dy_rot, mat))  # back
        sides.add(quad(rotate_point(vec3(min_pt.x, min_pt.y, min_pt.z)),  dz,  dy_rot, mat))  # left
        sides.add(quad(rotate_point(vec3(min_pt.x, max_pt.y, max_pt.z)),  dx, -dz, mat))  # top
        sides.add(quad(rotate_point(vec3(min_pt.x, min_pt.y, min_pt.z)),  dx,  dz, mat))  # bottom
    else:
        # No rotation - use original code
        sides.add(quad(vec3(min_pt.x, min_pt.y, max_pt.z),  dx,  dy, mat))  # front
        sides.add(quad(vec3(max_pt.x, min_pt.y, max_pt.z), -dz,  dy, mat))  # right
        sides.add(quad(vec3(max_pt.x, min_pt.y, min_pt.z), -dx,  dy, mat))  # back
        sides.add(quad(vec3(min_pt.x, min_pt.y, min_pt.z),  dz,  dy, mat))  # left
        sides.add(quad(vec3(min_pt.x, max_pt.y, max_pt.z),  dx, -dz, mat))  # top
        sides.add(quad(vec3(min_pt.x, min_pt.y, min_pt.z),  dx,  dz, mat))  # bottom

    return sides

#------------------------------------------------------------------------

def cornell_box():
    world = hittable_list()

    red = lambertian.from_color(color(0.65, 0.05, 0.05))
    white = lambertian.from_color(color(0.73, 0.73, 0.73))
    green = lambertian.from_color(color(0.12, 0.45, 0.15))
    light = diffuse_light.from_color(color(15, 15, 15))

    world.add(quad(point3(555, 0, 0), vec3(0, 0, 555), vec3(0, 555, 0), green))
    world.add(quad(point3(0, 0, 0), vec3(0, 555, 0), vec3(0, 0, 555), red))
    world.add(quad(point3(343, 554, 332), vec3(-130, 0, 0), vec3(0, 0, -105), light))
    world.add(quad(point3(0, 0, 0), vec3(0, 0, 555), vec3(555, 0, 0), white))
    world.add(quad(point3(555, 555, 555), vec3(-555, 0, 0), vec3(0, 0, -555), white))
    world.add(quad(point3(0, 0, 555), vec3(0, 555, 0), vec3(555, 0, 0), white))

    world.add(box(point3(130, 0, 65), point3(295, 165, 230), white, -18))
    world.add(box(point3(265, 0, 295), point3(430, 330, 460), white, 15))

    # Create BVH and wrap it
    bvh = bvh_node.from_objects(world.objects, 0, len(world.objects))
    world = hittable_list()
    world.add(bvh)

    cam = camera()

    cam.aspect_ratio = 1.0
    cam.img_width = 800  # Lower resolution for faster interactive response
    cam.samples_per_pixel = 500  # 100 samples is good for interactive viewing

    cam.vfov = 40
    cam.lookfrom = point3(278, 278, -800)
    cam.lookat = point3(278, 278, 0)
    cam.vup = vec3(0, 1, 0)

    cam.defocus_angle = 0

#-------------------------------------------------------
    # Use Interactive GPU renderer with mouse-controlled camera:
    from render_server.interactive_viewer import InteractiveViewer

    renderer = InteractiveViewer(world, cam, "../temp/cornell_box.ppm")
    renderer.background_color = color(0, 0, 0)
    renderer.max_depth = 50

    print("\n" + "="*60)
    print("INTERACTIVE CORNELL BOX")
    print("="*60)
    print("Controls:")
    print("  • Left-click + drag: Rotate camera around the scene")
    print("  • Horizontal drag: Rotate left/right (yaw)")
    print("  • Vertical drag: Rotate up/down (pitch)")
    print("  • Camera resets samples when rotated")
    print("="*60 + "\n")

    renderer.render_interactive()
#-------------------------------------------------------
    # CPU Render:
    # cam.max_depth = 10
    # cam.background = color(0, 0, 0)
    # cam.russian_roulette_enabled = False
    # print("\nRendering scene...")
    # cam.render(world, "../temp/cornell_box.ppm")
    # print("✓ Done! Check ../temp/cornell_box.ppm")

#------------------------------------------------------------------------

def cornell_smoke():
    """Cornell box with smoke volumes - Interactive version"""
    world = hittable_list()

    # Materials
    red = lambertian.from_color(color(0.65, 0.05, 0.05))
    white = lambertian.from_color(color(0.73, 0.73, 0.73))
    green = lambertian.from_color(color(0.12, 0.45, 0.15))
    light = diffuse_light.from_color(color(7, 7, 7))

    # Cornell box walls
    world.add(quad(point3(555, 0, 0), vec3(0, 555, 0), vec3(0, 0, 555), green))      # right wall
    world.add(quad(point3(0, 0, 0), vec3(0, 555, 0), vec3(0, 0, 555), red))          # left wall
    world.add(quad(point3(113, 554, 127), vec3(330, 0, 0), vec3(0, 0, 305), light))  # light
    world.add(quad(point3(0, 555, 0), vec3(555, 0, 0), vec3(0, 0, 555), white))      # ceiling
    world.add(quad(point3(0, 0, 0), vec3(555, 0, 0), vec3(0, 0, 555), white))        # floor
    world.add(quad(point3(0, 0, 555), vec3(555, 0, 0), vec3(0, 555, 0), white))      # back wall

    # Create boxes (rotated and positioned)
    # Box 1: tall box on the right (rotated 15 degrees)
    box1 = box(point3(265, 0, 295), point3(430, 330, 460), white, 15)

    # Box 2: short box on the left (rotated -18 degrees)
    box2 = box(point3(130, 0, 65), point3(295, 165, 230), white, -18)

    # Wrap boxes in constant medium (smoke/fog)
    # Box 1: black smoke (density = 0.01)
    world.add(constant_medium.from_color(box1, color(0, 0, 0), 0.01))

    # Box 2: white smoke (density = 0.01)
    world.add(constant_medium.from_color(box2, color(1, 1, 1), 0.01))

    # Create BVH
    bvh = bvh_node.from_objects(world.objects, 0, len(world.objects))
    world = hittable_list()
    world.add(bvh)

    # Camera setup
    cam = camera()
    cam.aspect_ratio = 1.0
    cam.img_width = 800
    cam.samples_per_pixel = 1000

    cam.vfov = 40
    cam.lookfrom = point3(278, 278, -800)
    cam.lookat = point3(278, 278, 0)
    cam.vup = vec3(0, 1, 0)
    cam.defocus_angle = 0

    # Use Interactive GPU renderer
    viewer = InteractiveViewer(world, cam, "../temp/cornell_smoke.ppm")
    viewer.background_color = color(0, 0, 0)
    viewer.max_depth = 50

    viewer.render_interactive()

#------------------------------------------------------------------------

def vol2_final_scene():
    """The final scene from Ray Tracing: The Next Week - Interactive version"""
    # Ground boxes
    boxes1 = hittable_list()
    ground = lambertian.from_color(color(0.48, 0.83, 0.53))

    boxes_per_side = 20
    for i in range(boxes_per_side):
        for j in range(boxes_per_side):
            w = 100.0
            x0 = -1000.0 + i * w
            z0 = -1000.0 + j * w
            y0 = 0.0
            x1 = x0 + w
            y1 = random.uniform(1, 101)
            z1 = z0 + w

            boxes1.add(box(point3(x0, y0, z0), point3(x1, y1, z1), ground))

    world = hittable_list()

    # Add ground boxes as BVH
    world.add(bvh_node.from_objects(boxes1.objects, 0, len(boxes1.objects)))

    # Light
    light = diffuse_light.from_color(color(7, 7, 7))
    world.add(quad(point3(123, 554, 147), vec3(300, 0, 0), vec3(0, 0, 265), light))

    # Moving sphere
    center1 = point3(400, 400, 200)
    center2 = center1 + vec3(30, 0, 0)
    sphere_material = lambertian.from_color(color(0.7, 0.3, 0.1))
    world.add(Sphere.moving(center1, center2, 50, sphere_material))

    # Glass sphere
    world.add(Sphere.stationary(point3(260, 150, 45), 50, dielectric(1.5)))

    # Metal sphere
    world.add(Sphere.stationary(
        point3(0, 150, 145), 50, metal(color(0.8, 0.8, 0.9), 1.0)
    ))

    # Glass sphere with volume inside
    boundary = Sphere.stationary(point3(360, 150, 145), 70, dielectric(1.5))
    world.add(boundary)
    world.add(constant_medium.from_color(boundary, color(0.2, 0.4, 0.9), 0.2))

    # Global fog
    boundary = Sphere.stationary(point3(0, 0, 0), 5000, dielectric(1.5))
    world.add(constant_medium.from_color(boundary, color(1, 1, 1), 0.0001))

    # Earth sphere
    emat = lambertian.from_texture(image_texture("assets/images/earthmap.jpg"))
    world.add(Sphere.stationary(point3(400, 200, 400), 100, emat))

    # Perlin noise sphere
    pertext = noise_texture(0.2)
    world.add(Sphere.stationary(point3(220, 280, 300), 80, lambertian.from_texture(pertext)))

    # Box of spheres (translated to vec3(-100, 270, 395))
    boxes2 = hittable_list()
    white = lambertian.from_color(color(0.73, 0.73, 0.73))
    ns = 1000
    offset = vec3(-100, 270, 395)
    for j in range(ns):
        random_pos = point3.random(0, 165)
        boxes2.add(Sphere.stationary(random_pos + offset, 10, white))

    world.add(bvh_node.from_objects(boxes2.objects, 0, len(boxes2.objects)))

    # Create final BVH
    bvh = bvh_node.from_objects(world.objects, 0, len(world.objects))
    world = hittable_list()
    world.add(bvh)

    # Camera setup
    cam = camera()

    cam.aspect_ratio = 1.0
    cam.img_width = 1000
    cam.samples_per_pixel = 10000

    cam.vfov = 40
    cam.lookfrom = point3(478, 278, -600)
    cam.lookat = point3(278, 278, 0)
    cam.vup = vec3(0, 1, 0)

    cam.defocus_angle = 0

    # Use Interactive GPU renderer
    viewer = InteractiveViewer(world, cam, "../temp/vol2_final_scene.ppm")
    viewer.background_color = color(0, 0, 0)
    viewer.max_depth = 50

    viewer.render_interactive()

#------------------------------------------------------------------------

def vol2_final_scene_simple():
    """Simplified version of the final scene for faster testing"""
    pass

#------------------------------------------------------------------------

def vol2_final_scene_comparison():
    """
    Compare megakernel vs wavefront on the complex vol2_final_scene.
    This scene has 1000+ objects and should show wavefront's benefits.
    """
    # Ground boxes
    boxes1 = hittable_list()
    ground = lambertian.from_color(color(0.48, 0.83, 0.53))

    boxes_per_side = 20
    for i in range(boxes_per_side):
        for j in range(boxes_per_side):
            w = 100.0
            x0 = -1000.0 + i * w
            z0 = -1000.0 + j * w
            y0 = 0.0
            x1 = x0 + w
            y1 = random.uniform(1, 101)
            z1 = z0 + w

            boxes1.add(box(point3(x0, y0, z0), point3(x1, y1, z1), ground))

    world = hittable_list()

    # Add ground boxes as BVH
    world.add(bvh_node.from_objects(boxes1.objects, 0, len(boxes1.objects)))

    # Light
    light = diffuse_light.from_color(color(7, 7, 7))
    world.add(quad(point3(123, 554, 147), vec3(300, 0, 0), vec3(0, 0, 265), light))

    # Moving sphere
    center1 = point3(400, 400, 200)
    center2 = center1 + vec3(30, 0, 0)
    sphere_material = lambertian.from_color(color(0.7, 0.3, 0.1))
    world.add(Sphere.moving(center1, center2, 50, sphere_material))

    # Glass sphere
    world.add(Sphere.stationary(point3(260, 150, 45), 50, dielectric(1.5)))

    # Metal sphere
    world.add(Sphere.stationary(
        point3(0, 150, 145), 50, metal(color(0.8, 0.8, 0.9), 1.0)
    ))

    # Glass sphere with volume inside
    boundary = Sphere.stationary(point3(360, 150, 145), 70, dielectric(1.5))
    world.add(boundary)
    world.add(constant_medium.from_color(boundary, color(0.2, 0.4, 0.9), 0.2))

    # Global fog
    boundary = Sphere.stationary(point3(0, 0, 0), 5000, dielectric(1.5))
    world.add(constant_medium.from_color(boundary, color(1, 1, 1), 0.0001))

    # Earth sphere
    emat = lambertian.from_texture(image_texture("assets/images/earthmap.jpg"))
    world.add(Sphere.stationary(point3(400, 200, 400), 100, emat))

    # Perlin noise sphere
    pertext = noise_texture(0.2)
    world.add(Sphere.stationary(point3(220, 280, 300), 80, lambertian.from_texture(pertext)))

    # Box of spheres (1000 small spheres - main complexity source)
    boxes2 = hittable_list()
    white = lambertian.from_color(color(0.73, 0.73, 0.73))
    ns = 1000
    offset = vec3(-100, 270, 395)
    for j in range(ns):
        random_pos = point3.random(0, 165)
        boxes2.add(Sphere.stationary(random_pos + offset, 10, white))

    world.add(bvh_node.from_objects(boxes2.objects, 0, len(boxes2.objects)))

    # Create final BVH
    bvh = bvh_node.from_objects(world.objects, 0, len(world.objects))
    world = hittable_list()
    world.add(bvh)

    # Camera setup - reduced resolution and samples for faster comparison
    cam = camera()
    cam.aspect_ratio = 1.0
    cam.img_width = 1000  # Reduced from 1000 for faster testing
    cam.samples_per_pixel = 10000  # Reduced from 10000 for faster testing

    cam.vfov = 40
    cam.lookfrom = point3(478, 278, -600)
    cam.lookat = point3(278, 278, 0)
    cam.vup = vec3(0, 1, 0)
    cam.defocus_angle = 0

#------------------------------------------------------------------------

    # CPU Render:
    # cam.max_depth = 15
    # cam.background = color(0, 0, 0)
    # cam.russian_roulette_enabled = False
    # print("\nRendering scene...")
    # cam.render(world, "../temp/vol2_final_interactive_cpu.ppm")
    # print("✓ Done! Check ../temp/vol2_final_interactive_cpu.ppm")

#------------------------------------------------------------------------

    # Test 1: Megakernel (traditional depth-first)
    print("\n" + "="*80)
    print("TEST 1: MEGAKERNEL MODE on vol2_final_scene (1000+ objects)")
    print("="*80)

    # renderer = RendererFactory.create(
    #     'taichi',
    #     world,
    #     cam,
    #     "../temp/vol2_final_megakernel.png"
    # )
    # renderer.background_color = color(0, 0, 0)
    # renderer.max_depth = 50

    # start_mega = time.time()
    # renderer.render(enable_preview=True)  # <-- MEGAKERNEL: calls render() method
    # time_mega = time.time() - start_mega

    # Interactive rendering with mouse controls
    viewer = InteractiveViewer(
        world,
        cam,
        "../temp/vol2_final_interactive.png"
    )
    viewer.background_color = color(0, 0, 0)
    viewer.max_depth = 50

    # print("\nInteractive Controls:")
    # print("  - Left-click and drag to rotate camera")
    # print("  - Close window to exit and save final image")
    # print()

    viewer.render_interactive()

#------------------------------------------------------------------------

    # # Test 2: Wavefront (breadth-first)
    # # NOTE: Same TaichiRenderer class, just calling a different method!
    # print("\n" + "="*80)
    # print("TEST 2: WAVEFRONT MODE on vol2_final_scene (1000+ objects)")
    # print("="*80)

    # # Re-create renderer to get fresh state (clear accumulation buffer)
    # renderer = RendererFactory.create(
    #     'taichi',
    #     world,
    #     cam,
    #     "../temp/vol2_final_wavefront.png"
    # )
    # renderer.background_color = color(0, 0, 0)
    # renderer.max_depth = 50

    # start_wave = time.time()
    # renderer.render_wavefront(enable_preview=False)  # <-- WAVEFRONT: calls render_wavefront() method
    # time_wave = time.time() - start_wave

    # # Print comparison
    # print("\n" + "="*80)
    # print("PERFORMANCE COMPARISON - COMPLEX SCENE")
    # print("="*80)
    # print(f"Scene complexity: 1000+ spheres, volumes, textures, complex materials")
    # print(f"Resolution: {cam.img_width}x{cam.img_width} | Samples: {cam.samples_per_pixel}")
    # print(f"\nMegakernel Time:  {time_mega:.2f}s")
    # print(f"Wavefront Time:   {time_wave:.2f}s")
    # if time_wave < time_mega:
    #     print(f"Speedup:          {time_mega/time_wave:.2f}x (wavefront is FASTER!)")
    # else:
    #     print(f"Slowdown:         {time_wave/time_mega:.2f}x (megakernel still faster)")
    # print(f"\nImages saved to:")
    # print(f"  - ../temp/vol2_final_megakernel.png")
    # print(f"  - ../temp/vol2_final_wavefront.png")
    # print("="*80)

#------------------------------------------------------------------------

def wavefront_comparison():
    """
    Compare megakernel vs wavefront path tracing performance.
    Tests both rendering modes on the same scene.
    """
    # Create a moderately complex scene for comparison
    world = hittable_list()

    # Ground
    ground = lambertian.from_color(color(0.5, 0.5, 0.5))
    world.add(Sphere.stationary(point3(0, -1000, 0), 1000, ground))

    # Random spheres
    for a in range(-3, 3):
        for b in range(-3, 3):
            choose_mat = random.random()
            center = point3(a + 0.9 * random.random(), 0.2, b + 0.9 * random.random())

            if (center - point3(4, 0.2, 0)).length() > 0.9:
                if choose_mat < 0.6:
                    # Diffuse
                    albedo = color.random() * color.random()
                    sphere_material = lambertian.from_color(albedo)
                    world.add(Sphere.stationary(center, 0.2, sphere_material))
                elif choose_mat < 0.85:
                    # Metal
                    albedo = color.random(0.5, 1)
                    fuzz = random.uniform(0, 0.5)
                    sphere_material = metal(albedo, fuzz)
                    world.add(Sphere.stationary(center, 0.2, sphere_material))
                else:
                    # Glass
                    sphere_material = dielectric(1.5)
                    world.add(Sphere.stationary(center, 0.2, sphere_material))

    # Three large spheres
    material1 = dielectric(1.5)
    world.add(Sphere.stationary(point3(0, 1, 0), 1.0, material1))

    material2 = lambertian.from_color(color(0.4, 0.2, 0.1))
    world.add(Sphere.stationary(point3(-4, 1, 0), 1.0, material2))

    material3 = metal(color(0.7, 0.6, 0.5), 0.0)
    world.add(Sphere.stationary(point3(4, 1, 0), 1.0, material3))

    # Add light source
    light = diffuse_light.from_color(color(4, 4, 4))
    world.add(Sphere.stationary(point3(0, 5, 0), 1.5, light))

    # Build BVH
    bvh = bvh_node.from_objects(world.objects, 0, len(world.objects))
    world = hittable_list()
    world.add(bvh)

    # Camera setup
    cam = camera()
    cam.aspect_ratio = 16.0 / 9.0
    cam.img_width = 800
    cam.samples_per_pixel = 200  # Low sample count for quick comparison

    cam.vfov = 20
    cam.lookfrom = point3(13, 2, 3)
    cam.lookat = point3(0, 0, 0)
    cam.vup = vec3(0, 1, 0)
    cam.defocus_angle = 0.0

    # Test 1: Megakernel (traditional depth-first)
    print("\n" + "="*80)
    print("TEST 1: MEGAKERNEL MODE (depth-first ray tracing)")
    print("="*80)

    renderer = RendererFactory.create(
        'taichi',
        world,
        cam,
        "../temp/comparison_megakernel.png"
    )
    renderer.background_color = color(0.70, 0.80, 1.00)
    renderer.max_depth = 50

    start_mega = time.time()
    renderer.render(enable_preview=False)  # <-- MEGAKERNEL: calls render() method
    time_mega = time.time() - start_mega

    # Test 2: Wavefront (breadth-first)
    # NOTE: Same TaichiRenderer class, just calling a different method!
    print("\n" + "="*80)
    print("TEST 2: WAVEFRONT MODE (breadth-first ray tracing)")
    print("="*80)

    # Re-create renderer to get fresh state (clear accumulation buffer)
    renderer = RendererFactory.create(
        'taichi',
        world,
        cam,
        "../temp/comparison_wavefront.png"
    )
    renderer.background_color = color(0.70, 0.80, 1.00)
    renderer.max_depth = 50

    start_wave = time.time()
    renderer.render_wavefront(enable_preview=False)  # <-- WAVEFRONT: calls render_wavefront() method
    time_wave = time.time() - start_wave

    # Print comparison
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON")
    print("="*80)
    print(f"Megakernel Time:  {time_mega:.2f}s")
    print(f"Wavefront Time:   {time_wave:.2f}s")
    print(f"Speedup:          {time_mega/time_wave:.2f}x" if time_wave < time_mega else f"Slowdown:         {time_wave/time_mega:.2f}x")
    print(f"\nImages saved to:")
    print(f"  - ../temp/comparison_megakernel.png")
    print(f"  - ../temp/comparison_wavefront.png")
    print("="*80)
