from core.material import *
from core.texture import checker_texture, image_texture, noise_texture
from core import quad
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
    """Load and render the silo mesh."""

    print("Loading silo model...")
    start_time = time.time()

    # Create material for the silo
    wood_material = lambertian.from_color(color(0.6, 0.4, 0.2))

    # Load the mesh from the models folder
    # The mesh class will automatically find the .obj file inside
    silo = mesh(
        model_path="assets/models/house",
        mat=wood_material,
        scale=0.1,  # Scale down to 10% of original size
        offset=point3(0, 0, 0)
    )

    load_time = time.time() - start_time
    print(f"✓ Loaded {silo.triangle_count()} triangles in {load_time:.2f}s")
    print(f"  Bounding box: {silo.bounding_box()}")

    # Build scene
    world = hittable_list()

    # Add ground plane
    ground_material = lambertian.from_color(color(0.5, 0.5, 0.5))
    from core import Sphere
    world.add(Sphere.stationary(point3(0, -1000, 0), 1000, ground_material))

    # Add the silo (it already has internal BVH built automatically)
    world.add(silo)

    # Setup camera
    cam = camera()
    cam.aspect_ratio = 16.0 / 9.0
    cam.img_width = 50
    cam.samples_per_pixel = 10
    cam.max_depth = 3

    # Camera position - adjusted for scaled model
    # Position camera to look at the model from a good angle
    cam.vfov = 20  # Wider field of view
    cam.lookfrom = point3(15, 5, 10)  # Further back and higher
    cam.lookat = point3(0, 1.5, 0)  # Look at center of model
    cam.vup = vec3(0, 1, 0)
    cam.defocus_angle = 0  # No depth of field
    cam.background = color(0.70, 0.80, 1.00)

    # Render
    print("\nRendering scene...")
    cam.render(world, "../temp/test_mesh.ppm")
    print("✓ Done! Check ../temp/test_mesh.ppm")

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

