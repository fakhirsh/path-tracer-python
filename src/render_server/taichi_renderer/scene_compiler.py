"""
Scene Compiler: Converts Python scene representation to GPU-friendly arrays.
Pure Python - no Taichi code here.
"""

import numpy as np
from typing import List, Tuple, Dict, Any

# Geometry type constants (must match kernels.py)
PRIM_SPHERE = 0
PRIM_TRIANGLE = 1
PRIM_QUAD = 2

# Material type constants
MAT_LAMBERTIAN = 0
MAT_METAL = 1
MAT_DIELECTRIC = 2

# Texture type constants
TEX_SOLID = 0
TEX_CHECKER = 1


def extract_spheres(world) -> List:
    """
    Recursively extract all Sphere objects from world hierarchy.
    Handles: Sphere, hittable_list, bvh_node
    Returns: List of Sphere objects
    """
    from core.sphere import Sphere
    from core.hittable_list import hittable_list
    from core.bvh_node import bvh_node

    spheres = []

    # Check if this is a Sphere
    if isinstance(world, Sphere):
        spheres.append(world)
    # Check if this is a hittable_list
    elif isinstance(world, hittable_list):
        for item in world.objects:
            spheres.extend(extract_spheres(item))
    # Check if this is a bvh_node
    elif isinstance(world, bvh_node):
        if world.left is not None:
            spheres.extend(extract_spheres(world.left))
        if world.right is not None:
            spheres.extend(extract_spheres(world.right))

    return spheres


def compile_geometry(spheres: List) -> Dict[str, np.ndarray]:
    """
    Pack sphere geometry into numpy arrays for GPU upload.

    Returns dict with:
        'sphere_data': np.ndarray of shape (N, 4) dtype=float32  # [cx, cy, cz, radius]
        'num_spheres': int
    """
    n = len(spheres)
    sphere_data = np.zeros((n, 4), dtype=np.float32)

    for i, sphere in enumerate(spheres):
        # Get center at t=0 (no motion blur in refactored version)
        center = sphere.center.at(0.0)
        sphere_data[i, 0] = center.x
        sphere_data[i, 1] = center.y
        sphere_data[i, 2] = center.z
        sphere_data[i, 3] = sphere.radius

    return {
        'sphere_data': sphere_data,
        'num_spheres': n
    }


def compile_materials(spheres: List) -> Dict[str, np.ndarray]:
    """
    Extract material properties from spheres.

    Returns dict with:
        'material_type': np.ndarray of shape (N,) dtype=int32
        'material_albedo': np.ndarray of shape (N, 3) dtype=float32
        'material_fuzz': np.ndarray of shape (N,) dtype=float32
        'material_ir': np.ndarray of shape (N,) dtype=float32
        'texture_type': np.ndarray of shape (N,) dtype=int32
        'texture_scale': np.ndarray of shape (N,) dtype=float32
        'texture_color1': np.ndarray of shape (N, 3) dtype=float32
        'texture_color2': np.ndarray of shape (N, 3) dtype=float32
    """
    n = len(spheres)

    material_type = np.zeros(n, dtype=np.int32)
    material_albedo = np.zeros((n, 3), dtype=np.float32)
    material_fuzz = np.zeros(n, dtype=np.float32)
    material_ir = np.zeros(n, dtype=np.float32)

    texture_type = np.zeros(n, dtype=np.int32)
    texture_scale = np.ones(n, dtype=np.float32)
    texture_color1 = np.zeros((n, 3), dtype=np.float32)
    texture_color2 = np.zeros((n, 3), dtype=np.float32)

    for i, sphere in enumerate(spheres):
        mat = sphere.material
        mat_type_name = type(mat).__name__
        center = sphere.center.at(0.0)

        if mat_type_name == 'lambertian':
            material_type[i] = MAT_LAMBERTIAN

            # Extract texture information
            tex_type_name = type(mat.tex).__name__

            if tex_type_name == 'checker_texture':
                # Checker texture - store scale and two colors
                texture_type[i] = TEX_CHECKER
                texture_scale[i] = 1.0 / mat.tex.inv_scale  # Convert back to scale

                # Get the two colors from even and odd textures
                color1 = mat.tex.even.value(0, 0, center)
                color2 = mat.tex.odd.value(0, 0, center)
                texture_color1[i] = [color1.x, color1.y, color1.z]
                texture_color2[i] = [color2.x, color2.y, color2.z]

                # Set albedo to white (will be overridden by texture evaluation)
                material_albedo[i] = [1.0, 1.0, 1.0]
            else:
                # Solid color or other texture - sample once
                texture_type[i] = TEX_SOLID
                try:
                    mat_color = mat.tex.value(0, 0, center)
                    material_albedo[i] = [mat_color.x, mat_color.y, mat_color.z]
                    texture_color1[i] = [mat_color.x, mat_color.y, mat_color.z]
                    texture_color2[i] = [mat_color.x, mat_color.y, mat_color.z]
                except:
                    material_albedo[i] = [0.8, 0.8, 0.8]
                    texture_color1[i] = [0.8, 0.8, 0.8]
                    texture_color2[i] = [0.8, 0.8, 0.8]

            material_fuzz[i] = 0.0
            material_ir[i] = 1.0

        elif mat_type_name == 'metal':
            material_type[i] = MAT_METAL
            material_albedo[i] = [mat.albedo.x, mat.albedo.y, mat.albedo.z]
            material_fuzz[i] = mat.fuzz
            material_ir[i] = 1.0

            # Initialize texture fields (not used for metal)
            texture_type[i] = TEX_SOLID
            texture_scale[i] = 1.0
            texture_color1[i] = [mat.albedo.x, mat.albedo.y, mat.albedo.z]
            texture_color2[i] = [mat.albedo.x, mat.albedo.y, mat.albedo.z]

        elif mat_type_name == 'dielectric':
            material_type[i] = MAT_DIELECTRIC
            material_albedo[i] = [1.0, 1.0, 1.0]  # Dielectric is always white
            material_fuzz[i] = 0.0
            material_ir[i] = mat.ir

            # Initialize texture fields (not used for dielectric)
            texture_type[i] = TEX_SOLID
            texture_scale[i] = 1.0
            texture_color1[i] = [1.0, 1.0, 1.0]
            texture_color2[i] = [1.0, 1.0, 1.0]

        else:
            # Unsupported material - default to lambertian
            material_type[i] = MAT_LAMBERTIAN
            material_albedo[i] = [0.8, 0.8, 0.8]
            material_fuzz[i] = 0.0
            material_ir[i] = 1.0

            # Initialize texture fields
            texture_type[i] = TEX_SOLID
            texture_scale[i] = 1.0
            texture_color1[i] = [0.8, 0.8, 0.8]
            texture_color2[i] = [0.8, 0.8, 0.8]

    return {
        'material_type': material_type,
        'material_albedo': material_albedo,
        'material_fuzz': material_fuzz,
        'material_ir': material_ir,
        'texture_type': texture_type,
        'texture_scale': texture_scale,
        'texture_color1': texture_color1,
        'texture_color2': texture_color2
    }


def compile_scene(world) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], List]:
    """
    Main entry point: compile entire scene.

    Returns:
        geometry_data: dict of numpy arrays
        material_data: dict of numpy arrays
        spheres: list of sphere objects (needed for BVH compiler)
    """
    spheres = extract_spheres(world)
    geometry = compile_geometry(spheres)
    materials = compile_materials(spheres)
    return geometry, materials, spheres
