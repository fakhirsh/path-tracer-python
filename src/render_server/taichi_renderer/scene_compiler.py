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
MAT_EMISSIVE = 3
MAT_ISOTROPIC = 4

# Texture type constants
TEX_SOLID = 0
TEX_CHECKER = 1
TEX_IMAGE = 2
TEX_NOISE = 3


def extract_spheres(world) -> List:
    """
    Recursively extract all Sphere objects from world hierarchy.
    Handles: Sphere, hittable_list, bvh_node
    Returns: List of unique Sphere objects (deduplicated by object identity)
    """
    from core.sphere import Sphere
    from core.hittable_list import hittable_list
    from core.bvh_node import bvh_node

    seen = set()
    spheres = []

    def _extract(obj):
        # Check if this is a constant_medium - extract its boundary
        from core.constant_medium import constant_medium
        if isinstance(obj, constant_medium):
            _extract(obj.boundary)
        # Check if this is a Sphere
        elif isinstance(obj, Sphere):
            if id(obj) not in seen:
                seen.add(id(obj))
                spheres.append(obj)
        # Check if this is a hittable_list
        elif isinstance(obj, hittable_list):
            for item in obj.objects:
                _extract(item)
        # Check if this is a bvh_node
        elif isinstance(obj, bvh_node):
            if obj.left is not None:
                _extract(obj.left)
            if obj.right is not None:
                _extract(obj.right)

    _extract(world)
    return spheres


def extract_quads(world) -> List:
    """
    Recursively extract all quad objects from world hierarchy.
    Handles: quad, hittable_list, bvh_node
    Returns: List of unique quad objects (deduplicated by object identity)
    """
    from core.quad import quad
    from core.hittable_list import hittable_list
    from core.bvh_node import bvh_node

    seen = set()
    quads = []

    def _extract(obj):
        # Check if this is a constant_medium - extract its boundary
        from core.constant_medium import constant_medium
        if isinstance(obj, constant_medium):
            _extract(obj.boundary)
        # Check if this is a quad
        elif isinstance(obj, quad):
            if id(obj) not in seen:
                seen.add(id(obj))
                quads.append(obj)
        # Check if this is a hittable_list
        elif isinstance(obj, hittable_list):
            for item in obj.objects:
                _extract(item)
        # Check if this is a bvh_node
        elif isinstance(obj, bvh_node):
            if obj.left is not None:
                _extract(obj.left)
            if obj.right is not None:
                _extract(obj.right)

    _extract(world)
    return quads


def extract_triangles(world) -> List:
    """
    Recursively extract all triangle objects from world hierarchy.
    Handles: triangle, mesh (extracts internal triangles), hittable_list, bvh_node
    Returns: List of unique triangle objects (deduplicated by object identity)
    """
    from core.triangle import triangle
    from core.hittable_list import hittable_list
    from core.bvh_node import bvh_node
    from core.mesh import mesh

    seen = set()
    triangles = []

    def _extract(obj):
        # Check if this is a constant_medium - extract its boundary
        from core.constant_medium import constant_medium
        if isinstance(obj, constant_medium):
            _extract(obj.boundary)
        # Check if this is a mesh - extract all its internal triangles
        elif isinstance(obj, mesh):
            for tri in obj.triangles:
                if id(tri) not in seen:
                    seen.add(id(tri))
                    triangles.append(tri)
        # Check if this is a triangle
        elif isinstance(obj, triangle):
            if id(obj) not in seen:
                seen.add(id(obj))
                triangles.append(obj)
        # Check if this is a hittable_list
        elif isinstance(obj, hittable_list):
            for item in obj.objects:
                _extract(item)
        # Check if this is a bvh_node
        elif isinstance(obj, bvh_node):
            if obj.left is not None:
                _extract(obj.left)
            if obj.right is not None:
                _extract(obj.right)

    _extract(world)
    return triangles


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


def compile_quad_geometry(quads: List) -> Dict[str, np.ndarray]:
    """
    Pack quad geometry into numpy arrays for GPU upload.

    Returns dict with:
        'quad_Q': np.ndarray of shape (N, 3) dtype=float32  # Corner point
        'quad_u': np.ndarray of shape (N, 3) dtype=float32  # First edge vector
        'quad_v': np.ndarray of shape (N, 3) dtype=float32  # Second edge vector
        'quad_normal': np.ndarray of shape (N, 3) dtype=float32
        'quad_D': np.ndarray of shape (N,) dtype=float32
        'quad_w': np.ndarray of shape (N, 3) dtype=float32
        'num_quads': int
    """
    n = len(quads)
    quad_Q = np.zeros((n, 3), dtype=np.float32)
    quad_u = np.zeros((n, 3), dtype=np.float32)
    quad_v = np.zeros((n, 3), dtype=np.float32)
    quad_normal = np.zeros((n, 3), dtype=np.float32)
    quad_D = np.zeros(n, dtype=np.float32)
    quad_w = np.zeros((n, 3), dtype=np.float32)

    for i, q in enumerate(quads):
        quad_Q[i] = [q.Q.x, q.Q.y, q.Q.z]
        quad_u[i] = [q.u.x, q.u.y, q.u.z]
        quad_v[i] = [q.v.x, q.v.y, q.v.z]
        quad_normal[i] = [q.normal.x, q.normal.y, q.normal.z]
        quad_D[i] = q.D
        quad_w[i] = [q.w.x, q.w.y, q.w.z]

    return {
        'quad_Q': quad_Q,
        'quad_u': quad_u,
        'quad_v': quad_v,
        'quad_normal': quad_normal,
        'quad_D': quad_D,
        'quad_w': quad_w,
        'num_quads': n
    }


def compile_triangle_geometry(triangles: List) -> Dict[str, np.ndarray]:
    """
    Pack triangle geometry into numpy arrays for GPU upload.

    Returns dict with:
        'triangle_v0': np.ndarray of shape (N, 3) dtype=float32  # First vertex
        'triangle_v1': np.ndarray of shape (N, 3) dtype=float32  # Second vertex
        'triangle_v2': np.ndarray of shape (N, 3) dtype=float32  # Third vertex
        'triangle_edge1': np.ndarray of shape (N, 3) dtype=float32  # Edge v1 - v0
        'triangle_edge2': np.ndarray of shape (N, 3) dtype=float32  # Edge v2 - v0
        'triangle_normal': np.ndarray of shape (N, 3) dtype=float32
        'num_triangles': int
    """
    n = len(triangles)
    triangle_v0 = np.zeros((n, 3), dtype=np.float32)
    triangle_v1 = np.zeros((n, 3), dtype=np.float32)
    triangle_v2 = np.zeros((n, 3), dtype=np.float32)
    triangle_edge1 = np.zeros((n, 3), dtype=np.float32)
    triangle_edge2 = np.zeros((n, 3), dtype=np.float32)
    triangle_normal = np.zeros((n, 3), dtype=np.float32)

    for i, tri in enumerate(triangles):
        triangle_v0[i] = [tri.v0.x, tri.v0.y, tri.v0.z]
        triangle_v1[i] = [tri.v1.x, tri.v1.y, tri.v1.z]
        triangle_v2[i] = [tri.v2.x, tri.v2.y, tri.v2.z]
        triangle_edge1[i] = [tri.edge1.x, tri.edge1.y, tri.edge1.z]
        triangle_edge2[i] = [tri.edge2.x, tri.edge2.y, tri.edge2.z]
        triangle_normal[i] = [tri.normal.x, tri.normal.y, tri.normal.z]

    return {
        'triangle_v0': triangle_v0,
        'triangle_v1': triangle_v1,
        'triangle_v2': triangle_v2,
        'triangle_edge1': triangle_edge1,
        'triangle_edge2': triangle_edge2,
        'triangle_normal': triangle_normal,
        'num_triangles': n
    }


def compile_materials(spheres: List, image_texture_registry: Dict = None, medium_registry: Dict = None) -> Dict[str, np.ndarray]:
    """
    Extract material properties from spheres.

    Args:
        spheres: List of sphere objects
        image_texture_registry: Dict mapping image_texture objects to their GPU indices
        medium_registry: Dict mapping primitive id() to constant medium properties

    Returns dict with:
        'material_type': np.ndarray of shape (N,) dtype=int32
        'material_albedo': np.ndarray of shape (N, 3) dtype=float32
        'material_fuzz': np.ndarray of shape (N,) dtype=float32
        'material_ir': np.ndarray of shape (N,) dtype=float32
        'material_emit_color': np.ndarray of shape (N, 3) dtype=float32
        'texture_type': np.ndarray of shape (N,) dtype=int32
        'texture_scale': np.ndarray of shape (N,) dtype=float32
        'texture_color1': np.ndarray of shape (N, 3) dtype=float32
        'texture_color2': np.ndarray of shape (N, 3) dtype=float32
        'texture_image_idx': np.ndarray of shape (N,) dtype=int32
        'is_constant_medium': np.ndarray of shape (N,) dtype=int32
        'medium_density': np.ndarray of shape (N,) dtype=float32
        'medium_albedo': np.ndarray of shape (N, 3) dtype=float32
    """
    n = len(spheres)
    if image_texture_registry is None:
        image_texture_registry = {}
    if medium_registry is None:
        medium_registry = {}

    material_type = np.zeros(n, dtype=np.int32)
    material_albedo = np.zeros((n, 3), dtype=np.float32)
    material_fuzz = np.zeros(n, dtype=np.float32)
    material_ir = np.zeros(n, dtype=np.float32)
    material_emit_color = np.zeros((n, 3), dtype=np.float32)

    texture_type = np.zeros(n, dtype=np.int32)
    texture_scale = np.ones(n, dtype=np.float32)
    texture_color1 = np.zeros((n, 3), dtype=np.float32)
    texture_color2 = np.zeros((n, 3), dtype=np.float32)
    texture_image_idx = np.full(n, -1, dtype=np.int32)  # -1 means no image texture

    # Constant medium data
    is_constant_medium = np.zeros(n, dtype=np.int32)
    medium_density = np.zeros(n, dtype=np.float32)
    medium_albedo = np.zeros((n, 3), dtype=np.float32)

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
            elif tex_type_name == 'image_texture':
                # Image texture - store reference to image
                texture_type[i] = TEX_IMAGE
                texture_image_idx[i] = image_texture_registry.get(id(mat.tex), -1)

                # Set albedo to white (will be overridden by texture evaluation)
                material_albedo[i] = [1.0, 1.0, 1.0]
                texture_color1[i] = [1.0, 1.0, 1.0]
                texture_color2[i] = [1.0, 1.0, 1.0]
            elif tex_type_name == 'noise_texture':
                # Perlin noise texture - store scale
                texture_type[i] = TEX_NOISE
                texture_scale[i] = mat.tex.scale

                # Base color (used in marble effect)
                texture_color1[i] = [0.5, 0.5, 0.5]
                texture_color2[i] = [0.5, 0.5, 0.5]

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

        elif mat_type_name == 'diffuse_light':
            material_type[i] = MAT_EMISSIVE

            # Extract emission color from texture
            try:
                emit_color = mat.tex.value(0, 0, ref_point if 'ref_point' in locals() else center)
                material_emit_color[i] = [emit_color.x, emit_color.y, emit_color.z]
            except:
                material_emit_color[i] = [1.0, 1.0, 1.0]

            # Emissive materials don't scatter
            material_albedo[i] = [0.0, 0.0, 0.0]
            material_fuzz[i] = 0.0
            material_ir[i] = 1.0

            # Initialize texture fields (not used for emissive)
            texture_type[i] = TEX_SOLID
            texture_scale[i] = 1.0
            texture_color1[i] = [0.0, 0.0, 0.0]
            texture_color2[i] = [0.0, 0.0, 0.0]

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

        # Check if this sphere is a constant medium boundary
        if id(sphere) in medium_registry:
            is_constant_medium[i] = 1
            medium_density[i] = medium_registry[id(sphere)]['density']
            medium_albedo[i] = medium_registry[id(sphere)]['albedo']

    return {
        'material_type': material_type,
        'material_albedo': material_albedo,
        'material_fuzz': material_fuzz,
        'material_ir': material_ir,
        'material_emit_color': material_emit_color,
        'texture_type': texture_type,
        'texture_scale': texture_scale,
        'texture_color1': texture_color1,
        'texture_color2': texture_color2,
        'texture_image_idx': texture_image_idx,
        'is_constant_medium': is_constant_medium,
        'medium_density': medium_density,
        'medium_albedo': medium_albedo
    }


def compile_quad_materials(quads: List, image_texture_registry: Dict = None, medium_registry: Dict = None) -> Dict[str, np.ndarray]:
    """
    Extract material properties from quads.
    Similar to compile_materials but for quads.

    Returns dict with:
        'material_type': np.ndarray of shape (N,) dtype=int32
        'material_albedo': np.ndarray of shape (N, 3) dtype=float32
        'material_fuzz': np.ndarray of shape (N,) dtype=float32
        'material_ir': np.ndarray of shape (N,) dtype=float32
        'material_emit_color': np.ndarray of shape (N, 3) dtype=float32
        'texture_type': np.ndarray of shape (N,) dtype=int32
        'texture_scale': np.ndarray of shape (N,) dtype=float32
        'texture_color1': np.ndarray of shape (N, 3) dtype=float32
        'texture_color2': np.ndarray of shape (N, 3) dtype=float32
        'texture_image_idx': np.ndarray of shape (N,) dtype=int32
        'is_constant_medium': np.ndarray of shape (N,) dtype=int32
        'medium_density': np.ndarray of shape (N,) dtype=float32
        'medium_albedo': np.ndarray of shape (N, 3) dtype=float32
    """
    n = len(quads)
    if image_texture_registry is None:
        image_texture_registry = {}
    if medium_registry is None:
        medium_registry = {}

    material_type = np.zeros(n, dtype=np.int32)
    material_albedo = np.zeros((n, 3), dtype=np.float32)
    material_fuzz = np.zeros(n, dtype=np.float32)
    material_ir = np.zeros(n, dtype=np.float32)
    material_emit_color = np.zeros((n, 3), dtype=np.float32)

    texture_type = np.zeros(n, dtype=np.int32)
    texture_scale = np.ones(n, dtype=np.float32)
    texture_color1 = np.zeros((n, 3), dtype=np.float32)
    texture_color2 = np.zeros((n, 3), dtype=np.float32)
    texture_image_idx = np.full(n, -1, dtype=np.int32)

    # Constant medium data
    is_constant_medium = np.zeros(n, dtype=np.int32)
    medium_density = np.zeros(n, dtype=np.float32)
    medium_albedo = np.zeros((n, 3), dtype=np.float32)

    for i, q in enumerate(quads):
        mat = q.mat
        mat_type_name = type(mat).__name__
        # Use quad corner as reference point for texture sampling
        ref_point = q.Q

        if mat_type_name == 'lambertian':
            material_type[i] = MAT_LAMBERTIAN

            # Extract texture information
            tex_type_name = type(mat.tex).__name__

            if tex_type_name == 'checker_texture':
                # Checker texture - store scale and two colors
                texture_type[i] = TEX_CHECKER
                texture_scale[i] = 1.0 / mat.tex.inv_scale  # Convert back to scale

                # Get the two colors from even and odd textures
                color1 = mat.tex.even.value(0, 0, ref_point)
                color2 = mat.tex.odd.value(0, 0, ref_point)
                texture_color1[i] = [color1.x, color1.y, color1.z]
                texture_color2[i] = [color2.x, color2.y, color2.z]

                # Set albedo to white (will be overridden by texture evaluation)
                material_albedo[i] = [1.0, 1.0, 1.0]
            elif tex_type_name == 'image_texture':
                # Image texture - store reference to image
                texture_type[i] = TEX_IMAGE
                texture_image_idx[i] = image_texture_registry.get(id(mat.tex), -1)

                # Set albedo to white (will be overridden by texture evaluation)
                material_albedo[i] = [1.0, 1.0, 1.0]
                texture_color1[i] = [1.0, 1.0, 1.0]
                texture_color2[i] = [1.0, 1.0, 1.0]
            elif tex_type_name == 'noise_texture':
                # Perlin noise texture - store scale
                texture_type[i] = TEX_NOISE
                texture_scale[i] = mat.tex.scale

                # Base color (used in marble effect)
                texture_color1[i] = [0.5, 0.5, 0.5]
                texture_color2[i] = [0.5, 0.5, 0.5]

                # Set albedo to white (will be overridden by texture evaluation)
                material_albedo[i] = [1.0, 1.0, 1.0]
            else:
                # Solid color or other texture - sample once
                texture_type[i] = TEX_SOLID
                try:
                    mat_color = mat.tex.value(0, 0, ref_point)
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

        elif mat_type_name == 'diffuse_light':
            material_type[i] = MAT_EMISSIVE

            # Extract emission color from texture
            try:
                emit_color = mat.tex.value(0, 0, ref_point if 'ref_point' in locals() else center)
                material_emit_color[i] = [emit_color.x, emit_color.y, emit_color.z]
            except:
                material_emit_color[i] = [1.0, 1.0, 1.0]

            # Emissive materials don't scatter
            material_albedo[i] = [0.0, 0.0, 0.0]
            material_fuzz[i] = 0.0
            material_ir[i] = 1.0

            # Initialize texture fields (not used for emissive)
            texture_type[i] = TEX_SOLID
            texture_scale[i] = 1.0
            texture_color1[i] = [0.0, 0.0, 0.0]
            texture_color2[i] = [0.0, 0.0, 0.0]

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

        # Check if this quad is a constant medium boundary
        if id(q) in medium_registry:
            is_constant_medium[i] = 1
            medium_density[i] = medium_registry[id(q)]['density']
            medium_albedo[i] = medium_registry[id(q)]['albedo']

    return {
        'material_type': material_type,
        'material_albedo': material_albedo,
        'material_fuzz': material_fuzz,
        'material_ir': material_ir,
        'material_emit_color': material_emit_color,
        'texture_type': texture_type,
        'texture_scale': texture_scale,
        'texture_color1': texture_color1,
        'texture_color2': texture_color2,
        'texture_image_idx': texture_image_idx,
        'is_constant_medium': is_constant_medium,
        'medium_density': medium_density,
        'medium_albedo': medium_albedo
    }


def compile_triangle_materials(triangles: List, image_texture_registry: Dict = None, medium_registry: Dict = None) -> Dict[str, np.ndarray]:
    """
    Extract material properties from triangles.
    Similar to compile_materials but for triangles.

    Returns dict with:
        'material_type': np.ndarray of shape (N,) dtype=int32
        'material_albedo': np.ndarray of shape (N, 3) dtype=float32
        'material_fuzz': np.ndarray of shape (N,) dtype=float32
        'material_ir': np.ndarray of shape (N,) dtype=float32
        'material_emit_color': np.ndarray of shape (N, 3) dtype=float32
        'texture_type': np.ndarray of shape (N,) dtype=int32
        'texture_scale': np.ndarray of shape (N,) dtype=float32
        'texture_color1': np.ndarray of shape (N, 3) dtype=float32
        'texture_color2': np.ndarray of shape (N, 3) dtype=float32
        'texture_image_idx': np.ndarray of shape (N,) dtype=int32
        'is_constant_medium': np.ndarray of shape (N,) dtype=int32
        'medium_density': np.ndarray of shape (N,) dtype=float32
        'medium_albedo': np.ndarray of shape (N, 3) dtype=float32
    """
    n = len(triangles)
    if image_texture_registry is None:
        image_texture_registry = {}
    if medium_registry is None:
        medium_registry = {}

    material_type = np.zeros(n, dtype=np.int32)
    material_albedo = np.zeros((n, 3), dtype=np.float32)
    material_fuzz = np.zeros(n, dtype=np.float32)
    material_ir = np.zeros(n, dtype=np.float32)
    material_emit_color = np.zeros((n, 3), dtype=np.float32)

    texture_type = np.zeros(n, dtype=np.int32)
    texture_scale = np.ones(n, dtype=np.float32)
    texture_color1 = np.zeros((n, 3), dtype=np.float32)
    texture_color2 = np.zeros((n, 3), dtype=np.float32)
    texture_image_idx = np.full(n, -1, dtype=np.int32)

    # Constant medium data
    is_constant_medium = np.zeros(n, dtype=np.int32)
    medium_density = np.zeros(n, dtype=np.float32)
    medium_albedo = np.zeros((n, 3), dtype=np.float32)

    for i, tri in enumerate(triangles):
        mat = tri.mat
        mat_type_name = type(mat).__name__
        # Use first vertex as reference point for texture sampling
        ref_point = tri.v0

        if mat_type_name == 'lambertian':
            material_type[i] = MAT_LAMBERTIAN

            # Extract texture information
            tex_type_name = type(mat.tex).__name__

            if tex_type_name == 'checker_texture':
                # Checker texture - store scale and two colors
                texture_type[i] = TEX_CHECKER
                texture_scale[i] = 1.0 / mat.tex.inv_scale  # Convert back to scale

                # Get the two colors from even and odd textures
                color1 = mat.tex.even.value(0, 0, ref_point)
                color2 = mat.tex.odd.value(0, 0, ref_point)
                texture_color1[i] = [color1.x, color1.y, color1.z]
                texture_color2[i] = [color2.x, color2.y, color2.z]

                # Set albedo to white (will be overridden by texture evaluation)
                material_albedo[i] = [1.0, 1.0, 1.0]
            elif tex_type_name == 'image_texture':
                # Image texture - store reference to image
                texture_type[i] = TEX_IMAGE
                texture_image_idx[i] = image_texture_registry.get(id(mat.tex), -1)

                # Set albedo to white (will be overridden by texture evaluation)
                material_albedo[i] = [1.0, 1.0, 1.0]
                texture_color1[i] = [1.0, 1.0, 1.0]
                texture_color2[i] = [1.0, 1.0, 1.0]
            elif tex_type_name == 'noise_texture':
                # Perlin noise texture - store scale
                texture_type[i] = TEX_NOISE
                texture_scale[i] = mat.tex.scale

                # Base color (used in marble effect)
                texture_color1[i] = [0.5, 0.5, 0.5]
                texture_color2[i] = [0.5, 0.5, 0.5]

                # Set albedo to white (will be overridden by texture evaluation)
                material_albedo[i] = [1.0, 1.0, 1.0]
            else:
                # Solid color or other texture - sample once
                texture_type[i] = TEX_SOLID
                try:
                    mat_color = mat.tex.value(0, 0, ref_point)
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

        elif mat_type_name == 'diffuse_light':
            material_type[i] = MAT_EMISSIVE

            # Extract emission color from texture
            try:
                emit_color = mat.tex.value(0, 0, ref_point if 'ref_point' in locals() else center)
                material_emit_color[i] = [emit_color.x, emit_color.y, emit_color.z]
            except:
                material_emit_color[i] = [1.0, 1.0, 1.0]

            # Emissive materials don't scatter
            material_albedo[i] = [0.0, 0.0, 0.0]
            material_fuzz[i] = 0.0
            material_ir[i] = 1.0

            # Initialize texture fields (not used for emissive)
            texture_type[i] = TEX_SOLID
            texture_scale[i] = 1.0
            texture_color1[i] = [0.0, 0.0, 0.0]
            texture_color2[i] = [0.0, 0.0, 0.0]

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

        # Check if this triangle is a constant medium boundary
        if id(tri) in medium_registry:
            is_constant_medium[i] = 1
            medium_density[i] = medium_registry[id(tri)]['density']
            medium_albedo[i] = medium_registry[id(tri)]['albedo']

    return {
        'material_type': material_type,
        'material_albedo': material_albedo,
        'material_fuzz': material_fuzz,
        'material_ir': material_ir,
        'material_emit_color': material_emit_color,
        'texture_type': texture_type,
        'texture_scale': texture_scale,
        'texture_color1': texture_color1,
        'texture_color2': texture_color2,
        'texture_image_idx': texture_image_idx,
        'is_constant_medium': is_constant_medium,
        'medium_density': medium_density,
        'medium_albedo': medium_albedo
    }


def build_image_texture_registry(world) -> Tuple[Dict[int, int], List]:
    """
    Build a registry of all unique image textures used in the scene.

    Returns:
        image_texture_registry: Dict mapping texture object id() to GPU index
        image_texture_list: List of image_texture objects in order
    """
    from core.texture import image_texture
    from core.hittable_list import hittable_list
    from core.bvh_node import bvh_node
    from core.sphere import Sphere
    from core.quad import quad
    from core.triangle import triangle

    image_textures = {}  # id -> texture object

    def extract_textures(obj):
        """Recursively extract image textures from objects"""
        if isinstance(obj, (Sphere, quad, triangle)):
            if hasattr(obj, 'material') or hasattr(obj, 'mat'):
                mat = obj.material if hasattr(obj, 'material') else obj.mat
                if hasattr(mat, 'tex') and isinstance(mat.tex, image_texture):
                    image_textures[id(mat.tex)] = mat.tex
        elif isinstance(obj, hittable_list):
            for item in obj.objects:
                extract_textures(item)
        elif isinstance(obj, bvh_node):
            if obj.left is not None:
                extract_textures(obj.left)
            if obj.right is not None:
                extract_textures(obj.right)

    extract_textures(world)

    # Build ordered list and registry
    image_texture_list = list(image_textures.values())
    image_texture_registry = {id(tex): i for i, tex in enumerate(image_texture_list)}

    return image_texture_registry, image_texture_list


def build_constant_medium_registry(world) -> Dict[int, Dict]:
    """
    Build a registry of all constant_medium objects and their properties.

    Returns:
        constant_medium_registry: Dict mapping boundary primitive id() to medium properties
            {
                boundary_prim_id: {
                    'density': float,
                    'albedo': (r, g, b) tuple
                }
            }
    """
    from core.constant_medium import constant_medium
    from core.hittable_list import hittable_list
    from core.bvh_node import bvh_node
    from core.sphere import Sphere
    from core.quad import quad
    from core.triangle import triangle

    medium_registry = {}  # boundary primitive id -> medium properties

    def extract_boundary_primitives(obj):
        """Recursively extract boundary primitives from a hittable object"""
        primitives = []
        if isinstance(obj, (Sphere, quad, triangle)):
            primitives.append(obj)
        elif isinstance(obj, hittable_list):
            for item in obj.objects:
                primitives.extend(extract_boundary_primitives(item))
        elif isinstance(obj, bvh_node):
            if obj.left is not None:
                primitives.extend(extract_boundary_primitives(obj.left))
            if obj.right is not None:
                primitives.extend(extract_boundary_primitives(obj.right))
        return primitives

    def extract_constant_mediums(obj):
        """Recursively find all constant_medium objects"""
        if isinstance(obj, constant_medium):
            # Extract density
            density = -1.0 / obj.neg_inv_density if obj.neg_inv_density != 0 else 0.01

            # Extract albedo from phase function (isotropic material)
            albedo = (1.0, 1.0, 1.0)
            if hasattr(obj.phase_function, 'tex'):
                try:
                    # Sample the texture to get albedo color
                    from util import point3
                    sample_color = obj.phase_function.tex.value(0, 0, point3(0, 0, 0))
                    albedo = (sample_color.x, sample_color.y, sample_color.z)
                except:
                    pass

            # Get all boundary primitives
            boundary_prims = extract_boundary_primitives(obj.boundary)

            # Register each boundary primitive
            for prim in boundary_prims:
                medium_registry[id(prim)] = {
                    'density': density,
                    'albedo': albedo
                }

        elif isinstance(obj, hittable_list):
            for item in obj.objects:
                extract_constant_mediums(item)
        elif isinstance(obj, bvh_node):
            if obj.left is not None:
                extract_constant_mediums(obj.left)
            if obj.right is not None:
                extract_constant_mediums(obj.right)

    extract_constant_mediums(world)
    return medium_registry


def compile_scene(world) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], List, Dict[str, np.ndarray], List, Dict[str, np.ndarray], List, Dict[int, int], List]:
    """
    Main entry point: compile entire scene.

    Returns:
        geometry_data: dict of numpy arrays (spheres)
        material_data: dict of numpy arrays (sphere materials)
        spheres: list of sphere objects (needed for BVH compiler)
        quad_geometry_data: dict of numpy arrays (quads)
        quad_materials: dict of numpy arrays (quad materials)
        quads: list of quad objects (needed for BVH compiler)
        triangle_geometry_data: dict of numpy arrays (triangles)
        triangle_materials: dict of numpy arrays (triangle materials)
        triangles: list of triangle objects (needed for BVH compiler)
    """
    # Build image texture registry first
    image_texture_registry, image_texture_list = build_image_texture_registry(world)

    # Build constant medium registry
    medium_registry = build_constant_medium_registry(world)

    spheres = extract_spheres(world)
    quads = extract_quads(world)
    triangles = extract_triangles(world)

    geometry = compile_geometry(spheres)
    materials = compile_materials(spheres, image_texture_registry, medium_registry)

    quad_geometry = compile_quad_geometry(quads)
    quad_materials = compile_quad_materials(quads, image_texture_registry, medium_registry)

    triangle_geometry = compile_triangle_geometry(triangles)
    triangle_materials = compile_triangle_materials(triangles, image_texture_registry, medium_registry)

    return geometry, materials, spheres, quad_geometry, quad_materials, quads, triangle_geometry, triangle_materials, triangles, image_texture_registry, image_texture_list
