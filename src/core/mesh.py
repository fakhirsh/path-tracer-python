"""
Mesh loader using PyWavefront for OBJ files.
Automatically finds OBJ files in a model folder and loads geometry, materials, and textures.
"""

from pathlib import Path
from typing import Optional
import pywavefront

from util import point3
from .hittable import hittable, hit_record
from .triangle import triangle
from .material import material
from .aabb import aabb
from .interval import interval
from .bvh_node import bvh_node
from util import ray


class mesh(hittable):
    """
    A mesh loaded from an OBJ file using PyWavefront.
    Automatically discovers OBJ files in the provided model folder.
    """

    def __init__(
        self,
        model_path: str,
        mat: material,
        scale: float = 1.0,
        offset: point3 = point3(0, 0, 0),
        obj_filename: Optional[str] = None,
        use_bvh: bool = True
    ):
        """
        Load a mesh from a model folder.

        Args:
            model_path: Path to the model root folder (e.g., 'house_model/')
            mat: Material to apply to all triangles
            scale: Scale factor for the mesh (default: 1.0)
            offset: Translation offset (default: origin)
            obj_filename: Optional specific OBJ file name. If None, auto-discovers the first .obj file
            use_bvh: Build internal BVH for faster ray-triangle intersection (default: True)
        """
        self.model_path = model_path
        self.mat = mat
        self.scale = scale
        self.offset = offset
        self.triangles = []
        self.scene = None
        self.use_bvh = use_bvh
        self.bvh = None

        obj_file = self._find_obj_file(model_path, obj_filename)
        self._load_with_pywavefront(obj_file)
        self.set_bounding_box()

        # Build internal BVH for this mesh's triangles
        if self.use_bvh and self.triangles:
            self.bvh = bvh_node.from_objects(self.triangles, 0, len(self.triangles))

    def _find_obj_file(self, model_path: str, obj_filename: Optional[str] = None) -> str:
        """
        Find the OBJ file in the model folder structure.

        Args:
            model_path: Root folder containing the model
            obj_filename: Optional specific filename to look for

        Returns:
            Full path to the OBJ file
        """
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        # If specific filename provided, search for it
        if obj_filename:
            # Search recursively for the file
            for obj_file in model_path.rglob(obj_filename):
                if obj_file.suffix.lower() == '.obj':
                    return str(obj_file)
            raise FileNotFoundError(f"OBJ file '{obj_filename}' not found in {model_path}")

        # Otherwise, find the first .obj file
        for obj_file in model_path.rglob('*.obj'):
            return str(obj_file)

        raise FileNotFoundError(f"No OBJ files found in {model_path}")

    def _load_with_pywavefront(self, obj_file: str):
        """
        Load OBJ file using PyWavefront and convert to triangles.

        Args:
            obj_file: Path to the OBJ file
        """
        # Load the scene with PyWavefront
        # parse=True loads everything, collect_faces=True gives us face data
        self.scene = pywavefront.Wavefront(obj_file, collect_faces=True, parse=True, strict=False)

        degenerate_count = 0

        # Process all materials in the scene
        for _, material_obj in self.scene.materials.items():
            # Get vertices - format depends on what's in the OBJ
            # Typical format: [x, y, z, nx, ny, nz, u, v, ...] repeated
            vertices = material_obj.vertices

            if not vertices:
                continue

            # Vertex format size (how many floats per vertex)
            # Check the vertex format to understand the layout
            vertex_format = material_obj.vertex_format

            # Parse vertex format to find stride
            # Common formats: 'T2F_N3F_V3F', 'N3F_V3F', 'V3F', 'T2F_V3F', etc.
            stride = self._calculate_stride(vertex_format)
            v_offset = self._find_position_offset(vertex_format)

            # Extract position data and create triangles from faces
            if hasattr(material_obj, 'faces'):
                # Faces are available - use them directly
                for face in material_obj.faces:
                    if len(face) >= 3:
                        # Get the vertex indices for this face
                        # Triangulate if needed (for quads/n-gons)
                        for i in range(1, len(face) - 1):
                            idx0 = face[0] * stride + v_offset
                            idx1 = face[i] * stride + v_offset
                            idx2 = face[i + 1] * stride + v_offset

                            v0 = self._extract_vertex(vertices, idx0)
                            v1 = self._extract_vertex(vertices, idx1)
                            v2 = self._extract_vertex(vertices, idx2)

                            # Check if triangle is degenerate (zero area)
                            if self._is_degenerate_triangle(v0, v1, v2):
                                degenerate_count += 1
                                continue

                            tri = triangle(v0, v1, v2, self.mat)
                            self.triangles.append(tri)
            else:
                # No faces - vertices are already in triangle order
                num_vertices = len(vertices) // stride

                # Create triangles from sequential vertices
                for i in range(0, num_vertices - 2, 3):
                    idx0 = i * stride + v_offset
                    idx1 = (i + 1) * stride + v_offset
                    idx2 = (i + 2) * stride + v_offset

                    v0 = self._extract_vertex(vertices, idx0)
                    v1 = self._extract_vertex(vertices, idx1)
                    v2 = self._extract_vertex(vertices, idx2)

                    # Check if triangle is degenerate (zero area)
                    if self._is_degenerate_triangle(v0, v1, v2):
                        degenerate_count += 1
                        continue

                    tri = triangle(v0, v1, v2, self.mat)
                    self.triangles.append(tri)

        if degenerate_count > 0:
            print(f"  ⚠ Skipped {degenerate_count} degenerate triangles (zero area)")

        if not self.triangles:
            raise ValueError(f"No valid triangles created from OBJ file")

    def _calculate_stride(self, vertex_format: str) -> int:
        """
        Calculate the stride (number of floats per vertex) from vertex format string.

        Examples: 'T2F_N3F_V3F' = 2 + 3 + 3 = 8 floats per vertex
        """
        stride = 0
        parts = vertex_format.split('_')
        for part in parts:
            if part:
                # Extract number (e.g., '3' from 'V3F')
                num = int(''.join(c for c in part if c.isdigit()))
                stride += num
        return stride

    def _find_position_offset(self, vertex_format: str) -> int:
        """
        Find the offset to position data in the vertex format.
        Position is marked with 'V' (e.g., V3F means 3 floats for position).

        Returns offset in number of floats.
        """
        offset = 0
        parts = vertex_format.split('_')
        for part in parts:
            if part.startswith('V'):
                return offset
            if part:
                num = int(''.join(c for c in part if c.isdigit()))
                offset += num
        return 0

    def _extract_vertex(self, vertices: list, index: int) -> point3:
        """
        Extract a vertex position from the vertices list and apply transformations.

        Args:
            vertices: Flat list of vertex data
            index: Starting index for the position (x, y, z)

        Returns:
            Transformed point3
        """
        x = vertices[index]
        y = vertices[index + 1]
        z = vertices[index + 2]

        # Apply scale and offset
        return point3(x * self.scale, y * self.scale, z * self.scale) + self.offset

    def _is_degenerate_triangle(self, v0: point3, v1: point3, v2: point3) -> bool:
        """
        Check if a triangle is degenerate (has zero area).

        Args:
            v0, v1, v2: Triangle vertices

        Returns:
            True if triangle is degenerate, False otherwise
        """
        from util import vec3

        # Compute edges
        edge1 = v1 - v0
        edge2 = v2 - v0

        # Compute cross product (normal)
        cross = vec3.cross(edge1, edge2)

        # Triangle is degenerate if cross product is near zero
        # (vertices are collinear or coincident)
        epsilon = 1e-10
        return cross.length_squared() < epsilon

    def set_bounding_box(self):
        """Compute bounding box from all triangles."""
        if not self.triangles:
            self.bbox = aabb.from_intervals(interval.empty, interval.empty, interval.empty)
            return

        # Start with first triangle's bbox
        self.bbox = self.triangles[0].bounding_box()

        # Expand to include all other triangles
        for tri in self.triangles[1:]:
            self.bbox = aabb.from_aabbs(self.bbox, tri.bounding_box())

    def bounding_box(self) -> aabb:
        """Return the bounding box of the entire mesh."""
        return self.bbox

    def hit(self, r: ray, ray_t: interval, rec: hit_record) -> bool:
        """
        Test ray intersection with all triangles in the mesh.
        Uses internal BVH if available for faster intersection testing.
        Returns the closest hit.
        """
        # Use BVH if available (much faster for large meshes)
        if self.bvh:
            return self.bvh.hit(r, ray_t, rec)

        # Fallback: linear search through all triangles
        temp_rec = hit_record()
        hit_anything = False
        closest_so_far = ray_t.max

        for tri in self.triangles:
            if tri.hit(r, interval.from_floats(ray_t.min, closest_so_far), temp_rec):
                hit_anything = True
                closest_so_far = temp_rec.t
                rec.copy_from(temp_rec)

        return hit_anything

    def triangle_count(self) -> int:
        """Return the number of triangles in the mesh."""
        return len(self.triangles)

    def __repr__(self):
        return f"mesh('{self.model_path}', triangles={len(self.triangles)})"
