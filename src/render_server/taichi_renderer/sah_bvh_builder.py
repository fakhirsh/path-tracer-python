"""
Surface Area Heuristic (SAH) BVH Builder

Builds a high-quality BVH using the Surface Area Heuristic for optimal ray tracing performance.
SAH minimizes the expected cost of ray-scene intersection by choosing optimal split planes.

Performance improvement over median split: 2-3x faster traversal on complex scenes.
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass

# Primitive type constants
PRIM_SPHERE = 0
PRIM_TRIANGLE = 1
PRIM_QUAD = 2


@dataclass
class AABB:
    """Axis-Aligned Bounding Box"""
    min: np.ndarray  # [x, y, z]
    max: np.ndarray  # [x, y, z]

    def surface_area(self) -> float:
        """Compute surface area (used in SAH cost function)"""
        extent = self.max - self.min
        return 2.0 * (extent[0] * extent[1] + extent[1] * extent[2] + extent[2] * extent[0])

    def pad_to_minimums(self):
        """Pad thin dimensions to avoid numerical issues (same as core/aabb.py)"""
        delta = 0.0001
        for axis in range(3):
            if self.max[axis] - self.min[axis] < delta:
                # Expand equally on both sides
                mid = (self.min[axis] + self.max[axis]) / 2.0
                self.min[axis] = mid - delta / 2.0
                self.max[axis] = mid + delta / 2.0

    def union(self, other: 'AABB') -> 'AABB':
        """Compute bounding box that contains both AABBs"""
        return AABB(
            min=np.minimum(self.min, other.min),
            max=np.maximum(self.max, other.max)
        )

    @staticmethod
    def empty() -> 'AABB':
        """Create empty AABB"""
        return AABB(
            min=np.array([np.inf, np.inf, np.inf], dtype=np.float32),
            max=np.array([-np.inf, -np.inf, -np.inf], dtype=np.float32)
        )


@dataclass
class Primitive:
    """Primitive reference for BVH construction"""
    prim_type: int      # PRIM_SPHERE, etc.
    prim_idx: int       # Index into geometry array
    bbox: AABB          # Bounding box
    centroid: np.ndarray  # Center point [x, y, z]


class SAHBVHNode:
    """BVH node for CPU-side construction"""
    def __init__(self):
        self.bbox = AABB.empty()
        self.left = None        # Left child node
        self.right = None       # Right child node
        self.parent = None      # Parent node (for stackless traversal)
        self.prim_type = -1     # Primitive type (leaf only)
        self.prim_idx = -1      # Primitive index (leaf only)
        self.node_idx = -1      # Final flattened index

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class SAHBVHBuilder:
    """
    Build BVH using Surface Area Heuristic.

    SAH Cost Function:
        cost = traverse_cost +
               P(left) * intersect_cost * num_left +
               P(right) * intersect_cost * num_right

    where P(child) = surface_area(child) / surface_area(parent)
    """

    def __init__(self, traverse_cost: float = 1.0, intersect_cost: float = 1.5):
        """
        Args:
            traverse_cost: Cost of traversing one BVH node
            intersect_cost: Cost of testing one primitive intersection
        """
        self.traverse_cost = traverse_cost
        self.intersect_cost = intersect_cost
        self.primitives: List[Primitive] = []
        self.root = None

    def add_sphere(self, sphere_idx: int, center: np.ndarray, radius: float):
        """Add a sphere primitive to the builder"""
        bbox = AABB(
            min=center - radius,
            max=center + radius
        )
        prim = Primitive(
            prim_type=PRIM_SPHERE,
            prim_idx=sphere_idx,
            bbox=bbox,
            centroid=center.copy()
        )
        self.primitives.append(prim)

    def add_quad(self, quad_idx: int, Q: np.ndarray, u: np.ndarray, v: np.ndarray):
        """Add a quad primitive to the builder"""
        # Compute bounding box from the four corners of the quad
        corners = [
            Q,
            Q + u,
            Q + v,
            Q + u + v
        ]
        min_corner = np.min(corners, axis=0)
        max_corner = np.max(corners, axis=0)

        bbox = AABB(min=min_corner, max=max_corner)
        # Pad thin dimensions to avoid numerical issues (quads are often planar)
        bbox.pad_to_minimums()

        # Centroid is the center of the quad
        centroid = Q + 0.5 * u + 0.5 * v

        prim = Primitive(
            prim_type=PRIM_QUAD,
            prim_idx=quad_idx,
            bbox=bbox,
            centroid=centroid.copy()
        )
        self.primitives.append(prim)

    def add_triangle(self, tri_idx: int, v0: np.ndarray, v1: np.ndarray, v2: np.ndarray):
        """Add a triangle primitive to the builder"""
        # Compute bounding box from the three vertices
        vertices = [v0, v1, v2]
        min_corner = np.min(vertices, axis=0)
        max_corner = np.max(vertices, axis=0)

        bbox = AABB(min=min_corner, max=max_corner)
        # Pad thin dimensions to avoid numerical issues (triangles are planar)
        bbox.pad_to_minimums()

        # Centroid is the center of the triangle
        centroid = (v0 + v1 + v2) / 3.0

        prim = Primitive(
            prim_type=PRIM_TRIANGLE,
            prim_idx=tri_idx,
            bbox=bbox,
            centroid=centroid.copy()
        )
        self.primitives.append(prim)

    def build(self) -> SAHBVHNode:
        """
        Build BVH tree using SAH.
        Returns root node.
        """
        if not self.primitives:
            return None

        # Build tree recursively
        self.root = self._build_recursive(self.primitives, depth=0)
        return self.root

    def _build_recursive(self, prims: List[Primitive], depth: int) -> SAHBVHNode:
        """
        Recursively build BVH subtree.

        Algorithm:
        1. Compute bounding box for all primitives
        2. If primitives <= 1, create leaf node
        3. Otherwise, find best split using SAH
        4. Partition primitives and recurse
        """
        node = SAHBVHNode()

        # Compute bounding box for all primitives
        node.bbox = AABB.empty()
        for prim in prims:
            node.bbox = node.bbox.union(prim.bbox)

        # Base case: create leaf node
        if len(prims) == 1:
            node.prim_type = prims[0].prim_type
            node.prim_idx = prims[0].prim_idx
            return node

        # Find best split using SAH
        best_axis, best_pos, best_cost = self._find_best_split(prims, node.bbox)

        # Check if split is worth it (compare to leaf cost)
        leaf_cost = self.intersect_cost * len(prims)
        if best_cost >= leaf_cost and len(prims) <= 4:
            # Too expensive to split - create leaf with multiple primitives
            # For now, just pick first primitive (TODO: handle multiple prims per leaf)
            node.prim_type = prims[0].prim_type
            node.prim_idx = prims[0].prim_idx
            return node

        # Partition primitives based on best split
        left_prims = []
        right_prims = []
        for prim in prims:
            if prim.centroid[best_axis] < best_pos:
                left_prims.append(prim)
            else:
                right_prims.append(prim)

        # Handle edge case: all primitives on one side
        if not left_prims or not right_prims:
            # Fall back to median split
            sorted_prims = sorted(prims, key=lambda p: p.centroid[best_axis])
            mid = len(sorted_prims) // 2
            left_prims = sorted_prims[:mid]
            right_prims = sorted_prims[mid:]

        # Recursively build children
        node.left = self._build_recursive(left_prims, depth + 1)
        node.right = self._build_recursive(right_prims, depth + 1)

        # Set parent pointers
        node.left.parent = node
        node.right.parent = node

        return node

    def _find_best_split(self, prims: List[Primitive], parent_bbox: AABB) -> Tuple[int, float, float]:
        """
        Find best split plane using BINNED SAH (much faster than full SAH).

        Bins primitives into buckets and evaluates splits between buckets.
        Complexity: O(N) instead of O(N²)

        Returns: (axis, position, cost)
        """
        NUM_BINS = 16
        best_axis = 0
        best_split_bucket = 0
        best_cost = np.inf

        # Try each axis
        for axis in range(3):
            # Compute centroid bounds along this axis
            min_centroid = min(p.centroid[axis] for p in prims)
            max_centroid = max(p.centroid[axis] for p in prims)

            if max_centroid - min_centroid < 1e-10:
                continue  # All centroids at same position on this axis

            # Initialize bins
            bin_counts = [0] * NUM_BINS
            bin_bboxes = [AABB.empty() for _ in range(NUM_BINS)]

            # Put primitives into bins
            extent = max_centroid - min_centroid
            for prim in prims:
                # Compute bin index
                offset = (prim.centroid[axis] - min_centroid) / extent
                bin_idx = min(NUM_BINS - 1, int(offset * NUM_BINS))

                bin_counts[bin_idx] += 1
                bin_bboxes[bin_idx] = bin_bboxes[bin_idx].union(prim.bbox)

            # Compute costs for splitting after each bin
            # Left-to-right sweep to compute left bbox and count
            left_counts = [0] * (NUM_BINS - 1)
            left_bboxes = [AABB.empty() for _ in range(NUM_BINS - 1)]
            running_bbox = AABB.empty()
            running_count = 0

            for i in range(NUM_BINS - 1):
                running_count += bin_counts[i]
                running_bbox = running_bbox.union(bin_bboxes[i])
                left_counts[i] = running_count
                left_bboxes[i] = running_bbox

            # Right-to-left sweep to compute right bbox and count
            right_counts = [0] * (NUM_BINS - 1)
            right_bboxes = [AABB.empty() for _ in range(NUM_BINS - 1)]
            running_bbox = AABB.empty()
            running_count = 0

            for i in range(NUM_BINS - 2, -1, -1):
                running_count += bin_counts[i + 1]
                running_bbox = running_bbox.union(bin_bboxes[i + 1])
                right_counts[i] = running_count
                right_bboxes[i] = running_bbox

            # Evaluate SAH cost for each split
            parent_sa = parent_bbox.surface_area()
            for i in range(NUM_BINS - 1):
                if left_counts[i] == 0 or right_counts[i] == 0:
                    continue

                left_sa = left_bboxes[i].surface_area()
                right_sa = right_bboxes[i].surface_area()

                cost = (self.traverse_cost +
                       (left_sa / parent_sa) * self.intersect_cost * left_counts[i] +
                       (right_sa / parent_sa) * self.intersect_cost * right_counts[i])

                if cost < best_cost:
                    best_cost = cost
                    best_axis = axis
                    best_split_bucket = i

        # Convert bucket split to actual position
        if best_cost < np.inf:
            min_centroid = min(p.centroid[best_axis] for p in prims)
            max_centroid = max(p.centroid[best_axis] for p in prims)
            extent = max_centroid - min_centroid
            # Split after bucket best_split_bucket
            best_pos = min_centroid + (best_split_bucket + 1) * extent / NUM_BINS
        else:
            # Fallback to median
            sorted_prims = sorted(prims, key=lambda p: p.centroid[0])
            best_pos = sorted_prims[len(sorted_prims) // 2].centroid[0]
            best_axis = 0

        return best_axis, best_pos, best_cost

    def flatten(self) -> Dict[str, np.ndarray]:
        """
        Flatten BVH tree into linear arrays for GPU upload.
        Uses depth-first traversal to maintain cache coherence.

        Returns dict with numpy arrays ready for GPU.
        """
        if self.root is None:
            return {
                'bvh_bbox_min': np.zeros((0, 3), dtype=np.float32),
                'bvh_bbox_max': np.zeros((0, 3), dtype=np.float32),
                'bvh_left_child': np.array([], dtype=np.int32),
                'bvh_right_child': np.array([], dtype=np.int32),
                'bvh_parent': np.array([], dtype=np.int32),
                'bvh_prim_type': np.array([], dtype=np.int32),
                'bvh_prim_idx': np.array([], dtype=np.int32),
                'num_bvh_nodes': 0
            }

        nodes_list = []

        def flatten_node(node: SAHBVHNode, parent_idx: int) -> int:
            """Recursively flatten tree"""
            if node is None:
                return -1

            current_idx = len(nodes_list)
            node.node_idx = current_idx

            # Add placeholder
            nodes_list.append({
                'bbox_min': node.bbox.min.copy(),
                'bbox_max': node.bbox.max.copy(),
                'left': -1,
                'right': -1,
                'parent': parent_idx,
                'prim_type': node.prim_type,
                'prim_idx': node.prim_idx
            })

            # Recursively flatten children
            if not node.is_leaf():
                left_idx = flatten_node(node.left, current_idx)
                right_idx = flatten_node(node.right, current_idx)
                nodes_list[current_idx]['left'] = left_idx
                nodes_list[current_idx]['right'] = right_idx

            return current_idx

        # Flatten starting from root (parent = -1)
        flatten_node(self.root, -1)

        # Convert to numpy arrays
        n = len(nodes_list)
        bvh_bbox_min = np.zeros((n, 3), dtype=np.float32)
        bvh_bbox_max = np.zeros((n, 3), dtype=np.float32)
        bvh_left_child = np.full(n, -1, dtype=np.int32)
        bvh_right_child = np.full(n, -1, dtype=np.int32)
        bvh_parent = np.full(n, -1, dtype=np.int32)
        bvh_prim_type = np.zeros(n, dtype=np.int32)
        bvh_prim_idx = np.full(n, -1, dtype=np.int32)

        for i, node_data in enumerate(nodes_list):
            bvh_bbox_min[i] = node_data['bbox_min']
            bvh_bbox_max[i] = node_data['bbox_max']
            bvh_left_child[i] = node_data['left']
            bvh_right_child[i] = node_data['right']
            bvh_parent[i] = node_data['parent']
            bvh_prim_type[i] = node_data['prim_type']
            bvh_prim_idx[i] = node_data['prim_idx']

        return {
            'bvh_bbox_min': bvh_bbox_min,
            'bvh_bbox_max': bvh_bbox_max,
            'bvh_left_child': bvh_left_child,
            'bvh_right_child': bvh_right_child,
            'bvh_parent': bvh_parent,
            'bvh_prim_type': bvh_prim_type,
            'bvh_prim_idx': bvh_prim_idx,
            'num_bvh_nodes': n
        }


def build_sah_bvh_from_spheres(spheres: List) -> Dict[str, np.ndarray]:
    """
    Build SAH BVH from list of sphere objects.

    Args:
        spheres: List of Sphere objects from scene

    Returns: Flattened BVH arrays ready for GPU upload
    """
    builder = SAHBVHBuilder()

    # Add all spheres to builder
    for i, sphere in enumerate(spheres):
        center = sphere.center.at(0.0)  # Get center at t=0
        center_np = np.array([center.x, center.y, center.z], dtype=np.float32)
        builder.add_sphere(i, center_np, sphere.radius)

    # Build BVH
    builder.build()

    # Flatten to arrays
    return builder.flatten()


def build_sah_bvh_from_primitives(spheres: List, quads: List, triangles: List = None) -> Dict[str, np.ndarray]:
    """
    Build SAH BVH from mixed primitive types (spheres, quads, and triangles).

    Args:
        spheres: List of Sphere objects from scene
        quads: List of quad objects from scene
        triangles: List of triangle objects from scene

    Returns: Flattened BVH arrays ready for GPU upload
    """
    if triangles is None:
        triangles = []

    builder = SAHBVHBuilder()

    # Add all spheres to builder
    for i, sphere in enumerate(spheres):
        center = sphere.center.at(0.0)  # Get center at t=0
        center_np = np.array([center.x, center.y, center.z], dtype=np.float32)
        builder.add_sphere(i, center_np, sphere.radius)

    # Add all quads to builder
    for i, q in enumerate(quads):
        Q_np = np.array([q.Q.x, q.Q.y, q.Q.z], dtype=np.float32)
        u_np = np.array([q.u.x, q.u.y, q.u.z], dtype=np.float32)
        v_np = np.array([q.v.x, q.v.y, q.v.z], dtype=np.float32)
        builder.add_quad(i, Q_np, u_np, v_np)

    # Add all triangles to builder
    for i, tri in enumerate(triangles):
        v0_np = np.array([tri.v0.x, tri.v0.y, tri.v0.z], dtype=np.float32)
        v1_np = np.array([tri.v1.x, tri.v1.y, tri.v1.z], dtype=np.float32)
        v2_np = np.array([tri.v2.x, tri.v2.y, tri.v2.z], dtype=np.float32)
        builder.add_triangle(i, v0_np, v1_np, v2_np)

    # Build BVH
    builder.build()

    # Flatten to arrays
    return builder.flatten()
