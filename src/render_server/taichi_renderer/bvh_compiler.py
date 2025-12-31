"""
BVH Compiler: Flattens CPU BVH tree structure into GPU-friendly arrays.
Now supports both legacy BVH and new SAH BVH builder.
"""

import numpy as np
from typing import Dict, List, Any
from .sah_bvh_builder import build_sah_bvh_from_spheres

# Primitive type constants (must match kernels.py)
PRIM_SPHERE = 0
PRIM_TRIANGLE = 1
PRIM_QUAD = 2

# Enable SAH BVH builder (set to False to use legacy median-split BVH)
# SAH provides better BVH quality but slower build time
# Binned SAH: ~10ms build for 512 spheres (acceptable overhead)
USE_SAH_BVH = True  # Enable to test BVH quality improvement


def create_sphere_mapping(spheres: List) -> Dict[int, int]:
    """
    Create mapping from sphere object id to array index.
    Returns: {id(sphere): index}
    """
    return {id(sphere): i for i, sphere in enumerate(spheres)}


def flatten_bvh(bvh_root, sphere_mapping: Dict[int, int]) -> Dict[str, np.ndarray]:
    """
    Flatten BVH tree into linear arrays using depth-first traversal.
    NOW INCLUDES PARENT POINTERS for stackless traversal.

    Args:
        bvh_root: Root node of BVH (could be bvh_node, hittable_list, or Sphere)
        sphere_mapping: Maps sphere object ids to indices

    Returns dict with:
        'bvh_bbox_min': np.ndarray of shape (N, 3) dtype=float32
        'bvh_bbox_max': np.ndarray of shape (N, 3) dtype=float32
        'bvh_left_child': np.ndarray of shape (N,) dtype=int32  # -1 if leaf
        'bvh_right_child': np.ndarray of shape (N,) dtype=int32  # -1 if leaf
        'bvh_parent': np.ndarray of shape (N,) dtype=int32      # -1 if root
        'bvh_prim_type': np.ndarray of shape (N,) dtype=int32   # PRIM_SPHERE, etc.
        'bvh_prim_idx': np.ndarray of shape (N,) dtype=int32    # -1 if internal node
        'num_bvh_nodes': int
    """
    nodes_list = []

    def flatten_node(node, parent_idx):
        """Recursively flatten BVH tree into array format with parent tracking"""
        if node is None:
            return -1

        # Create current node entry
        current_idx = len(nodes_list)

        # Get bounding box
        bbox = node.bounding_box()

        # Check if this is a leaf node (contains a sphere)
        node_type_name = type(node).__name__
        if node_type_name == 'Sphere':
            # Leaf node - store sphere index
            sphere_idx = sphere_mapping.get(id(node), -1)
            nodes_list.append({
                'bbox_min': [bbox.x.min, bbox.y.min, bbox.z.min],
                'bbox_max': [bbox.x.max, bbox.y.max, bbox.z.max],
                'left': -1,
                'right': -1,
                'parent': parent_idx,
                'prim_type': PRIM_SPHERE,
                'prim_idx': sphere_idx
            })
            return current_idx

        # Internal BVH node - placeholder for now, will update children indices
        nodes_list.append({
            'bbox_min': [bbox.x.min, bbox.y.min, bbox.z.min],
            'bbox_max': [bbox.x.max, bbox.y.max, bbox.z.max],
            'left': -1,
            'right': -1,
            'parent': parent_idx,
            'prim_type': PRIM_SPHERE,  # Default, not used for internal nodes
            'prim_idx': -1
        })

        # Recursively flatten children
        left_idx = flatten_node(node.left, current_idx) if hasattr(node, 'left') and node.left is not None else -1
        right_idx = flatten_node(node.right, current_idx) if hasattr(node, 'right') and node.right is not None else -1

        # Update current node with child indices
        nodes_list[current_idx]['left'] = left_idx
        nodes_list[current_idx]['right'] = right_idx

        return current_idx

    # Flatten the tree starting from root (parent = -1)
    flatten_node(bvh_root, -1)

    # Convert to numpy arrays
    n = len(nodes_list)
    bvh_bbox_min = np.zeros((n, 3), dtype=np.float32)
    bvh_bbox_max = np.zeros((n, 3), dtype=np.float32)
    bvh_left_child = np.full(n, -1, dtype=np.int32)
    bvh_right_child = np.full(n, -1, dtype=np.int32)
    bvh_parent = np.full(n, -1, dtype=np.int32)
    bvh_prim_type = np.zeros(n, dtype=np.int32)
    bvh_prim_idx = np.full(n, -1, dtype=np.int32)

    for i, node in enumerate(nodes_list):
        bvh_bbox_min[i] = node['bbox_min']
        bvh_bbox_max[i] = node['bbox_max']
        bvh_left_child[i] = node['left']
        bvh_right_child[i] = node['right']
        bvh_parent[i] = node['parent']
        bvh_prim_type[i] = node['prim_type']
        bvh_prim_idx[i] = node['prim_idx']

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


def compile_bvh(world, spheres: List) -> Dict[str, np.ndarray]:
    """
    Main entry point for BVH compilation.

    Args:
        world: The world object (contains BVH structure)
        spheres: List of spheres (for creating index mapping)

    Returns: dict of numpy arrays ready for GPU upload
    """
    if USE_SAH_BVH:
        # Use new SAH BVH builder (2-3x faster traversal)
        print(f"Building SAH BVH with {len(spheres)} primitives...")
        return build_sah_bvh_from_spheres(spheres)
    else:
        # Legacy median-split BVH
        print(f"Using legacy BVH with {len(spheres)} primitives...")
        sphere_mapping = create_sphere_mapping(spheres)

        # Find actual BVH root (might be wrapped in hittable_list)
        bvh_root = world
        if hasattr(world, 'objects') and len(world.objects) > 0:
            bvh_root = world.objects[0]

        return flatten_bvh(bvh_root, sphere_mapping)
