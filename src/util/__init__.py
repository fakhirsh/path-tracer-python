"""
Utility classes for 3D graphics and ray tracing.

Usage:
    # Import specific classes/functions
    from util import vec3, point3, color, dot, cross
    
    # Or import everything
    from util import *
    
Examples:
    # Create vectors, points, and colors (all use vec3 under the hood)
    direction = vec3(1, 0, 0)
    origin = point3(0, 0, 0)
    red = color(1, 0, 0)
    
    # All vec3 operations work on point3 and color
    offset = point3(1, 2, 3) + vec3(0, 1, 0)  # point3(1, 3, 3)
    blended = color(1, 0, 0).lerp(color(0, 0, 1), 0.5)  # purple
    
    # Use convenience functions
    cos_angle = dot(vec3(1, 0, 0), vec3(0, 1, 0))  # 0.0
    normal = cross(vec3(1, 0, 0), vec3(0, 1, 0))   # vec3(0, 0, 1)
    unit = normalize(vec3(3, 4, 0))  # vec3(0.6, 0.8, 0)
"""

from .vec3 import *
from .color import color, write_color
from .ray import Ray
from .rtw_image import rtw_image

# Type alias for semantic clarity
point3 = vec3  # 3D point in space

__all__ = [
    # Core class
    'vec3',
    'Ray',
    'rtw_image',
    # Type aliases
    'point3',
    'color',
    # Convenience functions
    'dot',
    'cross',
    'length',
    'normalize',
    'distance',
    'lerp',
    'write_color',
    'degrees_to_radians',
    'random_unit_vector',
    'random_on_hemisphere',
    'reflect',
    'refract',
    'random_in_unit_disk',
]