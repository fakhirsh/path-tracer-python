import math
from typing import Union
import random

class vec3:
    """
    A 3D vector class with common vector operations.
    Supports addition, subtraction, scalar multiplication, dot product, cross product, and more.
    """
    
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        """Initialize a 3D vector with x, y, z components."""
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
    
    def __repr__(self) -> str:
        """String representation of the vector."""
        return f"vec3({self.x}, {self.y}, {self.z})"
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"({self.x}, {self.y}, {self.z})"
    
    def __eq__(self, other: object) -> bool:
        """Check if two vectors are equal."""
        if not isinstance(other, vec3):
            return False
        return self.x == other.x and self.y == other.y and self.z == other.z
    
    def __add__(self, other: 'vec3') -> 'vec3':
        """Vector addition: self + other."""
        if not isinstance(other, vec3):
            raise TypeError("Can only add vec3 to vec3")
        return vec3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other: 'vec3') -> 'vec3':
        """Vector subtraction: self - other."""
        if not isinstance(other, vec3):
            raise TypeError("Can only subtract vec3 from vec3")
        return vec3(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, other: Union[int, float, 'vec3']) -> 'vec3':
        """Scalar multiplication: self * scalar or component-wise multiplication: self * vec3."""
        if isinstance(other, (int, float)):
            return vec3(self.x * other, self.y * other, self.z * other)
        elif isinstance(other, vec3):
            return vec3(self.x * other.x, self.y * other.y, self.z * other.z)
        else:
            raise TypeError("Can only multiply vec3 by scalar (int or float) or vec3")
    
    def __rmul__(self, scalar: Union[int, float]) -> 'vec3':
        """Scalar multiplication: scalar * self."""
        return self.__mul__(scalar)
    
    def __truediv__(self, scalar: Union[int, float]) -> 'vec3':
        """Scalar division: self / scalar."""
        if not isinstance(scalar, (int, float)):
            raise TypeError("Can only divide vec3 by scalar (int or float)")
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide by zero")
        return vec3(self.x / scalar, self.y / scalar, self.z / scalar)
    
    def __neg__(self) -> 'vec3':
        """Unary negation: -self."""
        return vec3(-self.x, -self.y, -self.z)
    
    def __abs__(self) -> float:
        """Magnitude (length) of the vector."""
        return self.length()
    
    def dot(self, other: 'vec3') -> float:
        """
        Dot product of two vectors.
        Returns: self · other = self.x * other.x + self.y * other.y + self.z * other.z
        """
        if not isinstance(other, vec3):
            raise TypeError("Dot product requires two vec3 objects")
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def cross(self, other: 'vec3') -> 'vec3':
        """
        Cross product of two vectors.
        Returns: self × other
        """
        if not isinstance(other, vec3):
            raise TypeError("Cross product requires two vec3 objects")
        return vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
    
    def length(self) -> float:
        """Magnitude (length) of the vector."""
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def length_squared(self) -> float:
        """Squared magnitude of the vector (faster than length() for comparisons)."""
        return self.x**2 + self.y**2 + self.z**2
    
    def normalize(self) -> 'vec3':
        """Return a unit vector in the same direction."""
        len_val = self.length()
        if len_val == 0:
            raise ZeroDivisionError("Cannot normalize zero vector")
        return self / len_val
    
    def normalized(self) -> 'vec3':
        """Alias for normalize() for consistency with some libraries."""
        return self.normalize()
    
    def unit_vector(self) -> 'vec3':
        """Alias for normalize()."""
        return self.normalize()
    
    def distance_to(self, other: 'vec3') -> float:
        """Distance between this vector and another vector."""
        return (self - other).length()
    
    def distance_squared_to(self, other: 'vec3') -> float:
        """Squared distance between this vector and another vector."""
        return (self - other).length_squared()
    
    def angle_with(self, other: 'vec3') -> float:
        """
        Angle between this vector and another vector in radians.
        Returns angle in range [0, π].
        """
        if not isinstance(other, vec3):
            raise TypeError("Angle calculation requires two vec3 objects")
        
        dot_product = self.dot(other)
        len1 = self.length()
        len2 = other.length()
        
        if len1 == 0 or len2 == 0:
            raise ValueError("Cannot calculate angle with zero vector")
        
        # Clamp to [-1, 1] to avoid numerical errors
        cos_angle = max(-1.0, min(1.0, dot_product / (len1 * len2)))
        return math.acos(cos_angle)
    
    def is_parallel_to(self, other: 'vec3', tolerance: float = 1e-10) -> bool:
        """Check if this vector is parallel to another vector."""
        if not isinstance(other, vec3):
            raise TypeError("Parallel check requires two vec3 objects")
        
        # Check if cross product is zero (within tolerance)
        cross_prod = self.cross(other)
        return cross_prod.length() < tolerance
    
    def is_perpendicular_to(self, other: 'vec3', tolerance: float = 1e-10) -> bool:
        """Check if this vector is perpendicular to another vector."""
        if not isinstance(other, vec3):
            raise TypeError("Perpendicular check requires two vec3 objects")
        
        # Check if dot product is zero (within tolerance)
        return abs(self.dot(other)) < tolerance
    
    def lerp(self, other: 'vec3', t: float) -> 'vec3':
        """
        Linear interpolation between this vector and another vector.
        t=0 returns self, t=1 returns other.
        """
        if not isinstance(other, vec3):
            raise TypeError("Lerp requires two vec3 objects")
        return self + (other - self) * t
    
    def reflect(self, normal: 'vec3') -> 'vec3':
        """
        Reflect this vector across a surface with given normal.
        Assumes normal is a unit vector.
        """
        if not isinstance(normal, vec3):
            raise TypeError("Reflection requires vec3 normal")
        return self - normal * (2 * self.dot(normal))
    
    def project_onto(self, other: 'vec3') -> 'vec3':
        """Project this vector onto another vector."""
        if not isinstance(other, vec3):
            raise TypeError("Projection requires two vec3 objects")
        
        other_len_sq = other.length_squared()
        if other_len_sq == 0:
            raise ZeroDivisionError("Cannot project onto zero vector")
        
        return other * (self.dot(other) / other_len_sq)
    
    def to_list(self) -> list:
        """Convert vector to list [x, y, z]."""
        return [self.x, self.y, self.z]
    
    def to_tuple(self) -> tuple:
        """Convert vector to tuple (x, y, z)."""
        return (self.x, self.y, self.z)
    
    def copy(self) -> 'vec3':
        """Create a copy of this vector."""
        return vec3(self.x, self.y, self.z)
    
    def near_zero(self, tolerance: float = 1e-8) -> bool:
        """Check if the vector is close to zero in all dimensions."""
        return abs(self.x) < tolerance and abs(self.y) < tolerance and abs(self.z) < tolerance
    
    @staticmethod
    def zero() -> 'vec3':
        """Create a zero vector (0, 0, 0)."""
        return vec3(0, 0, 0)
    
    @staticmethod
    def one() -> 'vec3':
        """Create a vector of all ones (1, 1, 1)."""
        return vec3(1, 1, 1)
    
    @staticmethod
    def up() -> 'vec3':
        """Create an up vector (0, 1, 0)."""
        return vec3(0, 1, 0)
    
    @staticmethod
    def right() -> 'vec3':
        """Create a right vector (1, 0, 0)."""
        return vec3(1, 0, 0)
    
    @staticmethod
    def forward() -> 'vec3':
        """Create a forward vector (0, 0, 1)."""
        return vec3(0, 0, 1)
    
    @staticmethod
    def random(min_val: float = 0.0, max_val: float = 1.0) -> 'vec3':
        """Create a random vector with components in [min_val, max_val]."""
        import random
        return vec3(
            random.uniform(min_val, max_val),
            random.uniform(min_val, max_val),
            random.uniform(min_val, max_val)
        )

# Convenience aliases for common operations
def dot(v1: vec3, v2: vec3) -> float:
    """Dot product of two vectors."""
    return v1.dot(v2)

def cross(v1: vec3, v2: vec3) -> vec3:
    """Cross product of two vectors."""
    return v1.cross(v2)

def length(v: vec3) -> float:
    """Magnitude of a vector."""
    return v.length()

def normalize(v: vec3) -> vec3:
    """Normalize a vector to unit length."""
    return v.normalize()

def distance(v1: vec3, v2: vec3) -> float:
    """Distance between two vectors."""
    return v1.distance_to(v2)

def lerp(v1: vec3, v2: vec3, t: float) -> vec3:
    """Linear interpolation between two vectors."""
    return v1.lerp(v2, t)

def degrees_to_radians(degrees: float) -> float:
    """Convert degrees to radians."""
    return degrees * math.pi / 180.0

def random_unit_vector() -> vec3:
    """Generate a random unit vector."""
    while True:
        p = vec3.random(-1, 1)
        if p.length_squared() >= 1 or p.length_squared() <= 1e-20:
            continue
        return p.normalize()
    
def random_on_hemisphere(normal: vec3) -> vec3:
    on_unit_sphere = random_unit_vector()
    if dot(on_unit_sphere, normal) > 0.0:
        return on_unit_sphere
    else:
        return -on_unit_sphere
    
def reflect(v: vec3, n: vec3) -> vec3:
    """Reflect vector v around normal n."""
    return v - n * (2 * dot(v, n))

def refract(uv: vec3, n: vec3, etai_over_etat: float) -> vec3:
    """Refract vector uv with normal n and ratio etai_over_etat."""
    cos_theta = min(dot(-uv, n), 1.0)
    r_out_perp = etai_over_etat * (uv + cos_theta * n)
    r_out_parallel = -math.sqrt(abs(1.0 - r_out_perp.length_squared())) * n
    return r_out_perp + r_out_parallel

def random_in_unit_disk() -> vec3:
    """Generate a random point inside a unit disk in the XY plane."""
    while True:
        p = vec3(random.uniform(-1, 1), random.uniform(-1, 1), 0)
        if p.length_squared() >= 1:
            continue
        return p
    
def random_cosine_direction(normal: vec3) -> vec3:
    """
    Generate cosine-weighted random direction in hemisphere around normal.
    PDF = cos(θ) / π
    
    Use this instead of (normal + random_unit_vector()) for diffuse surfaces.
    """
    r1 = random.random()
    r2 = random.random()
    
    # Cosine-weighted: z = cos(θ) = sqrt(1 - r2)
    z = math.sqrt(1.0 - r2)
    phi = 2.0 * math.pi * r1
    sin_theta = math.sqrt(r2)
    x = math.cos(phi) * sin_theta
    y = math.sin(phi) * sin_theta
    
    # Build orthonormal basis around normal
    w = normalize(normal)
    # Choose axis most perpendicular to w
    if abs(w.x) < 0.9:
        a = vec3(1, 0, 0)
    elif abs(w.y) < 0.9:
        a = vec3(0, 1, 0)
    else:
        a = vec3(0, 0, 1)
    v = normalize(cross(w, a))
    u = cross(w, v)
    
    # Transform to world space
    return normalize(x * u + y * v + z * w)