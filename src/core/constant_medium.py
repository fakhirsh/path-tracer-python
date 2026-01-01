
from core import aabb, interval
from core.hittable import hit_record, hittable
from core.material import isotropic
from core.texture import texture
from util.ray import Ray
from util import vec3
import math
import random

class constant_medium(hittable):
    def __init__(self):
        pass

    @classmethod
    def from_texture(cls, boundary: hittable, tex: texture, density: float):
        obj = cls()
        obj.boundary = boundary
        obj.phase_function = isotropic.from_texture(tex)
        obj.neg_inv_density = -1 / density
        return obj
    
    @classmethod
    def from_color(cls, boundary: hittable, color: tuple, density: float):
        obj = cls()
        obj.boundary = boundary
        obj.phase_function = isotropic.from_color(color)
        obj.neg_inv_density = -1 / density
        return obj
    
    def hit(self, r: Ray, ray_t: interval, rec: hit_record) -> bool:
        # Implementation of hit method for constant_medium
        rec1, rec2 = hit_record(), hit_record()
        if not self.boundary.hit(r, interval.universe, rec1):
            return False
        if not self.boundary.hit(r, interval.from_floats(rec1.t + 0.0001, float('inf')), rec2):
            return False
        if rec1.t < ray_t.min:
            rec1.t = ray_t.min
        if rec2.t > ray_t.max:
            rec2.t = ray_t.max
        if rec1.t >= rec2.t:
            return False
        if rec1.t < 0:
            rec1.t = 0
        ray_length = r.direction.length()
        distance_inside_boundary = (rec2.t - rec1.t) * ray_length
        hit_distance = self.neg_inv_density * math.log(random.random())
        if hit_distance > distance_inside_boundary:
            return False
        rec.t = rec1.t + hit_distance / ray_length
        rec.p = r.at(rec.t)
        rec.normal = vec3(1, 0, 0)  # arbitrary
        rec.front_face = True  # also arbitrary
        rec.material = self.phase_function
        return True
    
    def bounding_box(self) -> aabb:
        return self.boundary.bounding_box()