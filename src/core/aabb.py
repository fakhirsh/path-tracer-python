from .interval import interval
from util import point3, Ray

class aabb:
    def __init__(self):
        # Three intervals:
        self.x = None
        self.y = None
        self.z = None

    @classmethod
    def from_intervals(cls, x_interval: interval, y_interval: interval, z_interval: interval) -> "aabb":
        box = cls()
        box.x = x_interval
        box.y = y_interval
        box.z = z_interval
        return box
    
    @classmethod
    def from_points(cls, a: point3, b: point3) -> "aabb":
        box = cls()
        box.x = interval.from_floats(a.x, b.x) if a.x < b.x else interval.from_floats(b.x, a.x)
        box.y = interval.from_floats(a.y, b.y) if a.y < b.y else interval.from_floats(b.y, a.y)
        box.z = interval.from_floats(a.z, b.z) if a.z < b.z else interval.from_floats(b.z, a.z)
        return box
    
    @classmethod
    def from_aabbs(cls, a: "aabb", b: "aabb") -> "aabb":
        box = cls()
        box.x = interval.from_intervals(a.x, b.x)
        box.y = interval.from_intervals(a.y, b.y)
        box.z = interval.from_intervals(a.z, b.z)
        return box

    def axis_interval(self, n: int) -> interval:
        if n == 0:
            return self.x
        elif n == 1:
            return self.y
        elif n == 2:
            return self.z
        else:
            raise ValueError("Axis must be 0, 1, or 2.")
        
    def hit(self, r: Ray, ray_t: interval) -> bool:
        ray_orig = r.origin
        ray_dir = r.direction
        ray_dir_arr = [ray_dir.x, ray_dir.y, ray_dir.z]
        ray_orig_arr = [ray_orig.x, ray_orig.y, ray_orig.z]

        # Work with local copies to avoid mutating the input interval
        t_min = ray_t.min
        t_max = ray_t.max

        for axis in range(3):
            ax = self.axis_interval(axis)

            adivn = 1.0 / ray_dir_arr[axis]
            t0 = (ax.min - ray_orig_arr[axis]) * adivn
            t1 = (ax.max - ray_orig_arr[axis]) * adivn
            if t0 < t1:
                if t0 > t_min:
                    t_min = t0
                if t1 < t_max:
                    t_max = t1
            else:
                if t1 > t_min:
                    t_min = t1
                if t0 < t_max:
                    t_max = t0

            if t_max <= t_min:
                return False
        return True
    
    def longest_axis(self) -> int:
        if self.x.size() >= self.y.size():
            return 0 if self.x.size() >= self.z.size() else 2
        else:
            return 1 if self.y.size() >= self.z.size() else 2
    
    def _pad_to_minimums(self):
        # Adjust the AABB so that no side is narrower than some delta, padding if necessary.
        delta = 0.0001
        if self.x.size() < delta:
            self.x = self.x.expand(delta)
        if self.y.size() < delta:
            self.y = self.y.expand(delta)
        if self.z.size() < delta:
            self.z = self.z.expand(delta)

    def surface_area(self) -> float:
        self._pad_to_minimums()
        a = self.x.size()
        b = self.y.size()
        c = self.z.size()
        return 2.0 * (a * b + b * c + c * a)