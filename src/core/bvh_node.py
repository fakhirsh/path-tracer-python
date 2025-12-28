
from util import Ray
from core import hit_record, interval, hittable, hittable_list, aabb
import random
from functools import cmp_to_key

class bvh_node(hittable):

    @classmethod
    def from_list(cls, list: hittable_list) -> "bvh_node":
        return None
    
    @classmethod
    def from_objects(cls, objects: list[hittable], start: int, end: int) -> "bvh_node":
        instance = cls.__new__(cls)

        # Initialize bbox with the first object's bounding box
        instance.bbox = objects[start].bounding_box()

        # Merge remaining object bounding boxes
        for object_index in range(start + 1, end):
            instance.bbox = aabb.from_aabbs(instance.bbox, objects[object_index].bounding_box())

        axis = instance.bbox.longest_axis()

        comparator = None
        if axis == 0:
            comparator = bvh_node.box_x_compare
        elif axis == 1:
            comparator = bvh_node.box_y_compare
        else:
            comparator = bvh_node.box_z_compare

        object_span = end - start
        if object_span == 1:
            instance.left = instance.right = objects[start]
        elif object_span == 2:
            instance.left = objects[start]
            instance.right = objects[start + 1]
        else:
            objects[start:end] = sorted(objects[start:end], key=cmp_to_key(comparator))
            mid = start + object_span // 2

            instance.left = cls.from_objects(objects, start, mid)
            instance.right = cls.from_objects(objects, mid, end)

        return instance

    def hit(self, r: Ray, ray_t: interval, rec: hit_record) -> bool:
        if not self.bbox.hit(r, ray_t):
            return False
        
        hit_left = self.left.hit(r, ray_t, rec)
        hit_right = self.right.hit(r, interval.from_floats(ray_t.min, rec.t if hit_left else ray_t.max), rec)
        return hit_left or hit_right
    
    def bounding_box(self) -> aabb:
        return self.bbox
    
    @staticmethod
    def box_compare(a: hittable, b: hittable, axis_index: int) -> int:
        a_axis_interval = a.bounding_box().axis_interval(axis_index)
        b_axis_interval = b.bounding_box().axis_interval(axis_index)
        if a_axis_interval.min < b_axis_interval.min:
            return -1
        elif a_axis_interval.min > b_axis_interval.min:
            return 1
        else:
            return 0

    @staticmethod
    def box_x_compare(a: hittable, b: hittable) -> int:
        return bvh_node.box_compare(a, b, 0)

    @staticmethod
    def box_y_compare(a: hittable, b: hittable) -> int:
        return bvh_node.box_compare(a, b, 1)

    @staticmethod
    def box_z_compare(a: hittable, b: hittable) -> int:
        return bvh_node.box_compare(a, b, 2)
