
from abc import ABC, abstractmethod
from random import random
from .texture import texture, solid_color
from util import Ray, color, point3, random_unit_vector, reflect, refract, vec3, random_cosine_direction
from math import log, exp
from core.hittable import hit_record

class material(ABC):

    def emitted(self, u: float, v: float, p: point3) -> color:
        return color(0, 0, 0)

    @abstractmethod
    def scatter(self, r_in: Ray, rec: 'hit_record', attenuation: color, scattered: Ray) -> bool:
        return False

class lambertian(material):
    
    @classmethod
    def from_color(cls, albedo: color) -> "lambertian":
        instance = cls()
        instance.tex = solid_color.from_color(albedo)
        return instance

    @classmethod
    def from_texture(cls, tex: texture) -> "lambertian":
        instance = cls()
        instance.tex = tex
        return instance

    def scatter(self, r_in: Ray, rec: 'hit_record', attenuation: color, scattered: Ray) -> bool:
       
        scatter_direction = random_cosine_direction(rec.normal)

        if scatter_direction.near_zero():
            scatter_direction = rec.normal

        scattered.origin = rec.p
        scattered.direction = scatter_direction        
        scattered.time = r_in.time
        attenuation.x = self.tex.value(rec.u, rec.v, rec.p).x
        attenuation.y = self.tex.value(rec.u, rec.v, rec.p).y
        attenuation.z = self.tex.value(rec.u, rec.v, rec.p).z
        return True
    
class metal(material):
    def __init__(self, albedo: color, fuzz: float):
        self.albedo = albedo
        self.fuzz = fuzz if fuzz < 1.0 else 1.0

    def scatter(self, r_in: Ray, rec: 'hit_record', attenuation: color, scattered: Ray) -> bool:
        scatter_direction = reflect(r_in.direction, rec.normal) + self.fuzz * random_unit_vector()
        scattered.origin = rec.p
        scattered.direction = scatter_direction
        scattered.time = r_in.time
        attenuation.x = self.albedo.x
        attenuation.y = self.albedo.y
        attenuation.z = self.albedo.z
        return True
    
class dielectric(material):
    def __init__(self, index_of_refraction: float):
        self.ir = index_of_refraction

    def scatter(self, r_in: Ray, rec: 'hit_record', attenuation: color, scattered: Ray) -> bool:
        attenuation.x = 1.0
        attenuation.y = 1.0
        attenuation.z = 1.0

        refraction_ratio = (1.0 / self.ir) if rec.front_face else self.ir

        unit_direction = r_in.direction.unit_vector()
        cos_theta = min(-unit_direction.dot(rec.normal), 1.0)
        sin_theta = (1.0 - cos_theta * cos_theta) ** 0.5

        cannot_refract = refraction_ratio * sin_theta > 1.0
        direction = None

        if cannot_refract or self._reflectance(cos_theta, refraction_ratio) > random():
            direction = reflect(unit_direction, rec.normal)
        else:
            direction = refract(unit_direction, rec.normal, refraction_ratio)

        scattered.origin = rec.p
        scattered.direction = direction
        scattered.time = r_in.time
        return True
    
    def _reflectance(self, cosine: float, ref_idx: float) -> float:
        r0 = (1 - ref_idx) / (1 + ref_idx)
        r0 = r0 * r0
        return r0 + (1 - r0) * ((1 - cosine) ** 5)
    
#----------------------------------------------------------------------------------------

class diffuse_light(material):

    @classmethod
    def from_texture(cls, tex: texture) -> "diffuse_light":
        instance = cls()
        instance.tex = tex
        return instance

    @classmethod
    def from_color(cls, emit_color: color) -> "diffuse_light":
        instance = cls()
        instance.tex = solid_color.from_color(emit_color)
        return instance

    def emitted(self, u: float, v: float, p: point3) -> color:
        return self.tex.value(u, v, p)

    def scatter(self, r_in: Ray, rec: 'hit_record', attenuation: color, scattered: Ray) -> bool:
        return False
#----------------------------------------------------------------------------------------

class isotropic(material):
    def __init__(self):
        pass

    @classmethod
    def from_texture(cls, tex: texture) -> "isotropic":
        instance = cls()
        instance.tex = tex
        return instance
    
    @classmethod
    def from_color(cls, albedo: color) -> "isotropic":
        instance = cls()
        instance.tex = solid_color.from_color(albedo)
        return instance
   
    def scatter(self, r_in: Ray, rec: 'hit_record', attenuation: color, scattered: Ray) -> bool:
        scattered.origin = rec.p
        scattered.direction = random_unit_vector()
        scattered.time = r_in.time
        attenuation.x = self.tex.value(rec.u, rec.v, rec.p).x
        attenuation.y = self.tex.value(rec.u, rec.v, rec.p).y
        attenuation.z = self.tex.value(rec.u, rec.v, rec.p).z
        return True

#----------------------------------------------------------------------------------------

class subsurface_simple(material):
    """
    Simple SSS approximation for waxy/translucent materials.
    """
    def __init__(self, albedo: color, scatter_distance: float):
        self.albedo = albedo
        self.scatter_distance = scatter_distance
    
    def scatter(self, r_in: Ray, rec: 'hit_record', attenuation: color, scattered: Ray) -> bool:
        if random() < 0.5:
            # Regular diffuse scatter
            scatter_direction = rec.normal + random_unit_vector()
            exit_point = rec.p
        else:
            # Subsurface: light exits at displaced point
            displacement = random_unit_vector() * self.scatter_distance * random()
            exit_point = rec.p + displacement
            scatter_direction = rec.normal + random_unit_vector()
        
        if scatter_direction.near_zero():
            scatter_direction = rec.normal
        
        scattered.origin = exit_point
        scattered.direction = scatter_direction
        scattered.time = r_in.time
        
        attenuation.x = self.albedo.x
        attenuation.y = self.albedo.y
        attenuation.z = self.albedo.z
        return True

class subsurface_volumetric(material):
    """
    Volumetric SSS using random walk inside the medium.
    Models actual light transport through translucent materials.
    """
    def __init__(self, albedo: color, scatter_coeff: float, absorb_coeff: float, g: float = 0.0):
        """
        albedo: surface color
        scatter_coeff: σs - how often light scatters (higher = more opaque)
        absorb_coeff: σa - how much light is absorbed (higher = darker)
        g: anisotropy parameter for Henyey-Greenstein phase function
           g = 0: isotropic scattering
           g > 0: forward scattering (like skin)
           g < 0: back scattering
        """
        self.albedo = albedo
        self.sigma_s = scatter_coeff
        self.sigma_a = absorb_coeff
        self.sigma_t = scatter_coeff + absorb_coeff  # extinction coefficient
        self.g = g
        self.max_bounces = 64  # prevent infinite loops
    
    def scatter(self, r_in: Ray, rec: 'hit_record', attenuation: color, scattered: Ray) -> bool:
        # Start inside the material, just past the surface
        current_pos = rec.p - rec.normal * 0.001
        current_dir = r_in.direction.unit_vector()
        
        throughput = color(1.0, 1.0, 1.0)
        
        for _ in range(self.max_bounces):
            # Sample distance to next interaction (exponential distribution)
            t = -log(max(random(), 1e-10)) / self.sigma_t
            current_pos = current_pos + current_dir * t
            
            # Check if we've exited the object
            # (In a full implementation, you'd ray-march and check geometry)
            # For simplicity, we'll use a probabilistic exit based on distance
            distance_from_entry = (current_pos - rec.p).length()
            exit_probability = 1.0 - exp(-distance_from_entry * 0.5)
            
            if random() < exit_probability:
                # Exited the material - do a diffuse scatter outward
                scatter_direction = rec.normal + random_unit_vector()
                if scatter_direction.near_zero():
                    scatter_direction = rec.normal
                
                scattered.origin = current_pos
                scattered.direction = scatter_direction
                scattered.time = r_in.time
                
                # Apply accumulated throughput and albedo
                attenuation.x = throughput.x * self.albedo.x
                attenuation.y = throughput.y * self.albedo.y
                attenuation.z = throughput.z * self.albedo.z
                return True
            
            # Still inside - scatter or absorb?
            if random() < self.sigma_a / self.sigma_t:
                # Absorbed
                attenuation.x = 0.0
                attenuation.y = 0.0
                attenuation.z = 0.0
                return False
            
            # Scatter - pick new direction using phase function
            current_dir = self._sample_henyey_greenstein(current_dir)
            
            # Attenuate based on albedo
            throughput.x *= self.albedo.x
            throughput.y *= self.albedo.y
            throughput.z *= self.albedo.z
        
        # Exceeded max bounces - treat as absorbed
        return False
    
    def _sample_henyey_greenstein(self, incident: vec3) -> vec3:
        """Sample direction from Henyey-Greenstein phase function."""
        if abs(self.g) < 1e-3:
            # Isotropic - just return random direction
            return random_unit_vector()
        
        # Sample cos(theta) from HG distribution
        sqr_term = (1 - self.g * self.g) / (1 - self.g + 2 * self.g * random())
        cos_theta = (1 + self.g * self.g - sqr_term * sqr_term) / (2 * self.g)
        sin_theta = (1 - cos_theta * cos_theta) ** 0.5
        
        # Random azimuthal angle
        phi = 2 * 3.14159265 * random()
        
        # Create local coordinate system
        w = incident.unit_vector()
        a = vec3(1, 0, 0) if abs(w.x) > 0.9 else vec3(0, 1, 0)
        v = w.cross(a).unit_vector()
        u = w.cross(v)
        
        # Construct new direction
        from math import cos as mcos, sin as msin
        direction = (u * mcos(phi) * sin_theta + 
                    v * msin(phi) * sin_theta + 
                    w * cos_theta)
        return direction.unit_vector()
    
#----------------------------------------------------------------------------------