from core.hittable import hittable
from core.camera import camera
from util.color import color
from render_server.base_renderer import BaseRenderer
from render_server.gpu_renderer import GpuRenderer
from render_server.cpu_renderer import CpuRenderer
from render_server.taichi_renderer import TaichiRenderer


class RendererFactory:
    """Factory for creating different renderer implementations"""

    _renderers = {
        'gpu': GpuRenderer,
        'cpu': CpuRenderer,
        'taichi': TaichiRenderer,
    }

    @classmethod
    def create(cls, renderer_type: str, world: hittable, cam: camera, img_path: str, **kwargs) -> BaseRenderer:
        """
        Create a renderer instance based on the specified type.

        Args:
            renderer_type: Type of renderer ('gpu', 'cpu', 'taichi')
            world: The scene to render
            cam: Camera configuration
            img_path: Output image path
            **kwargs: Additional renderer parameters (max_depth, background_color, etc.)

        Returns:
            A renderer instance

        Raises:
            ValueError: If renderer_type is not recognized
        """
        renderer_type = renderer_type.lower()

        if renderer_type not in cls._renderers:
            available = ', '.join(cls._renderers.keys())
            raise ValueError(f"Unknown renderer type '{renderer_type}'. Available: {available}")

        renderer_class = cls._renderers[renderer_type]
        return renderer_class(world, cam, img_path, **kwargs)

    # @classmethod
    # def register_renderer(cls, name: str, renderer_class: type):
    #     """
    #     Register a custom renderer implementation.

    #     Args:
    #         name: Identifier for the renderer
    #         renderer_class: Class implementing BaseRenderer
    #     """
    #     if not issubclass(renderer_class, BaseRenderer):
    #         raise TypeError(f"{renderer_class} must inherit from BaseRenderer")

    #     cls._renderers[name.lower()] = renderer_class

    @classmethod
    def get_available_renderers(cls) -> list:
        """Return list of available renderer types"""
        return list(cls._renderers.keys())
