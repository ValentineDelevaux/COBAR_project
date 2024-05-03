import numpy as np
import flygym as flygym
from flygym.arena import BaseArena
from typing import Tuple, Optional


class FlatTerrainColored(BaseArena):
    """Flat terrain with no obstacles.

    Attributes
    ----------
    root_element : mjcf.RootElement
        The root MJCF element of the arena.
    friction : Tuple[float, float, float]
        The sliding, torsional, and rolling friction coefficients of the
        ground, by default (1, 0.005, 0.0001).

    Parameters
    ----------
    size : Tuple[float, float], optional
        The size of the arena in mm, by default (50, 50).
    friction : Tuple[float, float, float]
        The sliding, torsional, and rolling friction coefficients of the
        ground, by default (1, 0.005, 0.0001).
    ground_alpha : float
        Opacity of the ground, by default 1 (fully opaque).
    scale_bar_pos : Tuple[float, float, float], optional
        If supplied, a 1 mm scale bar will be placed at this location.
    """

    def __init__(
        self,
        size: Tuple[float, float] = (100, 100),
        friction: Tuple[float, float, float] = (1, 0.005, 0.0001),
        ground_alpha: float = 1.0,
        rgb1: Tuple[float, float, float] = (0.3, 0.3, 0.3),
        rgb2: Tuple[float, float, float] = (0.4, 0.4, 0.4), 
        scale_bar_pos: Optional[Tuple[float, float, float]] = None,
    ):
        super().__init__()

        ground_size = [*size, 1]
        chequered = self.root_element.asset.add(
            "texture",
            type="2d",
            builtin="checker",
            width=300,
            height=300,
            rgb1=rgb1,
            rgb2=rgb2,
        )
        grid = self.root_element.asset.add(
            "material",
            name="grid",
            texture=chequered,
            texrepeat=(10, 10),
            reflectance=0.1,
            rgba=(1.0, 1.0, 1.0, ground_alpha),
        )
        self.root_element.worldbody.add(
            "geom",
            type="plane",
            name="ground",
            material=grid,
            size=ground_size,
            friction=friction,
        )
        self.friction = friction
        if scale_bar_pos:
            self.root_element.worldbody.add(
                "geom",
                type="cylinder",
                size=(0.05, 0.5),
                pos=scale_bar_pos,
                rgba=(0, 0, 0, 1),
                euler=(0, np.pi / 2, 0),
            )

    def get_spawn_position(
        self, rel_pos: np.ndarray, rel_angle: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        return rel_pos, rel_angle
