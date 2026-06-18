"""Put-Everything-In-Box task for the YAM bimanual robot.

Thin subclass of BimanualPickPlaceEnv preserving the original two-object scene
(lego duplo left, tennis ball right) and box. Kept as a registered id for
backward compatibility with existing scripts/sbatch.
"""

from mani_skill.utils.registration import register_env

from ..pickplace_base import BimanualPickPlaceEnv


@register_env("BimanualYAMPutEverythingInBox-v1", max_episode_steps=1200)
class BimanualYAMPutEverythingInBoxEnv(BimanualPickPlaceEnv):
    """Two objects beside an open box; place both inside. Default robot = YAM."""

    OBJECTS = [
        ("073-a_lego_duplo", (-0.30, 0.22), "left"),
        ("056_tennis_ball",  (-0.30, -0.22), "right"),
    ]
    CONTAINER = {
        "kind": "box",
        "inner_half": 0.09,
        "pos": (-0.15, 0.0),
        "height": 0.06,
        "wall": 0.008,
        "color": [0.55, 0.35, 0.18, 1.0],
    }
    INSTRUCTION = "put everything into the box"

    def __init__(self, *args, robot_uids="bimanual_yam", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
