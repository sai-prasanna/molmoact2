"""Put-Everything-In-Box task for the bimanual TRLC-DK1 robot.

Thin subclass of BimanualPickPlaceEnv, same scene/objects/box/success as the
YAM version but defaulting to the DK1 robot. DK1 shares YAM's kinematics, so the
MolmoAct2-YAM policy drives it via the identity arm map. Kept as a registered id
for backward compatibility.
"""

from mani_skill.utils.registration import register_env

from ..pickplace_base import BimanualPickPlaceEnv


@register_env("DK1PutEverythingInBox-v1", max_episode_steps=1200)
class DK1PutEverythingInBoxEnv(BimanualPickPlaceEnv):
    """Two objects beside an open box; place both inside. Default robot = DK1."""

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

    def __init__(self, *args, robot_uids="bimanual_dk1", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
