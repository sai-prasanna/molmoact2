# sim_eval/tasks/pickplace_tasks/block_stack.py
"""Block-stack task: stack three primitive colored blocks.

The outlier task — primitive box geometry (no YCB asset), no container, and a
stacking success criterion. Overrides _build_objects, sets CONTAINER=None, and
overrides success_fn to use blocks_stacked.
"""

import sapien
import torch

import mani_skill.envs.utils.randomization as randomization
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from ..pickplace_base import BimanualPickPlaceEnv
from ..success_logic import blocks_stacked

_BLOCK_HALF = 0.025  # 5 cm cubes
_BLOCK_COLORS = [
    ([0.8, 0.1, 0.1, 1.0], "red"),
    ([0.1, 0.7, 0.1, 1.0], "green"),
    ([0.85, 0.8, 0.1, 1.0], "yellow"),
]


@register_env("BimanualBlockStack-v1", max_episode_steps=1200)
class BimanualBlockStackEnv(BimanualPickPlaceEnv):
    """Stack three primitive blocks (red, green, yellow)."""

    # anchors only (no ycb id); _build_objects ignores the id slot.
    OBJECTS = [
        ("block_red",    (-0.30, 0.22), "left"),
        ("block_green",  (-0.30, -0.18), "right"),
        ("block_yellow", (-0.24, -0.30), "right"),
    ]
    CONTAINER = None
    INSTRUCTION = "Stack the red, green, and yellow blocks."

    block_half = _BLOCK_HALF
    stack_xy_tol = 0.04

    def _build_objects(self):
        objs = []
        for i, ((_name, _anchor, _arm), (color, cname)) in enumerate(
            zip(self.OBJECTS, _BLOCK_COLORS)
        ):
            builder = self.scene.create_actor_builder()
            mat = sapien.render.RenderMaterial(base_color=color)
            hs = [self.block_half] * 3
            builder.add_box_visual(half_size=hs, material=mat)
            builder.add_box_collision(half_size=hs)
            builder.initial_pose = sapien.Pose()
            objs.append(builder.build(name=f"block_{i}_{cname}"))
        return objs

    def _after_reconfigure(self, options: dict):
        # primitive cubes: half-height is known, skip mesh bbox query.
        self.object_zs = torch.full(
            (len(self.objects),), self.block_half, device=self.device
        )

    def success_fn(self) -> torch.Tensor:
        # positions: (N, b, 3)
        positions = torch.stack([o.pose.p for o in self.objects], dim=0)
        stacked = blocks_stacked(
            positions, block_height=2 * self.block_half, xy_tol=self.stack_xy_tol
        )  # (b,)
        # evaluate() expects (N, b); report all-or-nothing by broadcasting the
        # stacked flag to every object row so n_in_box reflects stack progress
        # only as 0 or N. n_in_box is diagnostic; success is the real metric.
        return stacked.unsqueeze(0).expand(len(self.objects), -1)
