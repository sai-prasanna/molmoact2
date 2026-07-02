# sim_eval/tasks/pickplace_tasks/lift_place.py
"""Truly-bimanual lift-and-place-on-stand task.

One dynamic bar too long (0.30 m) for a single gripper to carry level is lifted
by BOTH arms and placed on a raised static stand. Success = placed on stand;
progress is reported through `n_in_box` as a 0/1/2 stage (0 on table, 1 lifted
level with both arms grasping, 2 placed) so the rollout gets graded signal even
at 0% binary success. See docs/superpowers/specs/2026-07-02-bimanual-lift-place-stand-design.md.
"""

from typing import Any, Dict

import numpy as np
import sapien
import sapien.physx as physx
import torch

from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from ..pickplace_base import BimanualPickPlaceEnv
from ..success_logic import placed_on_stand, lift_stage

_GRAY = [0.6, 0.6, 0.62, 1.0]
_BAR_COLOR = [0.85, 0.55, 0.15, 1.0]


def _build_stand(scene, name, pos, half_x, half_y, top_z, color):
    """Solid static box whose top surface is at z = top_z."""
    builder = scene.create_actor_builder()
    mat = sapien.render.RenderMaterial(base_color=color)
    half = [half_x, half_y, top_z / 2.0]
    builder.add_box_visual(half_size=half, material=mat)
    builder.add_box_collision(half_size=half)
    builder.initial_pose = sapien.Pose(p=[pos[0], pos[1], top_z / 2.0])
    return builder.build_static(name=name)


@register_env("BimanualLiftPlaceStand-v1", max_episode_steps=1200)
class BimanualLiftPlaceStandEnv(BimanualPickPlaceEnv):
    """Lift a long bar with both arms and place it on a raised stand."""

    INSTRUCTION = "Pick up the long bar with both arms and place it on the stand."
    # single "object" entry so base _after_reconfigure/_initialize hooks that
    # iterate self.objects work; arm="left" keeps the y>=0 assertion happy (y=0).
    OBJECTS = [("bar", (-0.32, 0.0), "left")]
    CONTAINER = {"kind": "stand", "pos": (-0.15, 0.0)}

    # --- geometry / thresholds (starting points; tune on GPU smoke) ----------
    BAR_HALF = [0.02, 0.15, 0.02]          # 0.30 m long along y
    BAR_SPAWN = (-0.32, 0.0)
    STAND_POS = (-0.15, 0.0)
    STAND_TOP_Z = 0.12
    STAND_HALF = (0.05, 0.17)
    LIFT_Z = 0.09
    LIFT_TILT_DEG = 30.0
    PLACE_XY_HALF = (0.06, 0.05)
    PLACE_Z = (0.11, 0.17)
    PLACE_TILT_DEG = 25.0
    PLACE_SPEED = 0.05

    # --- scene build --------------------------------------------------------
    def _build_objects(self):
        bar = actors.build_box(
            self.scene, half_sizes=self.BAR_HALF, color=_BAR_COLOR, name="bar",
            body_type="dynamic",
            initial_pose=sapien.Pose(p=[self.BAR_SPAWN[0], self.BAR_SPAWN[1], self.BAR_HALF[2]]),
        )
        for body in bar._bodies:
            for cs in body.get_collision_shapes():
                cs.physical_material = physx.PhysxMaterial(
                    static_friction=self.object_friction,
                    dynamic_friction=self.object_friction,
                    restitution=0.0,
                )
        self.bar = bar
        return [bar]

    def _build_container(self):
        self.stand = _build_stand(
            self.scene, name="stand", pos=self.STAND_POS,
            half_x=self.STAND_HALF[0], half_y=self.STAND_HALF[1],
            top_z=self.STAND_TOP_Z, color=_GRAY,
        )
        return self.stand

    def _after_reconfigure(self, options: dict):
        # primitive bar: half-height is known, skip the mesh bbox query
        # (matches block_stack.py; base path relies on get_first_collision_mesh).
        self.object_zs = torch.full(
            (len(self.objects),), self.BAR_HALF[2], device=self.device
        )

    # --- episode init: single bar, small yaw jitter only --------------------
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            self.agent.robot.set_pose(sapien.Pose(p=self.robot_base_p))
            if self.reset_robot_qpos:
                self.agent.robot.set_qpos(self.agent.keyframes["home"].qpos)

            xyz = torch.zeros((b, 3))
            xyz[:, 0] = self.BAR_SPAWN[0] + (torch.rand((b,)) * 2 - 1) * self.spawn_noise
            xyz[:, 1] = self.BAR_SPAWN[1] + (torch.rand((b,)) * 2 - 1) * self.spawn_noise
            xyz[:, 2] = self.object_zs[0]
            yaw = (torch.rand((b,)) * 2 - 1) * 0.10          # +-0.1 rad
            half = yaw / 2.0
            quat = torch.zeros((b, 4))
            quat[:, 0] = torch.cos(half)
            quat[:, 3] = torch.sin(half)
            self.bar.set_pose(Pose.create_from_pq(xyz, quat))

    # --- obs ---------------------------------------------------------------
    def _get_obs_extra(self, info: dict) -> Dict[str, Any]:
        cx, cy = self.STAND_POS
        stand_center = torch.tensor(
            [cx, cy, self.STAND_TOP_Z], device=self.device
        ).reshape(1, 3).expand(self.num_envs, 3)
        return {
            "bar_pose": self.bar.pose.raw_pose,
            "stand_center": stand_center,
            "left_tcp": self.agent.left_tcp.pose.p,
            "right_tcp": self.agent.right_tcp.pose.p,
        }

    # --- success + graded stage --------------------------------------------
    def evaluate(self) -> Dict[str, Any]:
        p = self.bar.pose.p
        q = self.bar.pose.q
        speed = torch.linalg.norm(self.bar.linear_velocity, dim=1)
        both_grasped = (
            self.agent.is_grasping(self.bar, arm="left")
            & self.agent.is_grasping(self.bar, arm="right")
        )
        cx, cy = self.STAND_POS
        placed = placed_on_stand(
            p, q, speed, cx=cx, cy=cy,
            half_x=self.PLACE_XY_HALF[0], half_y=self.PLACE_XY_HALF[1],
            z_lo=self.PLACE_Z[0], z_hi=self.PLACE_Z[1],
            max_tilt_deg=self.PLACE_TILT_DEG, max_speed=self.PLACE_SPEED,
        )
        stage = lift_stage(
            p, q, both_grasped, placed,
            z_lift=self.LIFT_Z, max_tilt_deg_lift=self.LIFT_TILT_DEG,
        )
        return {
            "success": placed,
            "n_in_box": stage,                         # 0/1/2, maxed over steps by run_eval
            "n_total": torch.full_like(stage, 2),
        }

    def compute_dense_reward(self, obs, action, info):
        return info["n_in_box"].float() / 2.0

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info)
