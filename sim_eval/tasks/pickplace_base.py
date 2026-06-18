# sim_eval/tasks/pickplace_base.py
"""Base bimanual pick-place env for the MolmoAct2 YAM/DK1 task suite.

Generalizes the original PutEverythingInBox env. Subclasses set OBJECTS,
CONTAINER, and INSTRUCTION; they may override _build_objects / _build_container /
success_fn for outlier tasks (e.g. block stacking, which has primitive objects,
no container, and a stacking success criterion).
"""

from typing import Any, Dict, Optional

import numpy as np
import sapien
import torch

import mani_skill.envs.utils.randomization as randomization
import sapien.physx as physx
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose

from .success_logic import object_in_footprint

# Importing the agents registers them with ManiSkill (side-effect).
from ..robots.bimanual_yam import BimanualYAM  # noqa: F401
from ..robots.bimanual_dk1 import BimanualDK1  # noqa: F401


def _build_open_box(scene, name, inner_half, height, wall_thickness,
                    color=None, body_type="static", initial_pose=None):
    """Open-top box: square/rect floor + 4 side walls, no lid.

    inner_half may be a float (square) or (hx, hy) tuple (rectangular).
    """
    if isinstance(inner_half, (tuple, list)):
        ihx, ihy = float(inner_half[0]), float(inner_half[1])
    else:
        ihx = ihy = float(inner_half)
    builder = scene.create_actor_builder()
    wt, h = wall_thickness, height
    outer_x, outer_y = ihx + wt, ihy + wt
    mat = sapien.render.RenderMaterial(
        base_color=color if color is not None else [0.55, 0.35, 0.18, 1.0]
    )
    floor_pose = sapien.Pose(p=[0, 0, wt / 2])
    builder.add_box_visual(pose=floor_pose, half_size=[outer_x, outer_y, wt / 2], material=mat)
    builder.add_box_collision(pose=floor_pose, half_size=[outer_x, outer_y, wt / 2])
    wall_z = wt + h / 2
    for sx in (+1.0, -1.0):
        pose = sapien.Pose(p=[sx * (ihx + wt / 2), 0, wall_z])
        builder.add_box_visual(pose=pose, half_size=[wt / 2, outer_y, h / 2], material=mat)
        builder.add_box_collision(pose=pose, half_size=[wt / 2, outer_y, h / 2])
    for sy in (+1.0, -1.0):
        pose = sapien.Pose(p=[0, sy * (ihy + wt / 2), wall_z])
        builder.add_box_visual(pose=pose, half_size=[ihx, wt / 2, h / 2], material=mat)
        builder.add_box_collision(pose=pose, half_size=[ihx, wt / 2, h / 2])
    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder.build_static(name=name) if body_type == "static" else builder.build(name=name)


def _build_tray(scene, name, inner_half, height, wall_thickness, n_compartments,
                divider, color=None, body_type="static", initial_pose=None):
    """Open box with (n_compartments-1) interior dividers along the y axis.

    Same outer shell as _build_open_box plus thin interior walls that split the
    interior into n_compartments lanes. Dividers are shorter than the outer walls
    so utensils can still be dropped in from above.
    """
    if isinstance(inner_half, (tuple, list)):
        ihx, ihy = float(inner_half[0]), float(inner_half[1])
    else:
        ihx = ihy = float(inner_half)
    builder = scene.create_actor_builder()
    wt, h = wall_thickness, height
    outer_x, outer_y = ihx + wt, ihy + wt
    mat = sapien.render.RenderMaterial(
        base_color=color if color is not None else [0.55, 0.35, 0.18, 1.0]
    )
    floor_pose = sapien.Pose(p=[0, 0, wt / 2])
    builder.add_box_visual(pose=floor_pose, half_size=[outer_x, outer_y, wt / 2], material=mat)
    builder.add_box_collision(pose=floor_pose, half_size=[outer_x, outer_y, wt / 2])
    wall_z = wt + h / 2
    for sx in (+1.0, -1.0):
        pose = sapien.Pose(p=[sx * (ihx + wt / 2), 0, wall_z])
        builder.add_box_visual(pose=pose, half_size=[wt / 2, outer_y, h / 2], material=mat)
        builder.add_box_collision(pose=pose, half_size=[wt / 2, outer_y, h / 2])
    for sy in (+1.0, -1.0):
        pose = sapien.Pose(p=[0, sy * (ihy + wt / 2), wall_z])
        builder.add_box_visual(pose=pose, half_size=[ihx, wt / 2, h / 2], material=mat)
        builder.add_box_collision(pose=pose, half_size=[ihx, wt / 2, h / 2])
    # interior dividers along y (split the y span into n_compartments lanes)
    div_h = h * 0.6
    div_z = wt + div_h / 2
    for k in range(1, n_compartments):
        yk = -ihy + (2 * ihy) * k / n_compartments
        pose = sapien.Pose(p=[0, yk, div_z])
        builder.add_box_visual(pose=pose, half_size=[ihx, divider, div_h / 2], material=mat)
        builder.add_box_collision(pose=pose, half_size=[ihx, divider, div_h / 2])
    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder.build_static(name=name) if body_type == "static" else builder.build(name=name)


class BimanualPickPlaceEnv(BaseEnv):
    """Parameterized bimanual pick-place env.

    Subclasses set OBJECTS / CONTAINER / INSTRUCTION. Default success = every
    object's center inside the container XY footprint and below rim + z-margin.
    Registered for both bimanual_yam and bimanual_dk1; the robot is chosen at
    launch (run_eval derives robot_uids from the policy type).
    """

    SUPPORTED_ROBOTS = ["bimanual_yam", "bimanual_dk1"]

    # --- per-task overrides -------------------------------------------------
    # OBJECTS: list of (ycb_id, (anchor_x, anchor_y), arm), arm in {"left","right"}
    OBJECTS: list = []
    # CONTAINER: dict (see schema) or None (no container, e.g. block stacking)
    CONTAINER: Optional[dict] = None
    INSTRUCTION: str = ""

    robot_base_p = [-0.65, 0.0, 0.01]
    spawn_noise = 0.02
    object_friction = 1.0
    # z-margin above the rim within which a piled object still counts as "in"
    success_z_margin = 0.05

    def __init__(
        self,
        *args,
        robot_uids="bimanual_yam",
        robot_init_qpos_noise=0.02,
        reset_robot_qpos=True,
        reconfiguration_freq=1,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.reset_robot_qpos = reset_robot_qpos
        super().__init__(
            *args,
            robot_uids=robot_uids,
            reconfiguration_freq=reconfiguration_freq,
            **kwargs,
        )

    # --- cameras (base render only; policy cams live on the agent) -----------
    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[-0.65, 0.0, 0.5], target=[-0.4, 0.0, 0.1])
        return [CameraConfig("base_camera", pose, 640, 480, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0.2, 0.2, 0.5], target=[-0.4, 0.0, 0.05])
        return CameraConfig("render_camera", pose=pose, width=1280, height=720,
                            fov=1.2, near=0.01, far=100, shader_pack="rt")

    # --- scene build (overridable pieces) -----------------------------------
    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.objects = self._build_objects()
        self.container = self._build_container()

    def _build_objects(self):
        """Build the YCB objects listed in OBJECTS. Override for primitives."""
        objs = []
        for i, (ycb_id, _anchor, _arm) in enumerate(self.OBJECTS):
            builder = actors.get_actor_builder(self.scene, id=f"ycb:{ycb_id}")
            builder.initial_pose = sapien.Pose()
            obj = builder.build(name=f"obj_{i}_{ycb_id}")
            for body in obj._bodies:
                for cs in body.get_collision_shapes():
                    cs.physical_material = physx.PhysxMaterial(
                        static_friction=self.object_friction,
                        dynamic_friction=self.object_friction,
                        restitution=0.0,
                    )
            objs.append(obj)
        return objs

    def _build_container(self):
        """Build CONTAINER. Returns None when CONTAINER is None."""
        c = self.CONTAINER
        if c is None:
            return None
        cx, cy = c["pos"]
        common_kw = dict(
            inner_half=c["inner_half"],
            height=c["height"],
            wall_thickness=c["wall"],
            color=c.get("color", [0.55, 0.35, 0.18, 1.0]),
            body_type="static",
            initial_pose=sapien.Pose(p=[cx, cy, 0.0]),
        )
        if c["kind"] == "tray":
            return _build_tray(
                self.scene, name="tray",
                n_compartments=c["n_compartments"], divider=c["divider"],
                **common_kw,
            )
        return _build_open_box(self.scene, name="open_box", **common_kw)

    def _after_reconfigure(self, options: dict):
        object_zs = []
        for obj in self.objects:
            collision_mesh = obj.get_first_collision_mesh()
            object_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.object_zs = common.to_tensor(object_zs, device=self.device)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            self.agent.robot.set_pose(sapien.Pose(p=self.robot_base_p))
            if self.reset_robot_qpos:
                self.agent.robot.set_qpos(self.agent.keyframes["home"].qpos)

            for i, (_ycb, (ax, ay), arm) in enumerate(self.OBJECTS):
                # validated placement assertion (guards anchor sign typos)
                if arm == "left":
                    assert ay >= 0, f"OBJECTS[{i}] arm=left but anchor_y={ay} < 0"
                elif arm == "right":
                    assert ay <= 0, f"OBJECTS[{i}] arm=right but anchor_y={ay} > 0"
                xyz = torch.zeros((b, 3))
                xyz[:, 0] = ax + (torch.rand((b,)) * 2 - 1) * self.spawn_noise
                xyz[:, 1] = ay + (torch.rand((b,)) * 2 - 1) * self.spawn_noise
                xyz[:, 2] = self.object_zs[i]
                qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
                self.objects[i].set_pose(Pose.create_from_pq(xyz, qs))

    def _get_obs_extra(self, info: dict) -> Dict[str, Any]:
        obs = {}
        if self.CONTAINER is not None:
            cx, cy = self.CONTAINER["pos"]
            obs["container_center"] = torch.tensor(
                [cx, cy, self.CONTAINER["wall"]], device=self.device
            ).reshape(1, 3).expand(self.num_envs, 3)
        for i, (ycb_id, _a, _arm) in enumerate(self.OBJECTS):
            obs[f"obj_{i}_pose"] = self.objects[i].pose.raw_pose
        return obs

    # --- success (overridable) ----------------------------------------------
    def _container_bounds(self):
        c = self.CONTAINER
        cx, cy = c["pos"]
        ih = c["inner_half"]
        hx, hy = (ih, ih) if not isinstance(ih, (tuple, list)) else (ih[0], ih[1])
        z_lo = c["wall"] - 0.01
        z_hi = c["wall"] + c["height"] + self.success_z_margin
        return cx, cy, hx, hy, z_lo, z_hi

    def success_fn(self) -> torch.Tensor:
        """Default: every object's center inside the container footprint + z band."""
        cx, cy, hx, hy, z_lo, z_hi = self._container_bounds()
        per_obj = [object_in_footprint(o.pose.p, cx, cy, hx, hy, z_lo, z_hi)
                   for o in self.objects]
        in_box = torch.stack(per_obj, dim=0)        # (N, b)
        return in_box

    def evaluate(self) -> Dict[str, Any]:
        in_box = self.success_fn()                  # (N, b)
        n_in_box = in_box.float().sum(dim=0)
        return {
            "success": in_box.all(dim=0),
            "n_in_box": n_in_box,
            "n_total": torch.full_like(n_in_box, len(self.objects)),
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict) -> torch.Tensor:
        reward = info["n_in_box"] / float(len(self.objects)) * 5.0
        reward[info["success"]] = 5.0
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict) -> torch.Tensor:
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5.0
