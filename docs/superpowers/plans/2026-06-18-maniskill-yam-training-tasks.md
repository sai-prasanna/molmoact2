# ManiSkill YAM/DK1 Training-Task Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 8 ManiSkill bimanual pick-place task environments (drawn from the MolmoAct2-BimanualYAM training mixture) so the YAM policy and its DK1 transfer can be rolled out zero-shot on more task variety, reusing the existing rollout/metrics machinery.

**Architecture:** Introduce a parameterized base env `BimanualPickPlaceEnv` (`sim_eval/tasks/pickplace_base.py`) that generalizes the existing `PutEverythingInBox` env. Object-building, container-building, and success-checking are overridable so the outlier "block stack" task (primitive blocks, no container, stacking success) fits the same base. Success math is extracted into pure tensor functions (`sim_eval/tasks/success_logic.py`) so it is unit-testable on CPU without a GPU/Vulkan env. Eight thin subclasses register one env id each, supporting both `bimanual_yam` and `bimanual_dk1`; `run_eval.py` derives `robot_uids` from `--policy-type` so a single env id serves whichever policy is under test.

**Tech Stack:** Python 3.11, `uv`, ManiSkill (SAPIEN/PhysX/Vulkan), gymnasium, torch, numpy, pytest (new, for pure-function tests only), imageio.

## Global Constraints

- All Python is run via `uv run` (bare `python` is not on PATH). Tests: `uv run pytest`.
- Python pinned to 3.11; do not change torch/transformers/maniskill pins.
- YCB object ids MUST be ones present in the downloaded set at
  `sim_eval/.maniskill_data/data/assets/mani_skill2_ycb/models/` (verified list below). Never hardcode an id not in that directory.
- Instruction strings name the **rendered YCB objects** (generic names like "the can"), NOT the real-world brand names from the training annotations. This is an OOD-grounding requirement (see spec design decisions).
- Object actor names MUST be unique within a single env (ManiSkill requires it). Name by index: `f"obj_{i}_{ycb_id}"`.
- `left = +y`, `right = −y` in the table frame. The robot base default pose is `robot_base_p = [-0.65, 0.0, 0.01]`.
- The existing env ids `BimanualYAMPutEverythingInBox-v1` and `DK1PutEverythingInBox-v1` MUST keep working (kept as registered aliases) — existing sbatch/scripts depend on them.
- Camera configs (`top_cam`, `left_cam`, `right_cam`) live on the **agent**, not the task. Tasks only add a `base_camera`. Do not add the three policy cameras in the task.
- `DK1_RECOLOR` is an env-var lever on the DK1 agent; it carries forward unchanged. Do not touch it.
- No new robots, agents, inference clients, or wire-protocol changes.

### Verified YCB ids available (subset relevant to this plan)

```
002_master_chef_can  003_cracker_box  004_sugar_box  005_tomato_soup_can
008_pudding_box  009_gelatin_box  030_fork  031_spoon  032_knife
037_scissors  042_adjustable_wrench  043_phillips_screwdriver  050_medium_clamp
056_tennis_ball  065-a_cups  065-b_cups  072-a_toy_airplane  073-a_lego_duplo
077_rubiks_cube
```

Note: there is **no** "chips/pringles can" in YCB. `002_master_chef_can` (tall cylindrical can) is the substitute for the "box" family's "chips can", named "the can" in the instruction.

### Empirical risks (NOT unit-testable — verify on GPU smoke, values below are starting points to TUNE)

- **Container sizing for large objects.** `003_cracker_box` is ~16×21 cm; it will NOT fit a 0.09 inner-half box. The `box`/`gro`/`tool` families need larger containers (inner_half ≥ 0.13). Even then, fitting several large boxes is a genuine fidelity risk that can fail rollouts for non-policy reasons. Container sizes are tagged "tune on GPU smoke".
- **Anchor reachability + non-overlap** for 3–5 objects across two arms. Starting anchors are given per task; tune so objects spawn within each arm's reach and don't overlap.
- These risks are real failure surface. The build smoke test confirms objects *spawn on the table within the footprint*; it does NOT prove the task is achievable. Report rollout numbers paired/relative only (OOD).

### `arm` field semantics

In `OBJECTS = [(ycb_id, anchor_xy, arm), ...]`, `arm ∈ {"left","right"}` is a **validated placement assertion**: `_initialize_episode` asserts `left ⇒ anchor_y ≥ 0` and `right ⇒ anchor_y ≤ 0`. It documents intended per-arm reachability and guards against a sign typo in anchors. The policy itself decides which arm acts.

---

## File Structure

- `sim_eval/tasks/success_logic.py` — **new.** Pure functions: `object_in_footprint(...)`, `blocks_stacked(...)`. No ManiSkill imports. CPU/torch only.
- `tests/test_success_logic.py` — **new.** pytest unit tests for the above. The only genuine TDD seam.
- `sim_eval/tasks/pickplace_base.py` — **new.** `BimanualPickPlaceEnv(BaseEnv)` + `_build_open_box` + `_build_tray`. Overridable `_build_objects`, `_build_container`, `success_fn`.
- `sim_eval/tasks/yam_tasks/bimanual_put_everything_in_box.py` — **modify.** Refactor to subclass base; keep id.
- `sim_eval/tasks/dk1_tasks/dk1_put_everything_in_box.py` — **modify.** Refactor to subclass base; keep id.
- `sim_eval/tasks/pickplace_tasks/__init__.py` — **new.** Imports the 8 new task modules.
- `sim_eval/tasks/pickplace_tasks/container_tasks.py` — **new.** 6 container subclasses (box, toy, cup, med, tool, gro).
- `sim_eval/tasks/pickplace_tasks/block_stack.py` — **new.** Block-stack subclass (primitive objects, no container, stacking success).
- `sim_eval/tasks/pickplace_tasks/utensil_tray.py` — **new.** Utensil-tray subclass (`_build_tray`).
- `sim_eval/tasks/__init__.py` — **modify.** Import `pickplace_tasks`.
- `sim_eval/run_eval.py` — **modify.** Derive `robot_uids` from `--policy-type`; add `--robot-uids`; fall back to env `INSTRUCTION` attr.
- `sim_eval/scripts/render_smoke.py` — **modify.** Add a `--env` flag so the smoke harness can render any of the new envs.

---

## Task 1: Pure success-logic functions + CPU unit tests

This is the real TDD seam — pure tensor math, runs on CPU, no GPU/env needed.

**Files:**
- Create: `sim_eval/tasks/success_logic.py`
- Test: `tests/test_success_logic.py`

**Interfaces:**
- Produces:
  - `object_in_footprint(p, cx, cy, half_x, half_y, z_lo, z_hi) -> torch.BoolTensor`
    where `p` is `(b, 3)` object center positions; `cx,cy` container center XY (floats);
    `half_x,half_y` container interior half-extents (floats); `z_lo,z_hi` allowed z band
    (floats). Returns `(b,)` bool: center inside XY footprint AND z in `[z_lo, z_hi]`.
  - `blocks_stacked(positions, block_height, xy_tol, gap_lo_frac=0.5, gap_hi_frac=1.5) -> torch.BoolTensor`
    where `positions` is `(N, b, 3)` for N blocks. Returns `(b,)` bool: per env, the N
    blocks form a single stack — sorted by z, consecutive z-gaps lie in
    `[gap_lo_frac*block_height, gap_hi_frac*block_height]` and all blocks' XY lie within
    `xy_tol` of the stack's mean XY.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_success_logic.py
import torch

from sim_eval.tasks.success_logic import object_in_footprint, blocks_stacked


def test_object_inside_footprint_is_true():
    p = torch.tensor([[-0.15, 0.0, 0.03]])  # at container center, low z
    inside = object_in_footprint(p, cx=-0.15, cy=0.0, half_x=0.1, half_y=0.1, z_lo=0.0, z_hi=0.1)
    assert bool(inside[0]) is True


def test_object_outside_xy_is_false():
    p = torch.tensor([[-0.15, 0.5, 0.03]])  # far in +y, outside footprint
    inside = object_in_footprint(p, cx=-0.15, cy=0.0, half_x=0.1, half_y=0.1, z_lo=0.0, z_hi=0.1)
    assert bool(inside[0]) is False


def test_object_above_rim_is_false():
    p = torch.tensor([[-0.15, 0.0, 0.5]])  # inside XY but flying above z band
    inside = object_in_footprint(p, cx=-0.15, cy=0.0, half_x=0.1, half_y=0.1, z_lo=0.0, z_hi=0.1)
    assert bool(inside[0]) is False


def test_object_in_footprint_batched():
    p = torch.tensor([[-0.15, 0.0, 0.03], [-0.15, 0.5, 0.03]])
    inside = object_in_footprint(p, cx=-0.15, cy=0.0, half_x=0.1, half_y=0.1, z_lo=0.0, z_hi=0.1)
    assert inside.tolist() == [True, False]


def test_blocks_stacked_true():
    # three 5cm blocks stacked at same xy, z = 0.025, 0.075, 0.125
    positions = torch.tensor([
        [[-0.15, 0.0, 0.025]],
        [[-0.15, 0.0, 0.075]],
        [[-0.15, 0.0, 0.125]],
    ])  # (N=3, b=1, 3)
    ok = blocks_stacked(positions, block_height=0.05, xy_tol=0.03)
    assert bool(ok[0]) is True


def test_blocks_not_stacked_when_spread_in_xy():
    positions = torch.tensor([
        [[-0.15, 0.0, 0.025]],
        [[-0.15, 0.20, 0.075]],   # second block far away in y
        [[-0.15, 0.0, 0.125]],
    ])
    ok = blocks_stacked(positions, block_height=0.05, xy_tol=0.03)
    assert bool(ok[0]) is False


def test_blocks_not_stacked_when_all_on_table():
    positions = torch.tensor([
        [[-0.15, 0.00, 0.025]],
        [[-0.15, 0.10, 0.025]],   # all at same z, not stacked
        [[-0.15, 0.20, 0.025]],
    ])
    ok = blocks_stacked(positions, block_height=0.05, xy_tol=0.03)
    assert bool(ok[0]) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_success_logic.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sim_eval.tasks.success_logic'`

- [ ] **Step 3: Write the implementation**

```python
# sim_eval/tasks/success_logic.py
"""Pure tensor success predicates for the bimanual pick-place tasks.

No ManiSkill / SAPIEN imports — these are unit-testable on CPU. Env code in
pickplace_base.py calls them from evaluate().
"""

import torch


def object_in_footprint(
    p: torch.Tensor,
    cx: float,
    cy: float,
    half_x: float,
    half_y: float,
    z_lo: float,
    z_hi: float,
) -> torch.Tensor:
    """True where an object's center is inside the container XY footprint and z band.

    p: (b, 3) object center positions. Returns (b,) bool.
    Piling-tolerant: tests the center only, with a z band that extends above the
    rim, so objects that stack/shove inside still count.
    """
    dx = torch.abs(p[:, 0] - cx)
    dy = torch.abs(p[:, 1] - cy)
    inside_xy = (dx < half_x) & (dy < half_y)
    z_ok = (p[:, 2] > z_lo) & (p[:, 2] < z_hi)
    return inside_xy & z_ok


def blocks_stacked(
    positions: torch.Tensor,
    block_height: float,
    xy_tol: float,
    gap_lo_frac: float = 0.5,
    gap_hi_frac: float = 1.5,
) -> torch.Tensor:
    """True per env where N blocks form a single vertical stack.

    positions: (N, b, 3). Returns (b,).
    Criterion: sort blocks by z; every consecutive z-gap lies in
    [gap_lo_frac, gap_hi_frac] * block_height, and all blocks' XY lie within
    xy_tol of the stack mean XY.
    """
    n, b, _ = positions.shape
    z = positions[..., 2]                       # (N, b)
    z_sorted, _ = torch.sort(z, dim=0)          # (N, b)
    gaps = z_sorted[1:] - z_sorted[:-1]         # (N-1, b)
    gap_lo = gap_lo_frac * block_height
    gap_hi = gap_hi_frac * block_height
    gaps_ok = ((gaps > gap_lo) & (gaps < gap_hi)).all(dim=0)   # (b,)

    xy = positions[..., :2]                      # (N, b, 2)
    mean_xy = xy.mean(dim=0, keepdim=True)       # (1, b, 2)
    xy_dev = torch.linalg.norm(xy - mean_xy, dim=-1)  # (N, b)
    xy_ok = (xy_dev < xy_tol).all(dim=0)         # (b,)

    return gaps_ok & xy_ok
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_success_logic.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add sim_eval/tasks/success_logic.py tests/test_success_logic.py
git commit -m "feat(sim_eval): add CPU-testable pick-place success predicates"
```

---

## Task 2: Base env `BimanualPickPlaceEnv`

Generalizes the existing PutEverythingInBox env. Object/container building and
success are overridable so the block task (no YCB ids, no container, stacking
success) fits the same base. **No standalone test** — env build requires GPU/Vulkan;
it is verified by porting the known-good task in Task 3 and smoke-rendering it.

**Files:**
- Create: `sim_eval/tasks/pickplace_base.py`

**Interfaces:**
- Consumes: `object_in_footprint` from Task 1; `BimanualYAM`/`BimanualDK1` agents (imported for side-effect registration).
- Produces (relied on by Tasks 3, 5, 6, 7):
  - class `BimanualPickPlaceEnv(BaseEnv)` with overridable class attrs
    `OBJECTS`, `CONTAINER` (dict or `None`), `INSTRUCTION` (str),
    `robot_base_p`, `spawn_noise`, `object_friction`.
  - methods (override points): `_build_objects(self)`, `_build_container(self)`,
    `success_fn(self) -> torch.BoolTensor (b,)`.
  - module helpers `_build_open_box(...)`, `_build_tray(...)`.
  - `CONTAINER` dict schema: `{"kind": "box"|"tray", "inner_half": (hx, hy) | float,
    "pos": (x, y), "height": float, "wall": float, "color": [r,g,b,a]}`. For `tray`,
    also `"n_compartments": int` and `"divider": float` (divider half-thickness).

- [ ] **Step 1: Write the base module**

```python
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
```

- [ ] **Step 2: Verify the module imports cleanly (no GPU needed for import)**

Run: `uv run python -c "import sim_eval.tasks.pickplace_base as m; print(m.BimanualPickPlaceEnv, m._build_open_box, m._build_tray)"`
Expected: prints the class and two helper functions, no error.

- [ ] **Step 3: Commit**

```bash
git add sim_eval/tasks/pickplace_base.py
git commit -m "feat(sim_eval): add BimanualPickPlaceEnv base with overridable objects/container/success"
```

---

## Task 3: Refactor existing PutEverythingInBox onto the base (keep alias ids) + smoke-verify

Proves the abstraction on the known-good task before it's copied 8×.

**Files:**
- Modify: `sim_eval/tasks/yam_tasks/bimanual_put_everything_in_box.py`
- Modify: `sim_eval/tasks/dk1_tasks/dk1_put_everything_in_box.py`
- Modify: `sim_eval/tasks/yam_tasks/__init__.py` (no symbol change needed, but confirm export name)
- Modify: `sim_eval/scripts/render_smoke.py` (add `--env` flag)

**Interfaces:**
- Consumes: `BimanualPickPlaceEnv` from Task 2.
- Produces: env ids `BimanualYAMPutEverythingInBox-v1`, `DK1PutEverythingInBox-v1` (unchanged), now backed by the base. The original objects (lego duplo left, tennis ball right) and box geometry are preserved so behavior matches.

- [ ] **Step 1: Replace the YAM PutEverythingInBox module with a base subclass**

```python
# sim_eval/tasks/yam_tasks/bimanual_put_everything_in_box.py
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
```

- [ ] **Step 2: Replace the DK1 PutEverythingInBox module with a base subclass**

```python
# sim_eval/tasks/dk1_tasks/dk1_put_everything_in_box.py
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
```

- [ ] **Step 3: Confirm the yam_tasks/dk1_tasks `__init__.py` still export the class names**

`sim_eval/tasks/yam_tasks/__init__.py` should remain:

```python
"""YAM bimanual tasks. Importing each module registers its env with ManiSkill."""

from .bimanual_put_everything_in_box import BimanualYAMPutEverythingInBoxEnv

__all__ = ["BimanualYAMPutEverythingInBoxEnv"]
```

Check `sim_eval/tasks/dk1_tasks/__init__.py` similarly imports `DK1PutEverythingInBoxEnv`. If the import name differs from what the file defines, fix it to match (`DK1PutEverythingInBoxEnv`).

- [ ] **Step 4: Add a `--env` flag to render_smoke.py**

In `sim_eval/scripts/render_smoke.py`, change the argparse + Stage 2 so any env id can be rendered. Replace the `ap.add_argument("--shader", ...)` block and Stage 2 env id:

```python
    ap.add_argument("--shader", default="rt-fast")
    ap.add_argument("--env", default="BimanualYAMPutEverythingInBox-v1",
                    help="Env id to render in stage 2")
    ap.add_argument("--out", default=str(Path(__file__).parent.parent / "outputs" / "render_smoke"))
```

And in Stage 2, use `args.env` in the `gym.make` call and in the saved filenames:

```python
    print(f"=== Stage 2: {args.env}, shader={args.shader} ===", flush=True)
    import sim_eval.tasks  # noqa: F401  (registers env + robot via side-effect)
    env = gym.make(args.env, obs_mode="rgb",
                   control_mode="pd_joint_pos", render_mode="rgb_array",
                   sensor_configs=dict(shader_pack=args.shader))
    obs, _ = env.reset(seed=0)
    cams = list((obs.get("sensor_data") or {}).keys())
    print(f"  sensor cameras: {cams}", flush=True)
    frame = _to_img(env.render())
    safe = args.env.replace('/', '_')
    imageio.imwrite(out / f"stage2_{safe}_{args.shader}.png", frame)
    print(f"  OK render -> {out/f'stage2_{safe}_{args.shader}.png'}  shape={frame.shape}", flush=True)
    for cam, sd in (obs.get("sensor_data") or {}).items():
        img = sd.get("rgb") if isinstance(sd, dict) else sd
        imageio.imwrite(out / f"stage2_{safe}_sensor_{cam}.png", _to_img(img))
        print(f"  sensor {cam} -> saved", flush=True)
    env.close()
    print("ALL OK", flush=True)
```

- [ ] **Step 5: GPU-gated smoke verification (run on a GPU node)**

Run: `uv run python -m sim_eval.scripts.render_smoke --env BimanualYAMPutEverythingInBox-v1`
Expected output contains:
- `sensor cameras: ['top_cam', 'left_cam', 'right_cam']` (the agent cameras)
- `ALL OK`
- A render PNG with the lego duplo (left, +y), tennis ball (right, −y), and the open box centered — matching the pre-refactor scene.

Also verify the DK1 id and action space:

Run:
```bash
uv run python -c "
import gymnasium as gym, sim_eval.tasks
for eid in ['BimanualYAMPutEverythingInBox-v1','DK1PutEverythingInBox-v1']:
    e=gym.make(eid, obs_mode='rgb', control_mode='pd_joint_pos'); e.reset(seed=0)
    print(eid, e.action_space.shape)
    e.close()
"
```
Expected: each prints action space `(14,)` (12 arm joints? confirm shape; YAM has 6+6 arm + 2 mimic grippers → 14 action dims). If shape differs from the pre-refactor env, the refactor changed behavior — STOP and diff against git HEAD before continuing.

> If no GPU is available in this session, mark Step 5 as **deferred to GPU node** and proceed — but do not claim the refactor verified until it runs.

- [ ] **Step 6: Commit**

```bash
git add sim_eval/tasks/yam_tasks/bimanual_put_everything_in_box.py \
        sim_eval/tasks/dk1_tasks/dk1_put_everything_in_box.py \
        sim_eval/tasks/dk1_tasks/__init__.py \
        sim_eval/scripts/render_smoke.py
git commit -m "refactor(sim_eval): port PutEverythingInBox onto BimanualPickPlaceEnv base"
```

---

## Task 4: `run_eval.py` robot selection + instruction fallback

Collapses the per-robot env duplication: one env id, robot derived from policy.

**Files:**
- Modify: `sim_eval/run_eval.py`

**Interfaces:**
- Consumes: env `INSTRUCTION` class attr (Tasks 2/3/5/6/7).
- Produces: CLI behavior — `--policy-type remote-yam` → `robot_uids="bimanual_yam"`, `remote-dk1` → `bimanual_dk1`, `remote-droid` → leave env default. New `--robot-uids` override. Instruction resolves `--language-instruction` > `DEFAULT_LANGUAGE_INSTRUCTIONS[env_id]` > env `INSTRUCTION` attr > `env_id`.

- [ ] **Step 1: Add the policy→robot map and `--robot-uids` field**

After the `PolicyType = Literal[...]` line, add:

```python
POLICY_ROBOT: dict[str, str] = {
    "remote-yam": "bimanual_yam",
    "remote-dk1": "bimanual_dk1",
    # remote-droid intentionally absent: DROID tasks are single-robot, use env default
}
```

In `EvalConfig`, after the `policy_type` field add:

```python
    robot_uids: Optional[str] = None
    """Override the robot. Default: derived from --policy-type (yam/dk1)."""
```

- [ ] **Step 2: Pass `robot_uids` into `gym.make` and resolve instruction from the env attr**

In `_evaluate_task`, replace the instruction line + `gym.make` block. Current:

```python
    instruction = config.language_instruction or DEFAULT_LANGUAGE_INSTRUCTIONS.get(env_id, env_id)
    logger.info("─" * 50)
    logger.info("Task: %s  |  instruction: %s", env_id, instruction)

    env = gym.make(
        env_id,
        obs_mode="rgb",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        max_episode_steps=config.max_episode_steps,
        reward_mode="none",
        sensor_configs=dict(shader_pack=config.shader_pack),
        sim_config=dict(sim_freq=config.sim_freq, control_freq=config.control_freq),
    )
```

Replace with:

```python
    robot_uids = config.robot_uids or POLICY_ROBOT.get(config.policy_type)
    make_kwargs = dict(
        obs_mode="rgb",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        max_episode_steps=config.max_episode_steps,
        reward_mode="none",
        sensor_configs=dict(shader_pack=config.shader_pack),
        sim_config=dict(sim_freq=config.sim_freq, control_freq=config.control_freq),
    )
    if robot_uids is not None:
        make_kwargs["robot_uids"] = robot_uids
    env = gym.make(env_id, **make_kwargs)

    instruction = (
        config.language_instruction
        or DEFAULT_LANGUAGE_INSTRUCTIONS.get(env_id)
        or getattr(env.unwrapped, "INSTRUCTION", None)
        or env_id
    )
    logger.info("─" * 50)
    logger.info("Task: %s  |  robot: %s  |  instruction: %s",
                env_id, robot_uids or "(env default)", instruction)
```

- [ ] **Step 3: Verify the CLI parses and the map is wired (no GPU needed)**

Run:
```bash
uv run python -c "
from sim_eval.run_eval import POLICY_ROBOT, EvalConfig
assert POLICY_ROBOT['remote-yam']=='bimanual_yam'
assert POLICY_ROBOT['remote-dk1']=='bimanual_dk1'
assert 'remote-droid' not in POLICY_ROBOT
print('POLICY_ROBOT ok')
"
```
Expected: `POLICY_ROBOT ok`

- [ ] **Step 4: Commit**

```bash
git add sim_eval/run_eval.py
git commit -m "feat(sim_eval): derive robot_uids from policy type; resolve instruction from env attr"
```

---

## Task 5: Six container pick-place tasks

**Files:**
- Create: `sim_eval/tasks/pickplace_tasks/__init__.py`
- Create: `sim_eval/tasks/pickplace_tasks/container_tasks.py`
- Modify: `sim_eval/tasks/__init__.py`

**Interfaces:**
- Consumes: `BimanualPickPlaceEnv` (Task 2).
- Produces: env ids `BimanualBox-v1`, `BimanualToy-v1`, `BimanualCup-v1`, `BimanualMed-v1`, `BimanualTool-v1`, `BimanualGrocery-v1`. Each default robot `bimanual_yam`, both robots supported.

> Anchor and container values below are **starting points — tune on GPU smoke** (reachability, non-overlap, large-object fit). The cracker box is large; box/gro/tool use bigger containers.

- [ ] **Step 1: Write the six container subclasses**

```python
# sim_eval/tasks/pickplace_tasks/container_tasks.py
"""Container pick-place tasks: place a set of YCB objects into one container.

Each is a thin subclass of BimanualPickPlaceEnv. Anchor/container values are
starting points tuned on GPU smoke (object reachability, non-overlap, and
large-object fit — the cracker box in particular is large).
"""

from mani_skill.utils.registration import register_env

from ..pickplace_base import BimanualPickPlaceEnv

# Reusable container presets (tune on GPU smoke).
_BLACK = [0.05, 0.05, 0.05, 1.0]
_BROWN = [0.55, 0.35, 0.18, 1.0]
_GRAY = [0.6, 0.6, 0.62, 1.0]


@register_env("BimanualBox-v1", max_episode_steps=1200)
class BimanualBoxEnv(BimanualPickPlaceEnv):
    """Place a can and three boxes into the black box."""
    OBJECTS = [
        ("002_master_chef_can", (-0.34, 0.20), "left"),
        ("004_sugar_box",       (-0.24, 0.28), "left"),
        ("003_cracker_box",     (-0.34, -0.20), "right"),
        ("008_pudding_box",     (-0.24, -0.28), "right"),
    ]
    CONTAINER = {"kind": "box", "inner_half": 0.15, "pos": (-0.12, 0.0),
                 "height": 0.07, "wall": 0.01, "color": _BLACK}
    INSTRUCTION = "Place the can, cracker box, sugar box, and pudding box into the black box."


@register_env("BimanualToy-v1", max_episode_steps=1200)
class BimanualToyEnv(BimanualPickPlaceEnv):
    """Place three toys into the box."""
    OBJECTS = [
        ("077_rubiks_cube",   (-0.30, 0.22), "left"),
        ("073-a_lego_duplo",  (-0.30, -0.18), "right"),
        ("072-a_toy_airplane",(-0.24, -0.30), "right"),
    ]
    CONTAINER = {"kind": "box", "inner_half": 0.12, "pos": (-0.13, 0.0),
                 "height": 0.07, "wall": 0.01, "color": _BROWN}
    INSTRUCTION = "Place the Rubik's cube, the lego block, and the toy airplane into the box."


@register_env("BimanualCup-v1", max_episode_steps=1200)
class BimanualCupEnv(BimanualPickPlaceEnv):
    """Place two cups, then a fork and spoon, into the container."""
    OBJECTS = [
        ("065-a_cups", (-0.32, 0.20), "left"),
        ("065-b_cups", (-0.24, 0.28), "left"),
        ("030_fork",   (-0.32, -0.20), "right"),
        ("031_spoon",  (-0.24, -0.28), "right"),
    ]
    CONTAINER = {"kind": "box", "inner_half": 0.12, "pos": (-0.13, 0.0),
                 "height": 0.06, "wall": 0.01, "color": _GRAY}
    INSTRUCTION = "Place the cups, then the fork and spoon, into the container."


@register_env("BimanualMed-v1", max_episode_steps=1200)
class BimanualMedEnv(BimanualPickPlaceEnv):
    """Place the medicine boxes into the container."""
    OBJECTS = [
        ("009_gelatin_box", (-0.30, 0.22), "left"),
        ("008_pudding_box", (-0.30, -0.22), "right"),
    ]
    CONTAINER = {"kind": "box", "inner_half": 0.11, "pos": (-0.13, 0.0),
                 "height": 0.06, "wall": 0.01, "color": _GRAY}
    INSTRUCTION = "Place the medicine boxes into the container."


@register_env("BimanualTool-v1", max_episode_steps=1200)
class BimanualToolEnv(BimanualPickPlaceEnv):
    """Place the screwdriver, wrench, clamp, and scissors into the black box."""
    OBJECTS = [
        ("043_phillips_screwdriver", (-0.34, 0.20), "left"),
        ("050_medium_clamp",         (-0.24, 0.28), "left"),
        ("042_adjustable_wrench",    (-0.34, -0.20), "right"),
        ("037_scissors",             (-0.24, -0.28), "right"),
    ]
    CONTAINER = {"kind": "box", "inner_half": 0.14, "pos": (-0.12, 0.0),
                 "height": 0.06, "wall": 0.01, "color": _BLACK}
    INSTRUCTION = "Place the screwdriver, wrench, clamp, and scissors into the black box."


@register_env("BimanualGrocery-v1", max_episode_steps=1200)
class BimanualGroceryEnv(BimanualPickPlaceEnv):
    """Place the cracker box, soup can, and sugar box into the basket."""
    OBJECTS = [
        ("003_cracker_box",     (-0.32, 0.22), "left"),
        ("005_tomato_soup_can", (-0.32, -0.18), "right"),
        ("004_sugar_box",       (-0.24, -0.30), "right"),
    ]
    CONTAINER = {"kind": "box", "inner_half": 0.15, "pos": (-0.12, 0.0),
                 "height": 0.07, "wall": 0.01, "color": _BROWN}
    INSTRUCTION = "Place the cracker box, soup can, and sugar box into the basket."
```

- [ ] **Step 2: Create the package `__init__.py`**

```python
# sim_eval/tasks/pickplace_tasks/__init__.py
"""MolmoAct2 pick-place task suite. Importing each module registers its env."""

from .container_tasks import (
    BimanualBoxEnv,
    BimanualToyEnv,
    BimanualCupEnv,
    BimanualMedEnv,
    BimanualToolEnv,
    BimanualGroceryEnv,
)

__all__ = [
    "BimanualBoxEnv",
    "BimanualToyEnv",
    "BimanualCupEnv",
    "BimanualMedEnv",
    "BimanualToolEnv",
    "BimanualGroceryEnv",
]
```

- [ ] **Step 3: Register the package in `sim_eval/tasks/__init__.py`**

```python
"""MolmoAct2 ManiSkill tasks. Importing this package registers every env."""

from .yam_tasks import *  # noqa: F401, F403
from .droid_tasks import *  # noqa: F401, F403
from .dk1_tasks import *  # noqa: F401, F403
from .pickplace_tasks import *  # noqa: F401, F403
```

- [ ] **Step 4: Verify all six register and resolve their YCB ids (import-time check; full build needs GPU)**

Run:
```bash
uv run python -c "
import gymnasium as gym
import sim_eval.tasks  # registers
from mani_skill.utils.registration import REGISTERED_ENVS
for eid in ['BimanualBox-v1','BimanualToy-v1','BimanualCup-v1','BimanualMed-v1','BimanualTool-v1','BimanualGrocery-v1']:
    assert eid in REGISTERED_ENVS, eid
print('6 container envs registered')
"
```
Expected: `6 container envs registered`

Also assert every YCB id used resolves to a builder path:
```bash
uv run python -c "
from pathlib import Path
root = Path('sim_eval/.maniskill_data/data/assets/mani_skill2_ycb/models')
from sim_eval.tasks.pickplace_tasks.container_tasks import (
    BimanualBoxEnv, BimanualToyEnv, BimanualCupEnv, BimanualMedEnv, BimanualToolEnv, BimanualGroceryEnv)
for cls in [BimanualBoxEnv,BimanualToyEnv,BimanualCupEnv,BimanualMedEnv,BimanualToolEnv,BimanualGroceryEnv]:
    for ycb,_,_ in cls.OBJECTS:
        assert (root/ycb).exists(), f'{cls.__name__}: missing {ycb}'
print('all YCB ids resolve')
"
```
Expected: `all YCB ids resolve`

- [ ] **Step 5: GPU-gated smoke (run on GPU node; TUNE anchors/containers here)**

For each env id, render and confirm objects spawn on the table within reach and the container is present:
```bash
for E in BimanualBox-v1 BimanualToy-v1 BimanualCup-v1 BimanualMed-v1 BimanualTool-v1 BimanualGrocery-v1; do
  uv run python -m sim_eval.scripts.render_smoke --env $E --out sim_eval/outputs/${E}_smoke
done
```
Expected: each prints `sensor cameras: ['top_cam', 'left_cam', 'right_cam']` and `ALL OK`; inspect the saved PNGs. If objects overlap at spawn, spill out of the container footprint, or sit outside arm reach, adjust that task's `OBJECTS` anchors / `CONTAINER` size and re-render. Large boxes (box/gro/tool) are the likely offenders.

> If no GPU in this session: mark deferred, do not claim verified.

- [ ] **Step 6: Commit**

```bash
git add sim_eval/tasks/pickplace_tasks/ sim_eval/tasks/__init__.py
git commit -m "feat(sim_eval): add 6 container pick-place tasks (box/toy/cup/med/tool/grocery)"
```

---

## Task 6: Block-stack task (the abstraction outlier)

Primitive colored blocks (no YCB id), no container, stacking success.

**Files:**
- Create: `sim_eval/tasks/pickplace_tasks/block_stack.py`
- Modify: `sim_eval/tasks/pickplace_tasks/__init__.py`

**Interfaces:**
- Consumes: `BimanualPickPlaceEnv` (Task 2), `blocks_stacked` (Task 1).
- Produces: env id `BimanualBlockStack-v1`.

- [ ] **Step 1: Write the block-stack subclass**

```python
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
```

> Note: `evaluate()` computes `success = success_fn().all(dim=0)`. Because every
> row is the same `stacked` flag, `all(dim=0)` == `stacked`. `n_in_box` becomes 0
> or N — diagnostic only, which is acceptable for this task.

- [ ] **Step 2: Add to the package `__init__.py`**

```python
# sim_eval/tasks/pickplace_tasks/__init__.py
"""MolmoAct2 pick-place task suite. Importing each module registers its env."""

from .container_tasks import (
    BimanualBoxEnv,
    BimanualToyEnv,
    BimanualCupEnv,
    BimanualMedEnv,
    BimanualToolEnv,
    BimanualGroceryEnv,
)
from .block_stack import BimanualBlockStackEnv

__all__ = [
    "BimanualBoxEnv",
    "BimanualToyEnv",
    "BimanualCupEnv",
    "BimanualMedEnv",
    "BimanualToolEnv",
    "BimanualGroceryEnv",
    "BimanualBlockStackEnv",
]
```

- [ ] **Step 3: Verify registration (import-time, no GPU)**

Run:
```bash
uv run python -c "
import sim_eval.tasks
from mani_skill.utils.registration import REGISTERED_ENVS
assert 'BimanualBlockStack-v1' in REGISTERED_ENVS
print('block stack registered')
"
```
Expected: `block stack registered`

- [ ] **Step 4: GPU-gated smoke (run on GPU node)**

Run: `uv run python -m sim_eval.scripts.render_smoke --env BimanualBlockStack-v1 --out sim_eval/outputs/BimanualBlockStack-v1_smoke`
Expected: `ALL OK`; PNG shows three colored cubes on the table, no container. Tune anchors if cubes overlap or are out of reach.

- [ ] **Step 5: Commit**

```bash
git add sim_eval/tasks/pickplace_tasks/block_stack.py sim_eval/tasks/pickplace_tasks/__init__.py
git commit -m "feat(sim_eval): add block-stack task (primitive blocks, stacking success)"
```

---

## Task 7: Utensil-tray task

Fork/spoon/knife into a primitive tray with compartments.

**Files:**
- Create: `sim_eval/tasks/pickplace_tasks/utensil_tray.py`
- Modify: `sim_eval/tasks/pickplace_tasks/__init__.py`

**Interfaces:**
- Consumes: `BimanualPickPlaceEnv` (Task 2), `_build_tray` (via the base's `_build_container` `kind="tray"` branch).
- Produces: env id `BimanualUtensilTray-v1`. Uses the default center-in-footprint success (utensil center inside the tray footprint counts; compartments are visual/physical structure, not separate success regions).

- [ ] **Step 1: Write the utensil-tray subclass**

```python
# sim_eval/tasks/pickplace_tasks/utensil_tray.py
"""Utensil-tray task: place fork, spoon, and knife into a compartmented tray.

Uses the base _build_tray container (kind="tray") and the default
center-in-footprint success. The tray's interior dividers are physical structure
for fidelity; success only requires each utensil's center inside the tray
footprint (piling-tolerant, like the box tasks).
"""

from mani_skill.utils.registration import register_env

from ..pickplace_base import BimanualPickPlaceEnv

_TRAY_WHITE = [0.9, 0.9, 0.9, 1.0]


@register_env("BimanualUtensilTray-v1", max_episode_steps=1200)
class BimanualUtensilTrayEnv(BimanualPickPlaceEnv):
    """Place the fork, spoon, and knife into the tray."""

    OBJECTS = [
        ("030_fork",  (-0.30, 0.22), "left"),
        ("031_spoon", (-0.30, -0.18), "right"),
        ("032_knife", (-0.24, -0.30), "right"),
    ]
    CONTAINER = {
        "kind": "tray",
        "inner_half": (0.10, 0.13),   # (hx, hy): long along y to hold utensils
        "pos": (-0.13, 0.0),
        "height": 0.04,
        "wall": 0.008,
        "color": _TRAY_WHITE,
        "n_compartments": 3,
        "divider": 0.004,
    }
    INSTRUCTION = "Place the fork, spoon, and knife into the tray."
```

- [ ] **Step 2: Add to the package `__init__.py`**

```python
# sim_eval/tasks/pickplace_tasks/__init__.py
"""MolmoAct2 pick-place task suite. Importing each module registers its env."""

from .container_tasks import (
    BimanualBoxEnv,
    BimanualToyEnv,
    BimanualCupEnv,
    BimanualMedEnv,
    BimanualToolEnv,
    BimanualGroceryEnv,
)
from .block_stack import BimanualBlockStackEnv
from .utensil_tray import BimanualUtensilTrayEnv

__all__ = [
    "BimanualBoxEnv",
    "BimanualToyEnv",
    "BimanualCupEnv",
    "BimanualMedEnv",
    "BimanualToolEnv",
    "BimanualGroceryEnv",
    "BimanualBlockStackEnv",
    "BimanualUtensilTrayEnv",
]
```

- [ ] **Step 3: Verify registration + YCB ids (import-time, no GPU)**

Run:
```bash
uv run python -c "
import sim_eval.tasks
from mani_skill.utils.registration import REGISTERED_ENVS
assert 'BimanualUtensilTray-v1' in REGISTERED_ENVS
from pathlib import Path
root = Path('sim_eval/.maniskill_data/data/assets/mani_skill2_ycb/models')
from sim_eval.tasks.pickplace_tasks.utensil_tray import BimanualUtensilTrayEnv
for ycb,_,_ in BimanualUtensilTrayEnv.OBJECTS:
    assert (root/ycb).exists(), ycb
print('utensil tray registered, ids resolve')
"
```
Expected: `utensil tray registered, ids resolve`

- [ ] **Step 4: GPU-gated smoke (run on GPU node)**

Run: `uv run python -m sim_eval.scripts.render_smoke --env BimanualUtensilTray-v1 --out sim_eval/outputs/BimanualUtensilTray-v1_smoke`
Expected: `ALL OK`; PNG shows the white tray with 3 compartments and fork/spoon/knife beside it. Tune tray size / divider count if utensils don't fit lanes.

- [ ] **Step 5: Commit**

```bash
git add sim_eval/tasks/pickplace_tasks/utensil_tray.py sim_eval/tasks/pickplace_tasks/__init__.py
git commit -m "feat(sim_eval): add utensil-tray task with compartmented tray"
```

---

## Task 8: Integration verification across all 8 env ids

**Files:**
- None new — verification + optional rollout.

**Interfaces:**
- Consumes: everything above.

- [ ] **Step 1: Confirm all 8 new ids + 2 aliases register, and the full unit suite passes**

Run:
```bash
uv run pytest tests/test_success_logic.py -v
uv run python -c "
import sim_eval.tasks
from mani_skill.utils.registration import REGISTERED_ENVS
ids = ['BimanualBox-v1','BimanualToy-v1','BimanualCup-v1','BimanualMed-v1',
       'BimanualTool-v1','BimanualGrocery-v1','BimanualBlockStack-v1','BimanualUtensilTray-v1',
       'BimanualYAMPutEverythingInBox-v1','DK1PutEverythingInBox-v1']
missing = [e for e in ids if e not in REGISTERED_ENVS]
assert not missing, missing
print('all 10 envs registered')
"
```
Expected: pytest PASS; `all 10 envs registered`.

- [ ] **Step 2: GPU-gated — build + reset every env for both robots, assert action space (14,) and the three policy cameras present**

Run on a GPU node:
```bash
uv run python -c "
import gymnasium as gym, sim_eval.tasks
new = ['BimanualBox-v1','BimanualToy-v1','BimanualCup-v1','BimanualMed-v1',
       'BimanualTool-v1','BimanualGrocery-v1','BimanualBlockStack-v1','BimanualUtensilTray-v1']
for eid in new:
    for robot in ['bimanual_yam','bimanual_dk1']:
        e = gym.make(eid, obs_mode='rgb', control_mode='pd_joint_pos',
                     render_mode='rgb_array', robot_uids=robot)
        obs,_ = e.reset(seed=0)
        cams = sorted((obs.get('sensor_data') or {}).keys())
        assert e.action_space.shape == (14,), (eid, robot, e.action_space.shape)
        assert {'top_cam','left_cam','right_cam'}.issubset(set(cams)), (eid, robot, cams)
        e.close()
    print('ok', eid)
print('ALL 8 build for both robots')
"
```
Expected: `ok <eid>` for each, then `ALL 8 build for both robots`.

> If no GPU available, mark deferred — this is the gate that proves the tasks
> actually build. Do not claim the suite verified until it runs on a GPU node.

- [ ] **Step 3: (Optional, GPU + server) one short rollout per family**

With the YAM server running (`uv run python examples/yam/host_server_yam.py --host 0.0.0.0`), for a couple of env ids:
```bash
uv run python -m sim_eval.run_eval -p remote-yam --remote-url http://<host>:8202/act \
    -e BimanualBox-v1 BimanualBlockStack-v1 -n 3
```
Expected: pipeline runs, produces `outputs/<ts>/videos/...` and `results.json` with `n_in_box`. Numbers are diagnostic only (OOD) — not a correctness gate.

- [ ] **Step 4: Final commit (if any tuning landed during smoke)**

```bash
git add -A
git commit -m "test(sim_eval): verify 8-task suite builds for yam+dk1; tune anchors/containers"
```

---

## Self-Review (completed against spec)

**Spec coverage:**
- 6 container tasks (box/toy/cup/med/tool/gro) → Task 5. ✓
- Block task (stacking success) → Task 6. ✓
- Utensil tray / cut (fork/spoon/knife + `_build_tray`) → Task 7. ✓
- Base class `BimanualPickPlaceEnv` with `OBJECTS`/`CONTAINER`/`INSTRUCTION`/`SUPPORTED_ROBOTS`/`robot_base_p`/overridable `success_fn` → Task 2. ✓
- `_build_open_box` moved to base; `_build_tray` added → Task 2. ✓
- Refactor existing YAM/DK1 PutEverythingInBox onto base, keep old ids as aliases → Task 3. ✓
- Robot selection: `run_eval` derives `robot_uids` from `--policy-type`, optional `--robot-uids` → Task 4. ✓ (collapses 16→8 ids; one env per task.)
- Instructions name rendered YCB objects (OOD honesty); paired/relative reporting + `DK1_RECOLOR` carry forward unchanged → constraints + Task 4 (instruction from attr); recolor untouched. ✓
- `max_episode_steps` raised to 1200 → all registered envs. ✓
- Asset check (every ycb id resolves) → Tasks 5/7 Step "ids resolve". ✓
- Build smoke test (gym.make+reset, action (14,), 3 cameras, render frame to outputs/*_smoke) → Tasks 5/6/7 GPU smoke + Task 8 Step 2. ✓
- Success-fn unit check (inside/outside + stacked/unstacked) → Task 1. ✓
- One short rollout per family (optional, GPU-gated) → Task 8 Step 3. ✓

**Deferred per spec (no task, intentionally):** toilet, clo, and all deformable/contact-rich families (fold cloth, untangle, plate-clean, charging, scan, scoop). No new robots/agents/clients/wire-protocol. ✓

**Type consistency:** `object_in_footprint`/`blocks_stacked` signatures identical in Task 1 def, base `success_fn` (Task 2), and block `success_fn` (Task 6). `CONTAINER` dict schema identical across base and all task subclasses. `OBJECTS` tuple shape `(ycb_id, (x,y), arm)` consistent everywhere (block ignores the id slot in its overridden `_build_objects`). `POLICY_ROBOT` keys match `PolicyType` literals. ✓

**Empirical-risk honesty:** container sizes / anchors flagged "tune on GPU smoke"; GPU-only steps explicitly gated and not claimable as verified without a GPU run. ✓
