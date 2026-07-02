# Bimanual Lift-and-Place-on-Stand Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one genuinely two-arm-coordinated ManiSkill3 task
(`BimanualLiftPlaceStand-v1`) — lift a bar too long for one gripper, level, with
both arms, and place it on a raised stand — and roll it out zero-shot on DK1
(black) and native YAM, reusing the existing rollout/metrics machinery.

**Architecture:** A new `BimanualLiftPlaceStandEnv` subclasses the existing
`BimanualPickPlaceEnv`. It builds one dynamic primitive bar + one static stand,
and overrides `evaluate()` to report a binary `success` (placed on stand) plus a
0/1/2 progress stage reused through the existing `n_in_box` key (0 on table →
1 lifted-level-with-both-arms → 2 placed). All geometry that can be tested on CPU
lives in pure functions in `success_logic.py`; the "both arms engaged" check uses
the agents' existing `is_grasping(bar, arm=...)`.

**Tech Stack:** ManiSkill3 (SAPIEN), PyTorch tensors, pytest. Python via `uv run`.

## Global Constraints

- Run everything with `uv run` (bare `python` is not on PATH).
- Tests run from the **repo root** (`pyproject.toml` sets `pythonpath = ["."]`);
  `success_logic` is imported as `from sim_eval.tasks.success_logic import ...`.
- Success-logic functions are **pure torch**, no ManiSkill/SAPIEN imports, inputs
  `(b, …)` batched, outputs `(b,)` — matches the existing `object_in_footprint`.
- ManiSkill pose quaternions are **wxyz**; tables live at z=0 (bar rests at
  z = bar half-height = 0.02).
- Env registered for **both** `bimanual_yam` and `bimanual_dk1`
  (`SUPPORTED_ROBOTS` inherited); robot chosen at launch via `--policy-type`.
- Env build needs GPU/Vulkan; only the success math is unit-testable. GPU/SLURM
  steps are marked; do not expect them to run on the login node.
- Do NOT modify `run_eval.py`, the inference clients, the wire protocol, the
  agents, or the DK1 transfer path.
- Commits scoped to the files each task touches (the branch carries unrelated
  pre-existing churn — never `git add -A`).

---

### Task 1: Pure success-logic functions (CPU/TDD)

**Files:**
- Restore + Test: `tests/test_success_logic.py` (git-tracked but deleted in the
  working tree; restore it, then append new tests)
- Modify: `sim_eval/tasks/success_logic.py`

**Interfaces:**
- Consumes: `torch` only.
- Produces (imported by Task 3's `evaluate()`):
  - `rotate_vec_by_quat(quat_wxyz: Tensor(b,4), v: Tensor(3,)) -> Tensor(b,3)`
  - `bar_tilt_ok(quat_wxyz: Tensor(b,4), max_deg: float) -> Tensor(b,) bool`
  - `placed_on_stand(bar_p: Tensor(b,3), quat_wxyz: Tensor(b,4), speed: Tensor(b,), cx, cy, half_x, half_y, z_lo, z_hi, max_tilt_deg, max_speed) -> Tensor(b,) bool`
  - `lift_stage(bar_p: Tensor(b,3), quat_wxyz: Tensor(b,4), both_grasped: Tensor(b,) bool, placed: Tensor(b,) bool, z_lift: float, max_tilt_deg_lift: float) -> Tensor(b,) long` (values in {0,1,2})

- [ ] **Step 1: Restore the deleted test file**

Run:
```bash
cd /mnt/lustre/work/martius/mot244/molmoact2
git checkout -- tests/test_success_logic.py
uv run pytest tests/test_success_logic.py -q
```
Expected: the existing suite passes (7 passed, per prior work). If `git checkout`
reports the path is unknown, it was never committed on this branch — recreate it
with the two imports it needs (`object_in_footprint`, `blocks_stacked`) later; the
new tests below stand alone regardless.

- [ ] **Step 2: Write the failing tests (append to `tests/test_success_logic.py`)**

```python
import math
from sim_eval.tasks.success_logic import (
    rotate_vec_by_quat, bar_tilt_ok, placed_on_stand, lift_stage,
)

_LONG_AXIS = torch.tensor([0.0, 1.0, 0.0])
_IDENT = torch.tensor([[1.0, 0.0, 0.0, 0.0]])           # no rotation
# 90 deg about x maps local +y -> world +z (bar stood on end)
_ROT90X = torch.tensor([[math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0]])


def test_rotate_identity_is_noop():
    out = rotate_vec_by_quat(_IDENT, _LONG_AXIS)
    assert torch.allclose(out, _LONG_AXIS.unsqueeze(0), atol=1e-6)


def test_rotate_90x_maps_y_to_z():
    out = rotate_vec_by_quat(_ROT90X, _LONG_AXIS)[0]
    assert abs(out[2].item() - 1.0) < 1e-5 and abs(out[1].item()) < 1e-5


def test_tilt_ok_when_level():
    assert bool(bar_tilt_ok(_IDENT, max_deg=25.0)[0]) is True


def test_tilt_not_ok_when_on_end():
    assert bool(bar_tilt_ok(_ROT90X, max_deg=25.0)[0]) is False


def test_placed_on_stand_true_when_centered_level_settled():
    ok = placed_on_stand(
        bar_p=torch.tensor([[-0.15, 0.0, 0.14]]), quat_wxyz=_IDENT,
        speed=torch.tensor([0.0]),
        cx=-0.15, cy=0.0, half_x=0.06, half_y=0.05,
        z_lo=0.11, z_hi=0.17, max_tilt_deg=25.0, max_speed=0.05,
    )
    assert bool(ok[0]) is True


def test_placed_false_when_off_footprint():
    ok = placed_on_stand(
        bar_p=torch.tensor([[-0.15, 0.30, 0.14]]), quat_wxyz=_IDENT,
        speed=torch.tensor([0.0]),
        cx=-0.15, cy=0.0, half_x=0.06, half_y=0.05,
        z_lo=0.11, z_hi=0.17, max_tilt_deg=25.0, max_speed=0.05,
    )
    assert bool(ok[0]) is False


def test_placed_false_when_tilted_or_moving():
    tilted = placed_on_stand(
        bar_p=torch.tensor([[-0.15, 0.0, 0.14]]), quat_wxyz=_ROT90X,
        speed=torch.tensor([0.0]), cx=-0.15, cy=0.0, half_x=0.06, half_y=0.05,
        z_lo=0.11, z_hi=0.17, max_tilt_deg=25.0, max_speed=0.05)
    moving = placed_on_stand(
        bar_p=torch.tensor([[-0.15, 0.0, 0.14]]), quat_wxyz=_IDENT,
        speed=torch.tensor([1.0]), cx=-0.15, cy=0.0, half_x=0.06, half_y=0.05,
        z_lo=0.11, z_hi=0.17, max_tilt_deg=25.0, max_speed=0.05)
    assert bool(tilted[0]) is False and bool(moving[0]) is False


def test_lift_stage_zero_on_table():
    s = lift_stage(
        bar_p=torch.tensor([[-0.32, 0.0, 0.02]]), quat_wxyz=_IDENT,
        both_grasped=torch.tensor([False]), placed=torch.tensor([False]),
        z_lift=0.09, max_tilt_deg_lift=30.0)
    assert int(s[0]) == 0


def test_lift_stage_one_when_lifted_level_both_grasped():
    s = lift_stage(
        bar_p=torch.tensor([[-0.30, 0.0, 0.12]]), quat_wxyz=_IDENT,
        both_grasped=torch.tensor([True]), placed=torch.tensor([False]),
        z_lift=0.09, max_tilt_deg_lift=30.0)
    assert int(s[0]) == 1


def test_lift_stage_zero_when_lifted_but_single_arm():
    s = lift_stage(
        bar_p=torch.tensor([[-0.30, 0.0, 0.12]]), quat_wxyz=_IDENT,
        both_grasped=torch.tensor([False]), placed=torch.tensor([False]),
        z_lift=0.09, max_tilt_deg_lift=30.0)
    assert int(s[0]) == 0


def test_lift_stage_two_when_placed_overrides():
    s = lift_stage(
        bar_p=torch.tensor([[-0.15, 0.0, 0.14]]), quat_wxyz=_IDENT,
        both_grasped=torch.tensor([True]), placed=torch.tensor([True]),
        z_lift=0.09, max_tilt_deg_lift=30.0)
    assert int(s[0]) == 2
```

- [ ] **Step 3: Run the new tests to verify they fail**

Run: `uv run pytest tests/test_success_logic.py -q -k "rotate or tilt or placed or lift_stage"`
Expected: FAIL / ERROR with `ImportError: cannot import name 'rotate_vec_by_quat'`.

- [ ] **Step 4: Implement the functions (append to `sim_eval/tasks/success_logic.py`)**

```python
import math


def rotate_vec_by_quat(quat_wxyz: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate a single 3-vector v by a batch of wxyz quaternions.

    quat_wxyz: (b, 4) w,x,y,z (assumed unit). v: (3,). Returns (b, 3).
    Uses v' = v + 2w (q_xyz x v) + 2 q_xyz x (q_xyz x v).
    """
    w = quat_wxyz[:, 0:1]                       # (b,1)
    xyz = quat_wxyz[:, 1:]                       # (b,3)
    vb = v.to(quat_wxyz).unsqueeze(0).expand(xyz.shape[0], 3)  # (b,3)
    t = 2.0 * torch.cross(xyz, vb, dim=-1)
    return vb + w * t + torch.cross(xyz, t, dim=-1)


def bar_tilt_ok(quat_wxyz: torch.Tensor, max_deg: float) -> torch.Tensor:
    """True where the bar's long (local +y) axis is within max_deg of horizontal.

    quat_wxyz: (b,4). Returns (b,) bool. Tilt = asin(|world_axis_z|).
    """
    axis = rotate_vec_by_quat(quat_wxyz, torch.tensor([0.0, 1.0, 0.0]))
    z = axis[:, 2].abs().clamp(0.0, 1.0)
    tilt = torch.asin(z)
    return tilt < math.radians(max_deg)


def placed_on_stand(
    bar_p: torch.Tensor,
    quat_wxyz: torch.Tensor,
    speed: torch.Tensor,
    cx: float, cy: float, half_x: float, half_y: float,
    z_lo: float, z_hi: float,
    max_tilt_deg: float, max_speed: float,
) -> torch.Tensor:
    """True where the bar rests level, settled, centered on the stand.

    bar_p: (b,3) center; quat_wxyz: (b,4); speed: (b,) linear speed. Returns (b,).
    """
    dx = (bar_p[:, 0] - cx).abs() < half_x
    dy = (bar_p[:, 1] - cy).abs() < half_y
    z_ok = (bar_p[:, 2] > z_lo) & (bar_p[:, 2] < z_hi)
    level = bar_tilt_ok(quat_wxyz, max_tilt_deg)
    settled = speed < max_speed
    return dx & dy & z_ok & level & settled


def lift_stage(
    bar_p: torch.Tensor,
    quat_wxyz: torch.Tensor,
    both_grasped: torch.Tensor,
    placed: torch.Tensor,
    z_lift: float,
    max_tilt_deg_lift: float,
) -> torch.Tensor:
    """Discrete progress stage per env: 0 on table, 1 lifted-level-both-arms, 2 placed.

    Returns (b,) long. `placed` overrides `lifted`.
    """
    lifted = (bar_p[:, 2] >= z_lift) & bar_tilt_ok(quat_wxyz, max_tilt_deg_lift) & both_grasped
    stage = torch.zeros(bar_p.shape[0], dtype=torch.long, device=bar_p.device)
    stage[lifted] = 1
    stage[placed] = 2
    return stage
```

- [ ] **Step 5: Run tests to verify pass**

Run: `uv run pytest tests/test_success_logic.py -q`
Expected: PASS (existing + 11 new).

- [ ] **Step 6: Commit**

```bash
git add tests/test_success_logic.py sim_eval/tasks/success_logic.py
git commit -m "feat(sim_eval): tilt/lift/placed success predicates for lift-place task"
```

---

### Task 2: `BimanualLiftPlaceStand-v1` env + registration

**Files:**
- Create: `sim_eval/tasks/pickplace_tasks/lift_place.py`
- Modify: `sim_eval/tasks/pickplace_tasks/__init__.py`

**Interfaces:**
- Consumes: `BimanualPickPlaceEnv` (base: `_load_scene`, `_after_reconfigure`,
  `robot_base_p`, `spawn_noise`, `object_friction`, `reset_robot_qpos`); the pure
  functions from Task 1; the agents' `is_grasping(actor, arm="left"|"right")`,
  `left_tcp`/`right_tcp`, and `keyframes["home"].qpos`; `actors.build_box`.
- Produces: registered env id `BimanualLiftPlaceStand-v1` (usable via
  `run_eval.py -e BimanualLiftPlaceStand-v1 -p {remote-yam,remote-dk1}`) and the
  class `BimanualLiftPlaceStandEnv`.

- [ ] **Step 1: Write the env module**

Create `sim_eval/tasks/pickplace_tasks/lift_place.py`:

```python
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
```

- [ ] **Step 2: Register the env**

In `sim_eval/tasks/pickplace_tasks/__init__.py` add the import and `__all__`
entry (place with the other imports / list entries):

```python
from .lift_place import BimanualLiftPlaceStandEnv
```
and add `"BimanualLiftPlaceStandEnv",` to `__all__`.

- [ ] **Step 3: Verify registration (no GPU needed)**

Run:
```bash
uv run python -c "import sim_eval.tasks; import gymnasium as gym; \
assert 'BimanualLiftPlaceStand-v1' in gym.registry, 'env not registered'; \
print('registered OK')"
```
Expected: `registered OK`. (If Vulkan warnings print, ignore them — registration
happens at import, before any GPU use.)

- [ ] **Step 4: Commit**

```bash
git add sim_eval/tasks/pickplace_tasks/lift_place.py sim_eval/tasks/pickplace_tasks/__init__.py
git commit -m "feat(sim_eval): BimanualLiftPlaceStand-v1 truly-bimanual lift-place env"
```

---

### Task 3: GPU build + smoke verification (needs A100/Vulkan)

**Files:**
- Use: `sim_eval/run_eval.py` (unchanged), the new env from Task 2.

**Interfaces:**
- Consumes: registered `BimanualLiftPlaceStand-v1`, both robot uids.
- Produces: confirmation that the env builds for both robots (action space (14,),
  three policy cams) and that the bar is dynamic + the stand is present.

*This task runs on a GPU node (`srun`/`sbatch`, a100-galvani). It has no unit
test; the deliverable is the printed/rendered evidence below.*

- [ ] **Step 1: Build matrix for both robots**

On a GPU node:
```bash
source .eval_env.sh
uv run python -c "
import sim_eval.tasks, gymnasium as gym, numpy as np
for uid in ['bimanual_yam','bimanual_dk1']:
    env = gym.make('BimanualLiftPlaceStand-v1', robot_uids=uid,
                   obs_mode='rgb', control_mode='pd_joint_pos', num_envs=1)
    obs,_ = env.reset(seed=0)
    print(uid, 'action', env.action_space.shape,
          'cams', sorted(obs['sensor_data'].keys()))
    env.close()
"
```
Expected: for each uid, `action (14,)` and the three policy cameras
(`top_cam`, `left_cam`, `right_cam`) present.

- [ ] **Step 2: Smoke render — bar dynamic, stand present**

```bash
uv run python -c "
import sim_eval.tasks, gymnasium as gym, numpy as np, imageio.v3 as iio, torch
env = gym.make('BimanualLiftPlaceStand-v1', robot_uids='bimanual_dk1',
               obs_mode='rgb', control_mode='pd_joint_pos', num_envs=1,
               render_mode='rgb_array', sensor_configs=dict(shader_pack='minimal'))
obs,_ = env.reset(seed=0)
z0 = env.unwrapped.bar.pose.p[0,2].item()
# let it settle under gravity with zero action to confirm the bar is not fixed
for _ in range(20):
    obs,_,_,_,info = env.step(env.action_space.sample()*0)
img = env.render()[0].cpu().numpy() if hasattr(env.render()[0],'cpu') else np.asarray(env.render())[0]
iio.imwrite('sim_eval/outputs/liftplace_smoke.png', img.astype('uint8'))
print('bar z0', round(z0,3), 'stage', int(info['n_in_box'].max()), 'saved smoke png')
env.close()
"
```
Expected: PNG saved showing the bar centered on the table and the raised stand in
front; `stage 0` at rest. Visually confirm the bar sits ON the table (dynamic),
not floating/fixed, and both arms can plausibly reach both ends.

- [ ] **Step 3: Record findings (no commit unless retuning)**

If the bar/stand geometry is visibly wrong (bar unreachable, stand overlapping the
bar, bar too thick to grasp), adjust the `BAR_*` / `STAND_*` / `PLACE_*` constants
in `lift_place.py`, re-run Steps 1–2, then:
```bash
git add sim_eval/tasks/pickplace_tasks/lift_place.py
git commit -m "fix(sim_eval): tune lift-place bar/stand geometry from GPU smoke"
```

---

### Task 4: Zero-shot rollout (DK1 + YAM) via SLURM

**Files:**
- Use: `sim_eval/scripts/dk1_tasks_sim_eval.sbatch`,
  `sim_eval/scripts/run_sim_eval_yam_tasks.sh` (both already env-parameterized:
  `ENVS=`, `MAX_STEPS=`, `POLICY_TYPE=`). No new script required.

**Interfaces:**
- Consumes: the launch scripts + a running MolmoAct2-BimanualYAM server on :8202.
- Produces: `results.json` per run with `success_rate`, `avg_n_in_box`
  (= mean progress stage), and `n_in_box_hist` over {0,1,2}.

*SLURM task; no unit test. Deliverable = the two results.json files.*

- [ ] **Step 1: Confirm the launch scripts accept the new env id**

Run (login node, no GPU):
```bash
grep -n "ENVS\|POLICY_TYPE\|MAX_STEPS" sim_eval/scripts/dk1_tasks_sim_eval.sbatch sim_eval/scripts/run_sim_eval_yam_tasks.sh
```
Expected: `ENVS`, `POLICY_TYPE`, `MAX_STEPS` are overridable env vars. (If the
sbatch hardcodes a default ENVS list, that's fine — we override on submit.)

- [ ] **Step 2: Submit the DK1 (black) zero-shot rollout**

```bash
cd /mnt/lustre/work/martius/mot244/molmoact2
ENVS=BimanualLiftPlaceStand-v1 POLICY_TYPE=remote-dk1 MAX_STEPS=1200 \
  sbatch sim_eval/scripts/dk1_tasks_sim_eval.sbatch
```
Expected: a job id. This starts the YAM server, then drives DK1 via `remote-dk1`,
3 episodes, saving to `sim_eval/outputs/dk1tasks_<jobid>/`.

- [ ] **Step 3: Submit the YAM native control**

```bash
ENVS=BimanualLiftPlaceStand-v1 POLICY_TYPE=remote-yam MAX_STEPS=1200 \
  sbatch sim_eval/scripts/dk1_tasks_sim_eval.sbatch
```
(Or `run_sim_eval_yam_tasks.sh` if that is the maintained YAM entry point — either
routes through `run_eval.py` with `-p remote-yam`.)

- [ ] **Step 4: Read results**

When the jobs finish:
```bash
cat sim_eval/outputs/dk1tasks_<jobid>/*/results.json | python -m json.tool
```
Expected keys per task: `success_rate`, `avg_n_in_box`, `n_in_box_hist`,
`avg_steps`. Report **in sim** with the OOD caveat. Headline = `avg_n_in_box`
(mean stage) + histogram; a nonzero count at stage ≥1 confirms lifts occurred
(the "did the lift happen" signal). Binary `success_rate` is expected low.

- [ ] **Step 5: Record the outcome**

Summarize both runs (stage histograms, whether any full placements) for the
session notes / memory update. No code commit.

---

## Self-Review

**Spec coverage:**
- Truly-bimanual single-object task → Task 2 (bar + both-arms `is_grasping`). ✓
- Dataset-honesty framing → in the spec (docstrings reference it); no code needed. ✓
- Success = placed on stand; graded `n_in_box` 0/1/2 → Task 1 (`lift_stage`,
  `placed_on_stand`) + Task 2 (`evaluate`). ✓
- No `run_eval.py` change → confirmed; `evaluate()` returns `success` + `n_in_box`
  keys the existing loop already maxes. ✓
- TDD seam in `success_logic.py` + `tests/` → Task 1. ✓
- Registered for both robots → inherited `SUPPORTED_ROBOTS`; Task 3 Step 1 checks
  both build. ✓
- DK1 + YAM rollout reusing existing scripts → Task 4. ✓
- TCP access risk → resolved: `agent.left_tcp`/`right_tcp` + `is_grasping` exist on
  both agents (verified). ✓

**Placeholder scan:** every code step shows complete code; commands have expected
output. Note deliberate GPU/SLURM steps have no unit test (documented). ✓

**Type consistency:** `rotate_vec_by_quat`/`bar_tilt_ok`/`placed_on_stand`/
`lift_stage` signatures match between Task 1 (definition + tests) and Task 2
(`evaluate` call sites). `evaluate()` returns `success` (bool (b,)), `n_in_box`
(long (b,)), `n_total` — the keys `run_eval._run_episode` consumes. ✓

**Known deviation from spec:** `_build_stand` lives in `lift_place.py` (not
`pickplace_base.py`) to keep the shared base untouched — a strictly smaller blast
radius; noted here.
