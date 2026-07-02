# Design: truly-bimanual lift-and-place task for `sim_eval` (`BimanualLiftPlaceStand-v1`)

**Date:** 2026-07-02
**Status:** draft (pending user review)
**Branch:** `sim-eval-yam-tasks`
**Scope:** Add one genuinely two-arm-coordinated ManiSkill3 task to the `sim_eval`
harness and roll out MolmoAct2-BimanualYAM zero-shot on it, plus the DK1 (black)
transfer, reusing the existing rollout/metrics machinery.

## Motivation

The existing 9 tasks in the suite are **not truly bimanual**: each object is
pre-assigned to one arm (`OBJECTS = [(ycb_id, anchor_xy, arm)]`, `arm ∈ {left,right}`)
and the two arms perform *independent* single-arm pick-places into a shared
container. Nothing forces the arms to cooperate on the same object.

This task requires the two arms to coordinate on a **single** object: a bar too
long for one gripper to carry level. The "delete one arm" litmus passes — one 8 cm
gripper cannot lift a 30 cm bar off the table and place it level on a raised stand.

### Dataset grounding (stated honestly)

Neither two-arm lift nor handover is a *literally-named* family in the 18-family
`yam_dual_molmoact2` training mixture. The mixture's genuinely two-hand behaviors
(fold cloth, untangle cables, scoop granular) are all deformable/contact-rich and
were explicitly deferred as infeasible in rigid-body ManiSkill3
(`2026-06-18-maniskill-yam-training-tasks-design.md`). The only *attested* rigid
two-arm coordination is "close the lid" (tool family), which is only weakly
bimanual. Two-arm lift is therefore **domain-inspired**, not attested: the real
bimanual YAM plainly lifts bulky items exceeding single-gripper span during
table-buss / box-packing, but that is not a distinct instruction. This is a
deliberate, user-approved choice of *canonical-bimanual over dataset-literal*.

### Reference

ManiSkill3 ships `TwoRobotPickCube-v1` / `TwoRobotStackCube-v1`. Their mechanism
(object reachable only by arm A, goal reachable only by arm B → forced handoff) is
the *handover* pattern, and their reward code is Panda-specific and unusable here.
We reuse only the *idea* of an object-geometry that forces coordination. Our robot
is a single 14-DOF bimanual agent (arms ~0.48 m apart on a shared base), not two
1-m-separated Pandas, so a centered wide object suits it far better than a handoff.

## Task specification

**Env id:** `BimanualLiftPlaceStand-v1`. Registered for **both** `bimanual_yam`
and `bimanual_dk1` (`SUPPORTED_ROBOTS`), robot chosen at launch via `--policy-type`
(`remote-yam` → `bimanual_yam`, `remote-dk1` → `bimanual_dk1`), exactly like the
existing suite.

**Instruction:** `"Pick up the long bar with both arms and place it on the stand."`
(names the rendered primitive — no brand/vision mismatch.)

### Scene (all coordinates in the env frame; robot base at `[-0.65, 0, 0.01]`)

- **Bar** (the object): a single **dynamic** primitive box, long along the y-axis
  (the left↔right axis so each end faces one arm).
  - half-size `[0.02, 0.15, 0.02]` → 0.30 m long, 0.04 m thick/tall.
  - spawn centered and near the robot: `x ≈ -0.32`, `y = 0`, resting on the table
    (`z = half-height`), with the existing small pos noise (`spawn_noise = 0.02`)
    and a *small* yaw jitter (lock roll/pitch). Ends sit at `y ≈ ±0.15`, inboard of
    the arm mounts (`±0.24`) so both are reachable.
  - mass ≈ 0.2 kg, friction = `object_friction` (1.0), restitution 0.
  - Built by overriding `_build_objects` (primitive, not YCB).
- **Stand** (the place target): a **static** box placed forward of the bar so the
  policy carries the bar up and away from its spawn.
  - top surface at `z_top ≈ 0.12`; footprint sized to receive a 0.30 m bar:
    half-size `[0.05, 0.17, 0.06]` at `pos = [-0.15, 0, 0.06]` (center), i.e. top
    at `z = 0.12`.
  - Built by overriding `_build_container` for `CONTAINER["kind"] == "stand"`
    (add a `_build_stand` helper alongside `_build_open_box` / `_build_tray`).

*(All sizes/positions are starting points for closed-loop visual tuning on GPU, as
with every task in this suite — flagged, not assumed correct.)*

### Success and graded metric (no `run_eval.py` change)

`run_eval.py::_run_episode` aggregates exactly two keys from `evaluate()`:
`success` (latched to True if ever True) and `n_in_box` (**max over steps**, plus a
histogram + average). We reuse both:

- **`success`** (binary, the real goal): bar **placed on the stand** —
  bar center XY within the stand footprint (`|x+0.15| < 0.06 ∧ |y| < 0.05`),
  bar center z within `[z_top, z_top + margin]` (resting on top, `margin ≈ 0.04`),
  **tilt < 25°** (level), and bar nearly static (low linear velocity → settled,
  not mid-throw).
- **`n_in_box`** (repurposed as a discrete **progress stage**, 0–2 — this is the
  "check if lift happened anyway" signal):
  - **0** — bar center z < 0.06 (still ~on the table).
  - **1 — lifted**: bar center z ≥ 0.09 **and** tilt < 30° **and** *both* arm TCPs
    within 0.08 m of their respective bar ends (`left_tcp`↔`+y end`,
    `right_tcp`↔`−y end`). The two-TCP check is what forces this stage to be a
    genuine two-arm lift rather than a single-arm flick.
  - **2 — placed**: the `success` condition above.
  `n_in_box` is the **max stage reached** during the episode; `avg_n_in_box` and
  `n_in_box_hist` (over {0,1,2}) then give the graded curve even when binary
  success is 0% (expected: in-dist YAM tops out ~33% on the *easiest* prior task).

`n_total` is reported as 2 (max stage) for histogram labeling consistency.

### Testable success logic (TDD seam)

The env build needs GPU/Vulkan, but the success **math** is pure and CPU-testable —
the same seam the suite already uses (`sim_eval/tasks/success_logic.py` +
`tests/test_success_logic.py`, currently 7/7). Add pure functions:

- `bar_tilt_ok(bar_quat, max_deg) -> mask` — angle of the bar's long (local-y) axis
  from the horizontal plane is below `max_deg` (via the z-component of the rotated
  axis; `asin`).
- `both_ends_grasped(left_tcp_p, right_tcp_p, bar_p, bar_quat, half_len, radius) -> mask`
  — each TCP is within `radius` of the corresponding rotated bar end.
- `placed_on_stand(bar_p, bar_quat, bar_lin_vel, stand_pos, stand_half, z_top, ...) -> mask`
  — the `success` predicate above.
- `lift_stage(...) -> int tensor` — composes the three into stage {0,1,2}.

`evaluate()` calls these with real tensors; unit tests call them with hand-built
poses (level vs tilted, on-table vs lifted vs on-stand, one-TCP vs two-TCP) and
assert the expected masks/stages. New tests added to the existing test file.

## Architecture / files

- **New** `sim_eval/tasks/pickplace_tasks/lift_place.py`:
  `BimanualLiftPlaceStandEnv(BimanualPickPlaceEnv)` overriding `_build_objects`
  (primitive bar), `_build_container` (stand via new `_build_stand`),
  `_get_obs_extra` (bar pose + stand center + per-arm TCP→end offsets),
  `success_fn`/`evaluate` (progress stage + placed), `INSTRUCTION`,
  `max_episode_steps` (default 1200, tunable), and one `@register_env(...)`.
  `SUPPORTED_ROBOTS = ["bimanual_yam", "bimanual_dk1"]`.
- **New** `_build_stand` helper (in `pickplace_base.py`, next to `_build_open_box` /
  `_build_tray`): a solid static box, top at a given z.
- **Edit** `sim_eval/tasks/success_logic.py`: add the 4 pure functions above.
- **Edit** `sim_eval/tasks/__init__.py`: import/register the new env.
- **Edit** `sim_eval/tests/test_success_logic.py`: add unit tests for the new math.
- **No change** to `run_eval.py`, inference clients, wire protocol, agents, or the
  DK1 transfer path — all reused verbatim.

### Accessing arm TCPs

`evaluate()`/`_get_obs_extra` need per-arm TCP world poses. The `BimanualDK1`/
`BimanualYAM` agents expose the arms; the exact attribute (e.g. `left_tcp`,
`agent.tcp`, or a named link) is **verified against `robots/bimanual_*.py` before
coding** (open risk if the agents expose only a single combined TCP — fallback:
use the gripper base links' world poses). Flagged for the implementation plan.

## Rollout plan (deliverable)

Reuse the existing launch scripts with the new env id:

- **DK1 (black), zero-shot** — primary deliverable. Same YAM server, `remote-dk1`
  client, as job 2647334:
  `POLICY_TYPE=remote-dk1 ENVS=BimanualLiftPlaceStand-v1 MAX_STEPS=1200`
  via `sim_eval/scripts/dk1_tasks_sim_eval.sbatch`.
- **YAM native** — in-distribution control (approved), same scripts, `remote-yam`.
- N ≈ 3–6 episodes each first (smoke + signal), A100. Report **in sim** with the
  standard OOD caveat (MolmoAct2 trained on real images, sees SAPIEN renders).
  Headline metric = `avg_n_in_box` (mean progress stage) + `n_in_box_hist`;
  binary `success_rate` reported but expected low.

## Success criteria for this work

1. Env builds for both robots; action space `(14,)`; top/left/right cams present
   (GPU build check, like the 16/16 matrix for the existing suite).
2. `pytest sim_eval/tests/test_success_logic.py` green (new + existing).
3. Smoke render shows: bar spawns centered on the table, stand present and raised,
   bar liftable (not fixed to the table).
4. A short DK1 rollout produces a `results.json` with the progress-stage histogram;
   at minimum, evidence of stage-1 (lift) events distinguishes this from a no-op.

## Explicitly out of scope

- Handover pattern (deferred; poor fit for close-set shared-base arms).
- Multiple objects, YCB objects, or a second bimanual behavior.
- Dense-reward shaping for RL (rollout is a pretrained VLA; `reward_mode="none"`).
- Any hardware/deploy change; any recolor/offset lever (orthogonal, see
  `dk1-yam-sim-eval-plan`).
- Extrapolating sim percentages to the real DK1.
