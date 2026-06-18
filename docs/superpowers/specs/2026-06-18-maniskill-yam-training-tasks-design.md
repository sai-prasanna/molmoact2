# Design: ManiSkill YAM/DK1 training-task suite for `sim_eval`

**Date:** 2026-06-18
**Status:** approved (pending spec review)
**Scope:** Extend the `sim_eval/` ManiSkill harness with more task environments drawn from
the MolmoAct2-BimanualYAM training mixture, for zero-shot rollout of the
MolmoAct2-BimanualYAM policy (and DK1 transfer).

## Background

MolmoAct2-BimanualYAM is fine-tuned on the `yam_dual_molmoact2` tag — the
`YAM_BIMANUAL_MOLMOACT2` list in `experiments/launch_scripts/data_constants.py`
(recovered from git HEAD; deleted in the working tree). That's ~741 LeRobot datasets
across 18 task families. Per-episode language instructions live in each dataset's
`meta/tasks_annotated.parquet`. The families and a representative real instruction:

| Family | #datasets | Example instruction |
|---|---|---|
| table buss | 101 | "Place food waste in bin, put dirty dishes in dishwasher, and arrange items neatly." |
| fold cloth | 92 | "Grasp white shirt, fold in half, then fold again, place neatly on table." |
| plate cleaning | 82 | "Pick up sponge, scrub plates, rinse, and stack clean plates aside." |
| untangle | 74 | "Untangle the cables by separating and straightening each one on the table." |
| scan | 57 | "Pick up the mouthwash bottle, scan its barcode, then place it back on the table." |
| block | 43 | "Lift the red block, then the green block, and finally the yellow block." |
| box | 42 | "Place the Pringles can, yellow packet, and green packet into the black box." |
| charging | 40 | "Grasp charger, plug into phone, then switch on socket." |
| scoop | 29 | "Use the red scoop to transfer soybeans into the bowl on the scale until it reads 50g." |
| toilet | 24 | "Place shampoo, soap, toothpaste, toothbrush, and razor into the box." |
| med | 24 | "Place all medicine boxes into the transparent container." |
| cut (utensils) | 22 | "Place the pink knife, forks, and spoons into the white tray's compartments." |
| tool | 20 | "Place the screwdriver, pliers, wrench, and box into the black container, then close the lid." |
| yam (cloth) | 15 | "Grasp the purple cloth, fold it in half, then fold it again into a smaller square." |
| cup | 13 | "Place the yellow cup, white cup, and gray cup into the container. Add the spoon and fork." |
| clo (clothes) | 13 | "Place all clothes into the clear plastic container." |
| gro (grocery) | 12 | "Place the green coffee bag, milk packets, and rice into the basket." |
| toy | 8 | "Place the Rubik's cube, plush toy, yellow figure, and two spinning tops into the box." |

The harness already runs MolmoAct2-YAM zero-shot in ManiSkill. One YAM task exists:
`sim_eval/tasks/yam_tasks/bimanual_put_everything_in_box.py`
(`BimanualYAMPutEverythingInBox-v1`) — two YCB objects placed into a primitive open box.
A near-identical DK1 clone exists (`DK1PutEverythingInBox-v1`): the only differences are
docstring, env id, and a `robot_base_p` value. `run_eval.py` selects the env with `-e <id>`
and the inference client with `-p {remote-yam,remote-dk1,remote-droid}`.

## Goal

Add ManiSkill task environments for the tractable subset of the training families so the
MolmoAct2-BimanualYAM policy (and DK1 via the existing transfer path) can be rolled out
zero-shot on more task variety, reusing the current rollout/metrics machinery.

## Scope

### In scope — 8 task families (first cut)

**Container pick-place cluster (6)** — "place this set of objects into a container":

| Task | YCB objects (verify ids at build) | Container | Instruction (names rendered objects) |
|---|---|---|---|
| box | chips can, cracker box, sugar box, pudding box | black open box | "Place the chips can, cracker box, sugar box, and pudding box into the black box." |
| toy | rubiks cube, lego duplo, toy airplane | open box | "Place the Rubik's cube, the lego block, and the toy airplane into the box." |
| cup | 2–3 nesting cups (065-*), fork, spoon | open container | "Place the cups, then the fork and spoon, into the container." |
| med | gelatin box, pudding box (as medicine boxes) | clear-ish container | "Place the medicine boxes into the container." |
| tool | phillips screwdriver, adjustable wrench, clamp, scissors | black box | "Place the screwdriver, wrench, clamp, and scissors into the black box." |
| gro | cracker box, tomato soup can, sugar box | basket (open box) | "Place the cracker box, soup can, and sugar box into the basket." |

**Block (1):** 3 primitive colored boxes (red/green/yellow). Instruction:
"Stack the red, green, and yellow blocks." Success = all three stacked (see Success below).

**Utensil tray / cut (1):** fork (030), spoon (031), knife (032) placed into a primitive
tray with compartments. Instruction: "Place the fork, spoon, and knife into the tray."

Each task is registered for **both** `bimanual_yam` and `bimanual_dk1` via `SUPPORTED_ROBOTS`;
the robot is chosen at launch (see "Robot selection"). → **8 env ids total.**

### Deferred / out of scope

- **toilet, clo** — within the chosen "container" theme but no YCB analogs (toiletries are
  absent from the set; clothes are deformable). Explicitly deferred.
- **Deformable / contact-rich families** — fold cloth, untangle (cables), plate cleaning
  (sponge/scrub), charging (plug insertion + switch), scan (barcode has no sim semantics),
  scoop (granular particles + weight readout). ManiSkill support is limited/absent; high
  effort, low fidelity. Out of scope.
- No new robots, agents, inference clients, or wire-protocol changes. Reuses the existing
  `bimanual_yam` / `bimanual_dk1` agents and `YAMClient` / `DK1Client`.

## Architecture

### Base class

New file `sim_eval/tasks/pickplace_base.py` defining `BimanualPickPlaceEnv(BaseEnv)`,
generalizing the existing PutEverythingInBox env. Parameterized by class attributes
(overridden per task):

- `OBJECTS`: list of `(ycb_id, anchor_xy, arm)` with `arm ∈ {"left","right"}`
  (left = +y, right = −y). Explicit per-arm assignment keeps 3–5 objects bimanually
  reachable. Anchors spread within each arm's reach so objects don't overlap at spawn.
- `CONTAINER`: spec dict — `kind ∈ {"box","tray"}`, `inner_half`/size, `pos`, `height`,
  `wall`, `color`. Sized so all `OBJECTS` fit without spilling over the rim.
- `INSTRUCTION`: string naming the **actual rendered YCB objects** (not the real-world
  brand names from training) — avoids adding a language↔vision grounding mismatch.
- `SUPPORTED_ROBOTS = ["bimanual_yam","bimanual_dk1"]`; `robot_base_p` class attr
  (default `[-0.65,0,0.01]`); default `robot_uids="bimanual_yam"`.
- `success_fn` (method, overridable): default = every object's center inside the container
  XY footprint and below rim + z-margin (the existing piling-tolerant center-in-box test).

Carried over verbatim from the existing env: scene/table build, YCB loading with
friction override, spawn noise + random yaw, camera configs, `_get_obs_extra`,
`n_in_box` / `n_total` reporting, dense reward. The `_build_open_box` helper moves into the
base; a new `_build_tray` helper adds interior dividers for the utensil task.

### Concrete tasks

Each task = a ~15-line subclass in `sim_eval/tasks/yam_tasks/` (or a new
`sim_eval/tasks/pickplace_tasks/`) setting `OBJECTS` / `CONTAINER` / `INSTRUCTION`
(+ `success_fn`/`max_episode_steps` where needed) and one `@register_env(...)`. Block and
utensil-tray override `success_fn` and (block) the container.

The existing `BimanualYAMPutEverythingInBox-v1` and `DK1PutEverythingInBox-v1` are
refactored to subclass the base, removing the current duplication. Their **old ids are
kept as registered aliases** so existing scripts/sbatch keep working.

### Robot selection (collapses 16 ids → 8)

`run_eval.py` currently does `gym.make(env_id, ...)` without `robot_uids`, so the robot is
the env default — the sole reason the existing YAM/DK1 tasks are separate ids. Change:
derive `robot_uids` from `--policy-type` (`remote-yam`→`bimanual_yam`,
`remote-dk1`→`bimanual_dk1`; `remote-droid` unaffected — DROID tasks are single-robot) and
pass it into `gym.make`. Add an optional `--robot-uids` override. ~3 lines. One env id per
task; the robot is tied to the policy under evaluation.

## Design decisions

- **OOD honesty.** This eval is out-of-distribution: MolmoAct2-YAM was trained on real YAM
  camera images and here sees ManiSkill renders. Per the prior DK1 study
  (`memory/dk1-yam-sim-eval-plan.md`), the dominant DK1 residual gap was *perception/color*
  (recolor → +33pp). Consequences baked into this design: (a) instructions name the rendered
  YCB objects; (b) results are reported **paired/relative** (e.g. baseline vs `DK1_RECOLOR`,
  or YAM-vs-DK1 on the same task/seeds), **not** as absolute "the policy succeeds X% at task
  Y". The existing `DK1_RECOLOR` lever and per-episode `n_in_box` logging carry forward
  unchanged.
- **N-object robustness.** Container sized for N objects; success uses the existing
  center-in-box + z-margin test, which tolerates objects piling/shoving (a strict
  all-fully-inside test would false-negative when items stack).
- **Block success (one unambiguous definition).** All three blocks stacked: each block's
  center is above the one below it (monotonic z by ~one block height) and within an XY
  tolerance of the stack axis. Chosen over "in a line" / "lifted together" (training
  paraphrases disagree); instruction is phrased to match ("Stack ...").
- **Episode horizon.** `max_episode_steps` raised to ~1200 for the 3–5 object tasks
  (the existing task uses 400, but successful YAM episodes ran ~950–1400 steps per the prior
  study; 400 truncates before success). Tunable per task.
- **Assets: ManiSkill YCB-76.** Download the set and **verify every object id against the
  actual downloaded list before hardcoding** — some recalled ids may not exist; substitute
  the closest available. Blocks and the tray are primitive geometry (no asset dependency).

## Components & interfaces

- `sim_eval/tasks/pickplace_base.py` — `BimanualPickPlaceEnv`, `_build_open_box`,
  `_build_tray`. Depends on: ManiSkill `BaseEnv`, `bimanual_yam` / `bimanual_dk1` agents,
  YCB actor builder.
- `sim_eval/tasks/.../<family>.py` × 8 — thin subclasses, each `@register_env(...)`.
- `sim_eval/tasks/__init__.py` — import the new modules so envs register.
- `run_eval.py` — `robot_uids` derivation + optional `--robot-uids` flag.
- Refactored PutEverythingInBox YAM/DK1 envs → subclasses + alias ids.

## Testing / verification

- **Asset check:** download YCB-76; assert every `ycb_id` used resolves to a builder.
- **Build smoke test:** for each env id, `gym.make` + `reset`; assert action space `(14,)`,
  `obs` contains the three cameras `[top_cam,left_cam,right_cam]` + base render, objects
  spawn on the table within reach, container present. Save a render frame per task
  (mirrors the existing `outputs/*_smoke/` convention).
- **Success-fn unit check:** synthetically place objects inside/outside the container (and
  stacked/unstacked for block) and assert `evaluate()` returns the expected `success`.
- **One short rollout per family (optional, GPU-gated):** launch the YAM server, run a few
  episodes via `run_eval.py`, confirm the pipeline produces videos + `results.json` with
  `n_in_box`. Numbers are diagnostic only (OOD) — not a correctness gate.

## Out of scope (restated)

Deformable/contact-rich families (fold cloth, untangle, plate-clean, charging, scan, scoop),
toilet, clo. New robots/agents/clients. Wire-protocol changes. Training or data collection
(this is eval only).
