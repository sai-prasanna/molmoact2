# Robometer validation on DK1/YAM sim rollouts — design

**Date:** 2026-07-01
**Goal:** Check whether Robometer-4B (a learned VLM reward model, https://github.com/robometer/robometer)
gives an *accurate* trajectory measure on the sim rollouts we collected, by comparing its scores
against ManiSkill ground-truth (`success`, `n_in_box`).

## Data

The `PutEverythingInBox` rollouts that have both saved video **and** per-episode labels
(only the first 10 episodes per collection have videos; `max_videos=10` in `run_eval.py`):

| collection (`sim_eval/outputs/...`)      | embodiment | videos | success/fail | n_in_box |
|------------------------------------------|-----------|--------|--------------|----------|
| `dk1_2620105/20260618_114931`            | DK1       | 10     | 8 / 2        | {1,2}    |
| `dk1_2620104/20260618_114820`            | DK1       | 10     | 5 / 5        | {0,1,2}  |
| `yam_2620161/20260618_115653`            | YAM       | 10     | 3 / 7        | {0,1,2}  |

Total: **30 videos, 16 success / 14 fail**, graded `n_in_box ∈ {0,1,2}`.

- Video = third-person `rgb_array` scene render (whole task visible), written by `_save_video` at fps=30.
  This is a coherent scene for a VLM to judge (not a single sensor cam — those go to `frames/*.png`).
- Task instruction for all three: `"put everything into the box"` (from `run_eval.py:40-42`).
- Ground truth per episode: `success` (bool), `n_in_box` (int) in each `results.json`.

The other collections (`dk1tasks_*`, `yamtasks_*`) are all-failure (`success_rate=0.0`) and are **not**
part of the headline validation; they may be added later as a correlate-only extension against `n_in_box`.

## Pipeline

1. **Clone robometer** into a sibling dir outside this repo's tree (its own `uv` env, Python 3.10).
   Not added to the molmoact2 repo.
2. **Inference (SLURM, bigger GPU** — matches the dk1/yam rollout jobs; local RTX 2080 Ti 11GB is too
   small for a 4B video model). Download `robometer/Robometer-4B`, then for each of the 30 videos call
   the local inference path (`scripts/example_inference_local.py` semantics):
   `--model-path robometer/Robometer-4B --video X.mp4 --task "put everything into the box" --out X.npy`.
   Save per-frame `progress_pred` and `success_probs` per video.
3. **Analysis script** reduces each trajectory to scalars — **both** `final-frame` and `max-over-frames`
   of both `progress_pred` and `success_probs` — joins to ground-truth from `results.json`, and computes:
   - **AUROC** for success-vs-failure (16/14 split; threshold-free, appropriate for this N).
   - **Spearman ρ** of robometer score vs `n_in_box`.
   - Per-video table + scatter plot for eyeballing.

## Metrics & reduction

- Report AUROC and Spearman for both `final` and `max` reductions; lead with the better-motivated one.
- `final-frame` matches "did it end in success"; `max-over-frames` is robust to end-of-episode drift.

## Honest reading (must be in the writeup)

Robometer was trained on RBM-1M **real-robot** video; our inputs are ManiSkill **renders** → out-of-distribution.
- A **positive** correlation is strong evidence robometer's measure works even on sim footage.
- A **null/weak** result is **ambiguous**: it cannot distinguish "robometer is inaccurate" from
  "sim renders are OOD for robometer." Do not report a null as "robometer is broken."

## Deliverable

Results table + headline numbers (AUROC, Spearman) + short writeup, plus reusable run/analysis scripts
under `sim_eval/robometer_eval/`.
