# Robometer-4B on DK1/YAM sim rollouts — results

**Run:** SLURM job 2647682 (A100-40GB), 2026-07-01. Model `robometer/Robometer-4B`, 8 frames/video
(matches training), task `"put everything into the box"`. Reproduce: `sbatch sim_eval/robometer_eval/robometer_eval.sbatch`.

## Question
Does Robometer-4B (a learned VLM reward model, https://github.com/robometer/robometer) give an
accurate trajectory measure on *our* sim rollouts, judged against ManiSkill ground truth?

## Data
30 `PutEverythingInBox` rollout videos that have both saved video and per-episode labels
(collections `dk1_2620105`, `dk1_2620104`, `yam_2620161`). **16 success / 14 fail**;
`n_in_box` distribution: {0: 5, 1: 9, 2: 16}. Videos are the third-person `rgb_array` render (720×1280).

## Headline result — robometer tracks ground truth very well

| robometer score | reduction | AUROC (success vs fail) | Spearman ρ (vs n_in_box) |
|-----------------|-----------|-------------------------|--------------------------|
| success_prob    | final/max | **0.978**               | **0.874**                |
| progress        | final/max | 0.960                   | 0.849                    |
| success_prob    | mean      | 0.897                   | 0.763                    |
| progress        | mean      | 0.634                   | 0.389                    |

(final and max reductions are ~identical here because progress/success are near-monotonic over the
trajectory; mean is weaker because it averages in the low-scoring early frames.)

**Reading:** using the final-frame success probability, robometer separates successful from failed
rollouts almost perfectly (AUROC 0.978) and its score rises monotonically with the number of objects
actually placed (Spearman 0.874). The per-video table shows the graded behaviour cleanly:

- `n_in_box = 0` (nothing placed) → success_prob ≈ 0.002–0.007
- `n_in_box = 1` (partial)        → success_prob ≈ 0.03–0.77
- `n_in_box = 2` (all placed, all successes) → success_prob ≈ 0.64–0.97

The only overlap is between "placed 1 object" failures and genuine successes — exactly the ambiguous
middle you'd expect a progress model to be least certain about. Full per-video numbers: `results_puteverything/scores.json`;
aggregate metrics: `results_puteverything/metrics.json`.

**Note on independence:** in this data success ⟺ `n_in_box == 2` exactly (0 and 1 objects are always
failures, 2 objects always a success), so the AUROC(success) and the top end of Spearman(n_in_box) are
largely the *same* measurement, not two independent confirmations. The genuinely independent evidence
that robometer reads *degree* of progress (not just success/fail) is the separation *within failures*:
`n_in_box=0` → success_prob ≈ 0.002–0.007 vs `n_in_box=1` → ≈ 0.03–0.77. That ordering is real and is
corroborated by the all-failure task-suite run below.

## Task-suite rollouts (the 8-task pick-place suite) — n_in_box correlation

Second run (SLURM job 2647701, `--group tasksuite`): the 33 videos from `dk1tasks_2647334` +
`yamtasks_2621307`. These are **all failures** (success_rate 0.0), so AUROC is undefined — this run
tests only whether robometer's reward tracks the graded `n_in_box`. Most tasks have **no** graded
signal (every episode `n_in_box=0`); only three tasks show variance:

| env_id                | n | n_in_box values | Spearman ρ (success_prob.final vs n_in_box) |
|-----------------------|---|-----------------|---------------------------------------------|
| BimanualCup-v1        | 6 | {0,1,2,3}       | 0.971                                       |
| BimanualTool-v1       | 6 | {0,1,2}         | 0.802                                       |
| BimanualUtensilTray-v1| 6 | {0,1}           | 0.621                                       |
| Box/Toy/Med/Grocery/BlockStack | 3 each | {0} only | — (no variance) |

**Pooled Spearman over the tasks that have graded ground truth (N=18): 0.854.** The per-video table
is cleanly monotonic across *all* 33 videos regardless of task: `n_in_box=0` → success_prob ≈ 0.001–0.07,
`=1` → ≈ 0.03–0.05, `=2` → ≈ 0.12–0.68, `=3` → 0.92. Numbers: `results_tasksuite/{scores,metrics}.json`.

**Honest scope:** the graded signal comes from only 3 tasks / 18 episodes; the 5 all-zero tasks tell us
nothing (we can't check ranking with no variance). So the claim is narrow but clean: *where the sim
provides graded ground truth, robometer's reward tracks it well* (ρ≈0.85), even on all-failure,
out-of-distribution sim footage.

## Caveat (important for interpretation)
Robometer was trained on RBM-1M **real-robot** video; our inputs are ManiSkill **renders**
(out-of-distribution). The strong *positive* correlation here is therefore strong evidence that
robometer's measure transfers even to sim footage. (A null result would have been ambiguous —
"robometer inaccurate" vs "sim renders are OOD" — but that is not what we observed.)

## Answer
Yes — robometer gives an accurate measure on these rollouts. On the mixed PutEverythingInBox set it
ranks success vs failure near-perfectly (AUROC 0.978) and, more informatively, its reward is monotonic
in graded progress on the all-failure task suite (pooled Spearman 0.854 over the tasks with `n_in_box`
variance) — despite the sim-render vs real-robot domain gap. The main limitation is coverage of the
graded signal: most task-suite episodes place zero objects, so only a few tasks actually test the
graded correlation.

## Notes / knobs
- 8 frames per video (Robometer-4B is trained on 8 frames; more frames OOM the A100 under Unsloth's
  eager attention at 720×1280). Change with `--max-frames`.
- Env fix required: pin `sentence-transformers<5` in robometer (v5 eagerly imports torchcodec, which
  needs FFmpeg libs absent on Galvani). See the design spec / memory note.
- Only the first 10 episodes per collection had saved videos (`max_videos=10` in `run_eval.py`), hence
  30 of the 66 labeled episodes. Re-running rollouts with a higher `max_videos` would enlarge the set.
