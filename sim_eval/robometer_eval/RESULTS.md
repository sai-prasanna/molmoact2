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
middle you'd expect a progress model to be least certain about. Full per-video numbers: `results/scores.json`;
aggregate metrics: `results/metrics.json`.

## Caveat (important for interpretation)
Robometer was trained on RBM-1M **real-robot** video; our inputs are ManiSkill **renders**
(out-of-distribution). The strong *positive* correlation here is therefore strong evidence that
robometer's measure transfers even to sim footage. (A null result would have been ambiguous —
"robometer inaccurate" vs "sim renders are OOD" — but that is not what we observed.)

## Answer
Yes — on these rollouts robometer gives an accurate measure: near-perfect success/failure ranking
and strong monotonic agreement with graded task progress, despite the sim-vs-real domain gap.

## Notes / knobs
- 8 frames per video (Robometer-4B is trained on 8 frames; more frames OOM the A100 under Unsloth's
  eager attention at 720×1280). Change with `--max-frames`.
- Env fix required: pin `sentence-transformers<5` in robometer (v5 eagerly imports torchcodec, which
  needs FFmpeg libs absent on Galvani). See the design spec / memory note.
- Only the first 10 episodes per collection had saved videos (`max_videos=10` in `run_eval.py`), hence
  30 of the 66 labeled episodes. Re-running rollouts with a higher `max_videos` would enlarge the set.
