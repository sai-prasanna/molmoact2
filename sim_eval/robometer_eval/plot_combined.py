#!/usr/bin/env python3
"""Render robometer's combined progress/success plots from saved per-video arrays.

The cheap visualization: reuses the per-frame progress + success arrays that
run_robometer_batch.py already saved under <results>/per_video/ (8 sampled frames
per video), and calls robometer's own create_combined_progress_success_plot to draw
progress / success-binary / success-prob over the sampled-frame index. No GPU / no
model reload. Each plot's title carries the ManiSkill ground truth for context.

Run in the robometer env (for the plot helper + matplotlib), e.g.:
    cd /mnt/lustre/work/martius/mot244/robometer && \
        uv run python /mnt/.../sim_eval/robometer_eval/plot_combined.py \
            --results-dir /mnt/.../sim_eval/robometer_eval/results_tasksuite
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np

from robometer.evals.eval_viz_utils import create_combined_progress_success_plot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True,
                    help="e.g. .../robometer_eval/results_tasksuite (must have scores.json + per_video/)")
    ap.add_argument("--success-threshold", type=float, default=0.5)
    args = ap.parse_args()

    res = Path(args.results_dir)
    per_video = res / "per_video"
    out_dir = res / "plots"
    out_dir.mkdir(exist_ok=True)

    recs = json.loads((res / "scores.json").read_text())
    n = 0
    for r in recs:
        stem = f"{r['collection']}__{r['env_id']}__ep{r['episode']:03d}"
        prog_p = per_video / f"{stem}.npy"
        succ_p = per_video / f"{stem}_success.npy"
        if not prog_p.exists():
            print(f"[skip] missing {prog_p.name}")
            continue
        progress = np.load(prog_p)
        success = np.load(succ_p) if succ_p.exists() else np.array([])
        if progress.size == 0:
            print(f"[skip] empty progress {stem}")
            continue

        show_success = success.size > 0 and success.size == progress.size
        success_binary = (success > args.success_threshold).astype(np.int32) if show_success else None
        title = (f"{r['env_id']}  ep{r['episode']:03d}  ({r['collection']})\n"
                 f"GT: success={r['gt_success']}  n_in_box={r['gt_n_in_box']}  "
                 f"| frames={progress.size}")
        fig = create_combined_progress_success_plot(
            progress_pred=progress,
            num_frames=progress.size,
            success_binary=success_binary,
            success_probs=success if show_success else None,
            success_labels=None,
            title=title,
        )
        out_png = out_dir / f"{stem}.png"
        fig.savefig(str(out_png), dpi=150, bbox_inches="tight")
        import matplotlib.pyplot as plt
        plt.close(fig)
        n += 1
        print(f"[plot] {out_png.name}  gt_success={r['gt_success']} n_in_box={r['gt_n_in_box']}")

    print(f"[done] {n} plots -> {out_dir}")


if __name__ == "__main__":
    main()
