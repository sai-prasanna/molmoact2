#!/usr/bin/env python3
"""Filmstrip + reward plot: the actual sampled frames above robometer's curves.

For each rollout video, re-extracts the SAME 8 frames the reward model saw
(extract_frames, fps=8, max_frames=8), lays them out as a row, and draws the
per-frame progress + success-probability curves directly below, x-aligned so
frame i sits over sample i. Reuses the saved per-frame arrays from
run_robometer_batch.py (no GPU / no model reload); only decodes the video for
the thumbnails.

Run in the robometer env (needs decord + matplotlib), e.g.:
    cd /mnt/lustre/work/martius/mot244/robometer && \
        uv run python /mnt/.../sim_eval/robometer_eval/plot_filmstrip.py \
            --results-dir /mnt/.../sim_eval/robometer_eval/results_tasksuite
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

from robometer.evals.eval_viz_utils import extract_frames

# Okabe-Ito colorblind-safe pair.
C_PROGRESS = "#0072B2"  # blue
C_SUCCESS = "#D55E00"   # vermillion


def montage(frames: np.ndarray, target_h: int = 180) -> tuple[np.ndarray, int]:
    """Downsample frames to ~target_h tall and concat horizontally. Returns (img, per_frame_w)."""
    T, H, W, _ = frames.shape
    step = max(1, H // target_h)
    small = frames[:, ::step, ::step, :]          # (T, h, w, 3)
    h, w = small.shape[1], small.shape[2]
    strip = np.concatenate([small[i] for i in range(T)], axis=1)  # (h, T*w, 3)
    return strip, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--max-frames", type=int, default=8)
    ap.add_argument("--success-threshold", type=float, default=0.5)
    args = ap.parse_args()

    res = Path(args.results_dir)
    per_video = res / "per_video"
    out_dir = res / "filmstrips"
    out_dir.mkdir(exist_ok=True)

    recs = json.loads((res / "scores.json").read_text())
    n = 0
    for r in recs:
        stem = f"{r['collection']}__{r['env_id']}__ep{r['episode']:03d}"
        prog_p = per_video / f"{stem}.npy"
        if not prog_p.exists():
            print(f"[skip] missing {prog_p.name}")
            continue
        progress = np.load(prog_p)
        succ_p = per_video / f"{stem}_success.npy"
        success = np.load(succ_p) if succ_p.exists() else np.array([])
        if progress.size == 0:
            print(f"[skip] empty {stem}")
            continue
        T = progress.size

        frames = extract_frames(r["video"], fps=args.fps, max_frames=args.max_frames)
        if frames is None or frames.size == 0:
            print(f"[skip] no frames {stem}")
            continue
        if frames.dtype != np.uint8:
            frames = np.clip(frames, 0, 255).astype(np.uint8)
        # align count (extract_frames should already give T, but be safe)
        m = min(T, frames.shape[0])
        frames = frames[:m]
        progress = progress[:m]
        success = success[:m] if success.size else success
        strip, _ = montage(frames)

        fig = plt.figure(figsize=(1.7 * m, 5.2))
        gs = GridSpec(2, 1, height_ratios=[1.15, 1.0], hspace=0.12)

        ax_img = fig.add_subplot(gs[0])
        ax_img.imshow(strip, extent=[-0.5, m - 0.5, 0, 1], aspect="auto")
        ax_img.set_xlim(-0.5, m - 0.5)
        ax_img.set_yticks([])
        ax_img.set_xticks([])
        for s in ax_img.spines.values():
            s.set_visible(False)

        ax = fig.add_subplot(gs[1])
        x = np.arange(m)
        ax.plot(x, progress, "-o", color=C_PROGRESS, lw=2, ms=6, label="Progress")
        if success.size == m:
            ax.plot(x, success, "-o", color=C_SUCCESS, lw=2, ms=6, label="Success prob")
        ax.set_xlim(-0.5, m - 0.5)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xticks(x)
        ax.set_xlabel("sampled frame (evenly spaced across the video)")
        ax.set_ylabel("robometer score")
        ax.grid(axis="x", ls=":", alpha=0.4)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.legend(loc="upper left", frameon=False)

        fig.suptitle(f"{r['env_id']}  ep{r['episode']:03d}  ({r['collection']})   "
                     f"GT: success={r['gt_success']}  n_in_box={r['gt_n_in_box']}",
                     fontsize=11)
        out_png = out_dir / f"{stem}.png"
        fig.savefig(str(out_png), dpi=130, bbox_inches="tight")
        plt.close(fig)
        n += 1
        print(f"[filmstrip] {out_png.name}  gt_success={r['gt_success']} n_in_box={r['gt_n_in_box']}")

    print(f"[done] {n} filmstrips -> {out_dir}")


if __name__ == "__main__":
    main()
