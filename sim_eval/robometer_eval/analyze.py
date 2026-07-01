#!/usr/bin/env python3
"""Analyze Robometer-4B scores against sim ground truth.

Reads scores.json (produced by run_robometer_batch.py) and reports, for each
(robometer score, reduction) combo:
  - AUROC for success-vs-failure discrimination
  - Spearman rho vs n_in_box (graded progress)
Plus a per-video table. Pure-numpy stats (no sklearn/scipy needed).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC via the Mann-Whitney U statistic (rank method, handles ties)."""
    pos = labels == 1
    neg = labels == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
        i = j + 1
    sum_ranks_pos = ranks[pos].sum()
    u = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho = Pearson correlation of ranks (average ranks for ties)."""
    def rankdata(a):
        order = np.argsort(a, kind="mergesort")
        ranks = np.empty(len(a), dtype=float)
        ranks[order] = np.arange(1, len(a) + 1)
        a_sorted = a[order]
        i = 0
        while i < len(a_sorted):
            j = i
            while j + 1 < len(a_sorted) and a_sorted[j + 1] == a_sorted[i]:
                j += 1
            if j > i:
                avg = (ranks[order[i]] + ranks[order[j]]) / 2.0
                for k in range(i, j + 1):
                    ranks[order[k]] = avg
            i = j + 1
        return ranks
    rx, ry = rankdata(x), rankdata(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help="scores.json from run_robometer_batch.py")
    ap.add_argument("--out", default=None, help="optional path to write metrics.json")
    args = ap.parse_args()

    recs = json.loads(Path(args.scores).read_text())
    recs = [r for r in recs if r["progress"]["final"] is not None]
    if not recs:
        print("No usable records (all empty score arrays).")
        return

    succ = np.array([1 if r["gt_success"] else 0 for r in recs])
    nbox = np.array([r["gt_n_in_box"] for r in recs], dtype=float)

    print(f"N videos: {len(recs)}  |  success={int(succ.sum())} fail={int((succ==0).sum())}")
    print(f"n_in_box distribution: {dict(zip(*np.unique(nbox.astype(int), return_counts=True)))}\n")

    metrics = {"n": len(recs), "n_success": int(succ.sum()), "n_fail": int((succ == 0).sum())}
    print(f"{'score':<16}{'reduction':<10}{'AUROC(succ)':<14}{'Spearman(n_in_box)':<20}")
    print("-" * 60)
    for score_key in ("progress", "success_prob"):
        for red in ("final", "max", "mean"):
            vals = np.array([r[score_key][red] for r in recs], dtype=float)
            a = auroc(vals, succ)
            s = spearman(vals, nbox)
            metrics[f"{score_key}.{red}"] = {"auroc_success": a, "spearman_n_in_box": s}
            print(f"{score_key:<16}{red:<10}{a:<14.3f}{s:<20.3f}")

    # Per-env breakdown — essential for the all-failure task suite, where many tasks have
    # zero n_in_box variance (no graded signal) and success is uniformly False (AUROC undefined).
    envs = sorted({r["env_id"] for r in recs})
    if len(envs) > 1:
        print("\nPer-env (success_prob.final vs n_in_box):")
        print(f"{'env_id':<30}{'n':<4}{'n_in_box vals':<20}{'succ':<6}{'Spearman':<10}")
        pooled_vals, pooled_nbox = [], []
        for env in envs:
            er = [r for r in recs if r["env_id"] == env]
            nb = np.array([r["gt_n_in_box"] for r in er], dtype=float)
            sv = np.array([r["success_prob"]["final"] for r in er], dtype=float)
            nsucc = sum(r["gt_success"] for r in er)
            uniq = sorted(set(int(x) for x in nb))
            rho = spearman(sv, nb) if len(uniq) > 1 else float("nan")
            print(f"{env:<30}{len(er):<4}{str(uniq):<20}{nsucc:<6}{rho:<10.3f}")
            if len(uniq) > 1:  # only tasks with graded signal contribute to the pooled stat
                pooled_vals.extend(sv.tolist()); pooled_nbox.extend(nb.tolist())
        if pooled_vals:
            pr = spearman(np.array(pooled_vals), np.array(pooled_nbox))
            metrics["pooled_spearman_n_in_box_variance_only"] = pr
            metrics["pooled_n_variance_only"] = len(pooled_vals)
            print(f"\nPooled Spearman over tasks WITH n_in_box variance "
                  f"(N={len(pooled_vals)}): {pr:.3f}")

    print("\nPer-video (sorted by ground-truth then robometer progress.final):")
    print(f"{'collection':<14}{'ep':<5}{'gt_succ':<9}{'n_box':<7}"
          f"{'prog.final':<12}{'prog.max':<10}{'succ.max':<10}")
    for r in sorted(recs, key=lambda r: (r["gt_success"], r["progress"]["final"])):
        print(f"{r['collection']:<14}{r['episode']:<5}{str(r['gt_success']):<9}{r['gt_n_in_box']:<7}"
              f"{r['progress']['final']:<12.3f}{r['progress']['max']:<10.3f}"
              f"{r['success_prob']['max'] if r['success_prob']['max'] is not None else float('nan'):<10.3f}")

    if args.out:
        Path(args.out).write_text(json.dumps(metrics, indent=2))
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
