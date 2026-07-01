#!/usr/bin/env python3
"""Batch Robometer-4B inference over sim rollout videos.

Loads the reward model ONCE, then runs per-frame progress + success inference over every
video listed in a collection manifest, joining each to its ManiSkill ground-truth
(``success``, ``n_in_box``) from the rollout ``results.json``.

Mirrors the semantics of robometer's ``scripts/example_inference_local.py`` (same frame
extraction and ``compute_batch_outputs`` call) but avoids reloading the 4B model per video.

Must run inside the robometer uv env, e.g.:
    cd /mnt/lustre/work/martius/mot244/robometer && \
        uv run python /mnt/.../sim_eval/robometer_eval/run_robometer_batch.py \
            --out-dir /mnt/.../sim_eval/robometer_eval/results

Outputs, under ``--out-dir``:
    per_video/<collection>__<env>__<epNNN>.npy            per-frame progress
    per_video/<collection>__<env>__<epNNN>_success.npy    per-frame success prob
    scores.json                                           reductions + ground truth per video
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from robometer.data.dataset_types import ProgressSample, Trajectory
from robometer.evals.eval_server import compute_batch_outputs
from robometer.evals.eval_viz_utils import extract_frames
from robometer.utils.save import load_model_from_hf
from robometer.utils.setup_utils import setup_batch_collator

# Rollout collections that have BOTH saved videos and per-episode labels.
# (path is relative to the molmoact2 repo root; env is the task key inside results.json)
COLLECTIONS = [
    ("dk1_2620105", "sim_eval/outputs/dk1_2620105/20260618_114931", "DK1PutEverythingInBox-v1"),
    ("dk1_2620104", "sim_eval/outputs/dk1_2620104/20260618_114820", "DK1PutEverythingInBox-v1"),
    ("yam_2620161", "sim_eval/outputs/yam_2620161/20260618_115653", "BimanualYAMPutEverythingInBox-v1"),
]
TASK_INSTRUCTION = "put everything into the box"


def build_model(model_path: str, device: torch.device):
    exp_config, tokenizer, processor, reward_model = load_model_from_hf(
        model_path=model_path, device=device
    )
    reward_model.eval()
    collator = setup_batch_collator(processor, tokenizer, exp_config, is_eval=True)

    loss_config = getattr(exp_config, "loss", None)
    is_discrete = (
        getattr(loss_config, "progress_loss_type", "l2").lower() == "discrete"
        if loss_config else False
    )
    num_bins = (
        getattr(loss_config, "progress_discrete_bins", None)
        or getattr(exp_config.model, "progress_discrete_bins", 10)
    )
    return reward_model, tokenizer, collator, is_discrete, num_bins


def infer_video(frames, task, *, reward_model, tokenizer, collator, is_discrete, num_bins, device):
    """Return (progress_per_frame, success_prob_per_frame) as float32 arrays."""
    T = int(frames.shape[0])
    traj = Trajectory(
        frames=frames,
        frames_shape=tuple(frames.shape),
        task=task,
        id="0",
        metadata={"subsequence_length": T},
        video_embeddings=None,
    )
    batch = collator([ProgressSample(trajectory=traj, sample_type="progress")])
    progress_inputs = batch["progress_inputs"]
    for k, v in progress_inputs.items():
        if hasattr(v, "to"):
            progress_inputs[k] = v.to(device)

    results = compute_batch_outputs(
        reward_model, tokenizer, progress_inputs,
        sample_type="progress", is_discrete_mode=is_discrete, num_bins=num_bins,
    )
    progress_pred = results.get("progress_pred", [])
    progress = (np.asarray(progress_pred[0], dtype=np.float32)
                if progress_pred else np.array([], dtype=np.float32))
    outputs_success = results.get("outputs_success", {}) or {}
    success_probs = outputs_success.get("success_probs", [])
    success = (np.asarray(success_probs[0], dtype=np.float32)
               if success_probs else np.array([], dtype=np.float32))
    return progress, success


def reductions(arr):
    if arr.size == 0:
        return {"final": None, "max": None, "mean": None}
    return {"final": float(arr[-1]), "max": float(arr.max()), "mean": float(arr.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default="/mnt/lustre/work/martius/mot244/molmoact2")
    ap.add_argument("--model-path", default="robometer/Robometer-4B")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fps", type=float, default=8.0,
                    help="Sampling fps; capped by --max-frames (default 8)")
    ap.add_argument("--max-frames", type=int, default=32,
                    help="Max frames evenly sampled across each video (default 32)")
    ap.add_argument("--limit", type=int, default=0,
                    help="If >0, process only the first N videos (smoke test)")
    args = ap.parse_args()

    repo = Path(args.repo_root)
    out_dir = Path(args.out_dir)
    (out_dir / "per_video").mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] loading {args.model_path} on {device} ...", flush=True)
    model, tokenizer, collator, is_discrete, num_bins = build_model(args.model_path, device)
    print(f"[setup] done. is_discrete={is_discrete} num_bins={num_bins}", flush=True)

    records = []
    for coll_name, coll_rel, env_id in COLLECTIONS:
        coll_dir = repo / coll_rel
        results = json.loads((coll_dir / "results.json").read_text())
        labels = {e["episode"]: e for e in results["tasks"][env_id]["episodes"]}
        video_dir = coll_dir / "videos" / env_id
        for vid in sorted(video_dir.glob("ep*.mp4")):
            if args.limit and len(records) >= args.limit:
                break
            ep = int(vid.stem.replace("ep", ""))
            lab = labels.get(ep)
            if lab is None:
                print(f"[skip] {vid} has no label", flush=True)
                continue
            frames = extract_frames(str(vid), fps=args.fps, max_frames=args.max_frames)
            if frames is None or frames.size == 0:
                print(f"[skip] {vid} no frames", flush=True)
                continue
            if frames.dtype != np.uint8:
                frames = np.clip(frames, 0, 255).astype(np.uint8)
            progress, success = infer_video(
                frames, TASK_INSTRUCTION, reward_model=model, tokenizer=tokenizer,
                collator=collator, is_discrete=is_discrete, num_bins=num_bins, device=device,
            )
            stem = f"{coll_name}__{env_id}__ep{ep:03d}"
            np.save(out_dir / "per_video" / f"{stem}.npy", progress)
            np.save(out_dir / "per_video" / f"{stem}_success.npy", success)
            rec = {
                "collection": coll_name, "env_id": env_id, "episode": ep,
                "video": str(vid), "num_frames": int(frames.shape[0]),
                "gt_success": bool(lab["success"]), "gt_n_in_box": int(lab["n_in_box"]),
                "progress": reductions(progress), "success_prob": reductions(success),
            }
            records.append(rec)
            print(f"[done] {stem} frames={frames.shape[0]} "
                  f"gt_success={rec['gt_success']} n_in_box={rec['gt_n_in_box']} "
                  f"prog_final={rec['progress']['final']} succ_max={rec['success_prob']['max']}",
                  flush=True)
            # write incrementally so a crash still leaves partial results
            (out_dir / "scores.json").write_text(json.dumps(records, indent=2))

    print(f"[all done] {len(records)} videos -> {out_dir/'scores.json'}", flush=True)


if __name__ == "__main__":
    main()
