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

# Rollout collections that have saved videos + per-episode labels, grouped by experiment.
# Each collection is (name, path-relative-to-repo-root); a collection's results.json may hold
# several tasks (env_ids). Instructions per env_id come from INSTRUCTIONS below.
GROUPS = {
    # Successful/mixed PutEverythingInBox rollouts (16 succ / 14 fail): success-vs-fail + n_in_box.
    "puteverything": [
        ("dk1_2620105", "sim_eval/outputs/dk1_2620105/20260618_114931"),
        ("dk1_2620104", "sim_eval/outputs/dk1_2620104/20260618_114820"),
        ("yam_2620161", "sim_eval/outputs/yam_2620161/20260618_115653"),
    ],
    # The 8-task pick-place suite (this branch's tasks): all failures, so n_in_box-only correlation.
    "tasksuite": [
        ("dk1tasks_2647334", "sim_eval/outputs/dk1tasks_2647334/20260701_163624"),
        ("yamtasks_2621307", "sim_eval/outputs/yamtasks_2621307/20260618_172556"),
    ],
}

# Per-env instruction strings, copied from the tasks' INSTRUCTION class attrs (tasks/**/*.py).
INSTRUCTIONS = {
    "DK1PutEverythingInBox-v1": "put everything into the box",
    "BimanualYAMPutEverythingInBox-v1": "put everything into the box",
    "BimanualBox-v1": "Place the can, cracker box, sugar box, and pudding box into the black box.",
    "BimanualToy-v1": "Place the Rubik's cube, the lego block, and the toy airplane into the box.",
    "BimanualCup-v1": "Place the cups, then the fork and spoon, into the container.",
    "BimanualMed-v1": "Place the medicine boxes into the container.",
    "BimanualTool-v1": "Place the screwdriver, wrench, clamp, and scissors into the black box.",
    "BimanualGrocery-v1": "Place the cracker box, soup can, and sugar box into the basket.",
    "BimanualUtensilTray-v1": "Place the fork, spoon, and knife into the tray.",
    "BimanualBlockStack-v1": "Stack the red, green, and yellow blocks.",
}


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
    # Diagnostics: Unsloth can leave the model in MIXED precision (e.g. fp32 visual patch-embed
    # while norms are bf16), which trips "expected BFloat16 but found Float" in the visual tower.
    from collections import Counter
    dtype_hist = Counter(str(p.dtype) for p in reward_model.parameters())
    print(f"[setup] param dtype histogram (pre-cast): {dict(dtype_hist)}", flush=True)

    # Force a uniform floating dtype so no sub-module dtype mismatches remain at inference.
    model_dtype = max(dtype_hist, key=dtype_hist.get)  # majority dtype string, e.g. 'torch.bfloat16'
    model_dtype = getattr(torch, model_dtype.split(".")[-1])
    reward_model = reward_model.to(model_dtype)
    print(f"[setup] cast whole model to {model_dtype}", flush=True)
    return reward_model, tokenizer, collator, is_discrete, num_bins, model_dtype


def infer_video(frames, task, *, reward_model, tokenizer, collator, is_discrete, num_bins,
                device, model_dtype):
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
            v = v.to(device)
            # bf16 weights + fp32 pixel_values -> layer_norm dtype mismatch; cast floats to model dtype
            if torch.is_floating_point(v):
                v = v.to(model_dtype)
            progress_inputs[k] = v

    if not getattr(infer_video, "_dbg", False):
        infer_video._dbg = True
        dbg = {k: (str(v.dtype) if hasattr(v, "dtype") else type(v).__name__)
               for k, v in progress_inputs.items()}
        print(f"[dbg] progress_inputs dtypes: {dbg}", flush=True)

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
    ap.add_argument("--max-frames", type=int, default=8,
                    help="Max frames evenly sampled across each video (default 8, matches training)")
    ap.add_argument("--limit", type=int, default=0,
                    help="If >0, process only the first N videos (smoke test)")
    ap.add_argument("--group", choices=[*GROUPS, "all"], default="puteverything",
                    help="Which collection group to run (default: puteverything)")
    args = ap.parse_args()

    if args.group == "all":
        collections = [c for g in GROUPS.values() for c in g]
    else:
        collections = GROUPS[args.group]

    repo = Path(args.repo_root)
    out_dir = Path(args.out_dir)
    (out_dir / "per_video").mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] loading {args.model_path} on {device} ...", flush=True)
    model, tokenizer, collator, is_discrete, num_bins, model_dtype = build_model(args.model_path, device)
    print(f"[setup] model_dtype={model_dtype}", flush=True)
    print(f"[setup] done. is_discrete={is_discrete} num_bins={num_bins}", flush=True)

    records = []
    for coll_name, coll_rel in collections:
        coll_dir = repo / coll_rel
        results = json.loads((coll_dir / "results.json").read_text())
        for env_id, task_res in results["tasks"].items():
            instruction = INSTRUCTIONS.get(env_id)
            if instruction is None:
                print(f"[skip] no instruction for env {env_id}", flush=True)
                continue
            labels = {e["episode"]: e for e in task_res["episodes"]}
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
                    frames, instruction, reward_model=model, tokenizer=tokenizer,
                    collator=collator, is_discrete=is_discrete, num_bins=num_bins, device=device,
                    model_dtype=model_dtype,
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
