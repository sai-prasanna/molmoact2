#!/usr/bin/env python3
"""Diagnostic: does the embodiment gap live in EXECUTION or PERCEPTION?

Phase A — run the MolmoAct2-YAM policy closed-loop on the YAM env at a seed
known to succeed, recording the per-step APPLIED action (post-adapter 14-D
joint+gripper target) and saving YAM's video.

Phase B — replay that exact action sequence OPEN-LOOP on the DK1 env at the
same seed (no policy in the loop), saving DK1's video + wrist/top input frames.

Because the YAM and DK1 action adapters are identical (identity arm + 1-2g
gripper), the recorded YAM applied action can be fed straight to the DK1 env.

Interpretation:
  DK1 replay SUCCEEDS -> execution/kinematics transfer fine; the closed-loop
      failure is PERCEPTION (DK1 looks out-of-distribution to the VLM).
  DK1 replay FAILS    -> genuine execution/geometry gap -> IK retargeting.

Run (inside a GPU alloc, YAM server already up on --url):
  PYTHONPATH=. python -m sim_eval.scripts.replay_yam_on_dk1 \
      --url http://127.0.0.1:8202/act --seed 43 --max-steps 1200
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym

import mani_skill.envs  # noqa: F401
import sim_eval.tasks   # noqa: F401  (registers YAM + DK1 envs)
from sim_eval.inference.client import YAMClient
from sim_eval.run_eval import _save_video, _save_image, _capture_frame, _extract_input_frames


def _success(info) -> bool:
    sv = info.get("success", False)
    if isinstance(sv, torch.Tensor):
        return bool(sv.item() if sv.numel() == 1 else sv.any().item())
    return bool(sv)


def _make_env(env_id, shader):
    return gym.make(env_id, obs_mode="rgb", control_mode="pd_joint_pos",
                    render_mode="rgb_array", max_episode_steps=10_000,
                    reward_mode="none", sensor_configs=dict(shader_pack=shader),
                    sim_config=dict(sim_freq=150, control_freq=30))


def _yam_attempt(env, client, seed, max_steps, instruction, out):
    """One YAM closed-loop rollout at `seed`; record applied actions + success."""
    obs, _ = env.reset(seed=seed)
    torch.manual_seed(seed); np.random.seed(seed)
    client.reset()
    actions, frames = [], []
    success_step = None
    f = _capture_frame(env)
    if f is not None: frames.append(f)
    for step in range(max_steps):
        action = client.infer(obs, instruction)   # post-adapter, == DK1 apply
        actions.append(np.asarray(action, dtype=np.float32).copy())
        obs, _, term, trunc, info = env.step(action)
        f = _capture_frame(env)
        if f is not None: frames.append(f)
        if _success(info) and success_step is None:
            success_step = step + 1
            print(f"[A] YAM SUCCESS at seed {seed} step {success_step}", flush=True)
        term = bool(term.any()) if isinstance(term, torch.Tensor) else bool(term)
        trunc = bool(trunc.any()) if isinstance(trunc, torch.Tensor) else bool(trunc)
        if term or trunc:
            break
        if success_step is not None and step + 1 >= success_step + 60:
            break  # keep replay bounded
    return actions, frames, success_step is not None


def run_phase_a(url, seed, max_steps, attempts, instruction, shader, out):
    """Loop YAM over seeds until a SUCCESS (policy is non-deterministic — flow
    matching isn't server-seeded). Return (actions, winning_seed) of the first
    successful rollout, or the last attempt if none succeed."""
    env = _make_env("BimanualYAMPutEverythingInBox-v1", shader)
    client = YAMClient(url=url)
    last = None
    for i in range(attempts):
        s = seed + i
        print(f"[A] YAM attempt {i+1}/{attempts} at seed {s}", flush=True)
        actions, frames, ok = _yam_attempt(env, client, s, max_steps, instruction, out)
        last = (actions, s, frames)
        if ok:
            _save_video(frames, out / "yam_closedloop.mp4")
            _save_image(_extract_input_frames(env.unwrapped.get_obs()).get("top_cam"),
                        out / "yam_init_top.png")
            env.close()
            print(f"[A] recorded {len(actions)} actions; winning seed={s}", flush=True)
            return actions, s, True
        print(f"[A] seed {s} failed ({len(actions)} steps); retrying", flush=True)
    # none succeeded
    actions, s, frames = last
    _save_video(frames, out / "yam_closedloop.mp4")
    env.close()
    print(f"[A] NO YAM success in {attempts} attempts; using last seed={s}", flush=True)
    return actions, s, False


def run_phase_b(actions, seed, instruction, shader, out):
    """Replay the recorded actions open-loop on DK1; return success."""
    env = _make_env("DK1PutEverythingInBox-v1", shader)
    obs, _ = env.reset(seed=seed)
    torch.manual_seed(seed); np.random.seed(seed)
    for cam, fr in _extract_input_frames(obs).items():
        _save_image(fr, out / f"dk1_init_{cam}.png")
    frames = []
    f = _capture_frame(env)
    if f is not None: frames.append(f)
    success_step = None
    for step, action in enumerate(actions):
        obs, _, term, trunc, info = env.step(action)
        f = _capture_frame(env)
        if f is not None: frames.append(f)
        if _success(info) and success_step is None:
            success_step = step + 1
            print(f"[B] DK1 REPLAY SUCCESS at step {success_step}", flush=True)
        term = bool(term.any()) if isinstance(term, torch.Tensor) else bool(term)
        trunc = bool(trunc.any()) if isinstance(trunc, torch.Tensor) else bool(trunc)
        if term or trunc:
            break
    # also dump DK1 wrist/top input frames at the end of the reach for visual diff
    for cam, fr in _extract_input_frames(obs).items():
        _save_image(fr, out / f"dk1_final_{cam}.png")
    _save_video(frames, out / "dk1_replay.mp4")
    env.close()
    print(f"[B] DK1 replay success={success_step is not None}", flush=True)
    return success_step is not None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8202/act")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--attempts", type=int, default=6)
    ap.add_argument("--shader", default="rt-fast")
    ap.add_argument("--out", default="sim_eval/outputs/replay_diag")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    instruction = "put everything into the box"

    print("=== Phase A: YAM closed-loop until success (record actions) ===", flush=True)
    actions, win_seed, yam_ok = run_phase_a(
        args.url, args.seed, args.max_steps, args.attempts, instruction, args.shader, out)
    np.save(out / "yam_actions.npy", np.stack(actions))
    print(f"[A] winning seed = {win_seed}", flush=True)

    print("=== Phase B: replay YAM actions open-loop on DK1 (same seed) ===", flush=True)
    dk1_ok = run_phase_b(actions, win_seed, instruction, args.shader, out)

    verdict = ("EXECUTION OK -> failure is PERCEPTION" if (yam_ok and dk1_ok)
               else "EXECUTION GAP -> IK retargeting" if yam_ok and not dk1_ok
               else "INCONCLUSIVE (YAM did not succeed this seed)")
    print(f"\n=== RESULT  YAM={yam_ok}  DK1_replay={dk1_ok}  ->  {verdict} ===", flush=True)
    (out / "verdict.txt").write_text(f"yam={yam_ok} dk1_replay={dk1_ok}\n{verdict}\n")


if __name__ == "__main__":
    main()
