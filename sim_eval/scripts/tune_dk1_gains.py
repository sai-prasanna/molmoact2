#!/usr/bin/env python3
"""Local, render-free PD-gain tuning for DK1 via open-loop replay.

The DK1 arms diverge from YAM under identical joint-position commands because
DK1's links are heavier (single-arm ~4.1 kg vs YAM ~3.0 kg, shoulder 5x) and
the YAM-tuned soft gains (stiffness 10-40, force_limit 10-28) let DK1 sag/
saturate differently under gravity. The policy was trained expecting YAM's
gravity droop, so the fix is to make DK1's ACHIEVED joint trajectory match
YAM's — i.e. scale up DK1's stiffness/damping/force_limit.

This replays the recorded YAM action trajectory (sim_eval/outputs/replay_*/
yam_actions.npy) on YAM (reference) and on DK1 at several gain multipliers,
with NO rendering (obs_mode=none) so it runs fast on any node. Metrics:
  - arm-joint tracking RMS of DK1 ACHIEVED qpos vs YAM ACHIEVED qpos
  - max n_in_box reached during the replay (did the ball land in the box)

Run:  PYTHONPATH=. python -m sim_eval.scripts.tune_dk1_gains \
          --actions sim_eval/outputs/replay_2619518/yam_actions.npy --seed 43
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym

import mani_skill.envs  # noqa: F401
import sim_eval.tasks   # noqa: F401
from sim_eval.robots.bimanual_dk1 import BimanualDK1

# interleaved arm-joint indices in the 16-D qpos: [L1,R1,...,L6,R6, grippers]
ARM_IDX = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# DK1 baseline gains (must match bimanual_dk1.py).
BASE_STIFF = [40., 40., 40., 20., 10., 10.]
BASE_DAMP  = [2.5, 2.5, 2.5, 0.5, 1.0, 1.0]
BASE_FLIM  = [28., 28., 28., 10., 10., 10.]


def _set_dk1_gains(k):
    """Scale DK1 class gains: stiffness*k, damping*sqrt(k), force_limit*k."""
    BimanualDK1.arm_stiffness   = [s * k for s in BASE_STIFF]
    BimanualDK1.arm_damping     = [d * np.sqrt(k) for d in BASE_DAMP]
    BimanualDK1.arm_force_limit = [f * k for f in BASE_FLIM]


def replay(env_id, actions, seed):
    env = gym.make(env_id, obs_mode="none", control_mode="pd_joint_pos",
                   reward_mode="sparse", max_episode_steps=len(actions) + 5,
                   sim_config=dict(sim_freq=150, control_freq=30))
    env.reset(seed=seed)
    torch.manual_seed(seed); np.random.seed(seed)
    robot = env.unwrapped.agent.robot
    achieved, max_in_box = [], 0
    for a in actions:
        _, _, _, _, info = env.step(np.asarray(a, dtype=np.float32))
        q = robot.get_qpos()
        q = q.cpu().numpy() if isinstance(q, torch.Tensor) else np.asarray(q)
        achieved.append(q.reshape(-1)[ARM_IDX].copy())
        nib = info.get("n_in_box", 0)
        if isinstance(nib, torch.Tensor):
            nib = float(nib.max().item())
        max_in_box = max(max_in_box, float(nib))
    env.close()
    return np.array(achieved), max_in_box


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actions", required=True)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--scales", type=float, nargs="+", default=[1, 2, 3, 5, 8])
    args = ap.parse_args()

    actions = np.load(args.actions)
    print(f"loaded {actions.shape} actions from {args.actions}\n")

    print("=== YAM reference (open-loop replay of its own actions) ===", flush=True)
    yam_q, yam_box = replay("BimanualYAMPutEverythingInBox-v1", actions, args.seed)
    print(f"  YAM achieved arm-traj {yam_q.shape}  max_n_in_box={yam_box}\n")

    print("=== DK1 at gain scales (tracking RMS vs YAM, max_n_in_box) ===", flush=True)
    print(f"  {'scale':>6} {'armRMS(deg)':>12} {'max_n_in_box':>13}")
    results = []
    for k in args.scales:
        _set_dk1_gains(k)
        dk1_q, dk1_box = replay("DK1PutEverythingInBox-v1", actions, args.seed)
        rms = float(np.sqrt(np.mean((dk1_q - yam_q) ** 2)))
        results.append((k, np.degrees(rms), dk1_box))
        print(f"  {k:>6.1f} {np.degrees(rms):>12.2f} {dk1_box:>13.0f}", flush=True)

    best = min(results, key=lambda r: r[1])
    print(f"\n  best tracking: scale={best[0]}  armRMS={best[1]:.2f}deg  "
          f"n_in_box={best[2]:.0f}   (YAM ref n_in_box={yam_box:.0f})")


if __name__ == "__main__":
    main()
