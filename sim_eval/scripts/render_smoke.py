#!/usr/bin/env python3
"""Headless SAPIEN/Vulkan render smoke test for sim_eval.

Stage 1: render a stock ManiSkill env with rasterization (isolates Vulkan).
Stage 2: render the bimanual YAM env with the requested shader pack.

Run on a GPU node:
    uv run python -m sim_eval.scripts.render_smoke --shader rt-fast
"""
import argparse
import sys
from pathlib import Path

import numpy as np


def _to_img(frame):
    import torch
    if isinstance(frame, torch.Tensor):
        frame = frame.cpu().numpy()
    if frame.ndim == 4:
        frame = frame[0]
    if frame.dtype != np.uint8:
        frame = (frame * 255).clip(0, 255).astype(np.uint8) if frame.max() <= 1 else frame.astype(np.uint8)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shader", default="rt-fast")
    ap.add_argument("--env", default="BimanualYAMPutEverythingInBox-v1",
                    help="Env id to render in stage 2")
    ap.add_argument("--out", default=str(Path(__file__).parent.parent / "outputs" / "render_smoke"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import gymnasium as gym
    import imageio
    import mani_skill.envs  # noqa: F401

    print("=== Stage 1: stock PickCube-v1, default (rasterization) shader ===", flush=True)
    env = gym.make("PickCube-v1", obs_mode="rgb", render_mode="rgb_array",
                   sensor_configs=dict(shader_pack="default"))
    env.reset(seed=0)
    frame = _to_img(env.render())
    imageio.imwrite(out / "stage1_pickcube_default.png", frame)
    print(f"  OK -> {out/'stage1_pickcube_default.png'}  shape={frame.shape}", flush=True)
    env.close()

    print(f"=== Stage 2: {args.env}, shader={args.shader} ===", flush=True)
    import sim_eval.tasks  # noqa: F401  (registers env + robot via side-effect)
    env = gym.make(args.env, obs_mode="rgb",
                   control_mode="pd_joint_pos", render_mode="rgb_array",
                   sensor_configs=dict(shader_pack=args.shader))
    obs, _ = env.reset(seed=0)
    cams = list((obs.get("sensor_data") or {}).keys())
    print(f"  sensor cameras: {cams}", flush=True)
    frame = _to_img(env.render())
    safe = args.env.replace('/', '_')
    imageio.imwrite(out / f"stage2_{safe}_{args.shader}.png", frame)
    print(f"  OK render -> {out/f'stage2_{safe}_{args.shader}.png'}  shape={frame.shape}", flush=True)
    for cam, sd in (obs.get("sensor_data") or {}).items():
        img = sd.get("rgb") if isinstance(sd, dict) else sd
        imageio.imwrite(out / f"stage2_{safe}_sensor_{cam}.png", _to_img(img))
        print(f"  sensor {cam} -> saved", flush=True)
    env.close()
    print("ALL OK", flush=True)


if __name__ == "__main__":
    sys.exit(main())
