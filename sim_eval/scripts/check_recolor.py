"""Render one top_cam frame of DK1 with DK1_RECOLOR active and report the
mean arm-pixel color, to confirm the runtime material override took effect
before launching a multi-hour eval. Run inside a GPU allocation."""
import os
import numpy as np
import gymnasium as gym
import imageio.v2 as iio

import sim_eval.tasks  # noqa: F401  (registers DK1PutEverythingInBox-v1)


def main():
    env = gym.make(
        "DK1PutEverythingInBox-v1",
        obs_mode="rgb",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        sensor_configs=dict(shader_pack="rt-fast"),
    )
    obs, _ = env.reset(seed=0)
    img = obs["sensor_data"]["top_cam"]["rgb"]
    if hasattr(img, "detach"):
        img = img.detach().cpu().numpy()
    img = np.asarray(img)
    if img.ndim == 4:
        img = img[0]
    out = f"/tmp/recolor_check_{os.environ.get('DK1_RECOLOR','off')}.png"
    iio.imwrite(out, img.astype(np.uint8))

    f = img[..., :3].astype(float)
    H, W, _ = f.shape
    mx, mn = f.max(-1), f.min(-1)
    sat = (mx - mn) / (mx + 1e-6)
    gray = (sat < 0.12) & (mx > 20) & (mx < 230)
    mask = np.zeros((H, W), bool); mask[int(H * 0.55):] = True
    px = f[gray & mask]
    print(f"RECOLOR={os.environ.get('DK1_RECOLOR','off')}  saved {out}")
    if len(px):
        print("  arm-gray median RGB:", np.median(px, 0).round(1),
              " (YAM target ~91,89,89 ; black DK1 ~34,34,34)")
    env.close()


if __name__ == "__main__":
    main()
