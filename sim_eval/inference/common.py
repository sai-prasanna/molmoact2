"""MolmoAct2 wire-format schemas, adapters, and obs-extraction primitives."""

import os
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch


@dataclass(frozen=True)
class RemoteServerSchema:
    """Wire contract of a MolmoAct2 /act HTTP endpoint."""
    name: str
    camera_keys: tuple[str, ...]
    state_dim: int
    default_port: int
    norm_tag: str


MOLMOACT2_SCHEMAS: dict[str, RemoteServerSchema] = {
    "droid": RemoteServerSchema(
        name="droid",
        camera_keys=("external_cam", "wrist_cam"),
        state_dim=8,
        default_port=8000,
        norm_tag="franka_droid",
    ),
    "yam": RemoteServerSchema(
        name="yam",
        camera_keys=("top_cam", "left_cam", "right_cam"),
        state_dim=14,
        default_port=8202,
        norm_tag="yam_dual_molmoact2",
    ),
    # DK1 shares YAM's kinematics and is driven by the SAME MolmoAct2-YAM
    # server (identity arm action map), so it reuses the YAM wire contract.
    "dk1": RemoteServerSchema(
        name="dk1",
        camera_keys=("top_cam", "left_cam", "right_cam"),
        state_dim=14,
        default_port=8202,
        norm_tag="yam_dual_molmoact2",
    ),
}


StateAdapter  = Callable[[np.ndarray], np.ndarray]
ActionAdapter = Callable[[np.ndarray], np.ndarray]

_YAM_FINGER_OPEN_RANGE = 0.0475  # matches bimanual_yam.py gripper lower=-0.0475


def droid_state_adapter(qpos: np.ndarray) -> np.ndarray:
    """franka_droid qpos (13-D) → MolmoAct2 DROID state (8-D).

    Active joints: [0..6] fr3 arm, [7] left_outer_knuckle (gripper), [8..12] passive.
    Server expects: [q1..q7, gripper_pos].
    """
    qpos = np.asarray(qpos, dtype=np.float32)
    if qpos.shape[-1] != 13:
        raise ValueError(f"droid_state_adapter: expected (13,), got {qpos.shape}")
    return np.concatenate([qpos[:7], qpos[7:8]])


def yam_state_adapter(qpos: np.ndarray) -> np.ndarray:
    """yam_bimanual qpos (16-D) → MolmoAct2 YAM state (14-D).

    ManiSkill interleaves arms: [L1,R1,L2,R2,...,L6,R6, Lf1,Lf2,Rf1,Rf2].
    Server expects left-block-first: [L1..L6, L_grip, R1..R6, R_grip],
    grip in [0,1] (1=open, 0=closed).
    """
    qpos = np.asarray(qpos, dtype=np.float32)
    if qpos.shape[-1] != 16:
        raise ValueError(f"yam_state_adapter: expected (16,), got {qpos.shape}")
    left_arm  = qpos[[0, 2, 4, 6,  8, 10]]
    right_arm = qpos[[1, 3, 5, 7,  9, 11]]
    l_grip = np.clip(-qpos[12] / _YAM_FINGER_OPEN_RANGE, 0.0, 1.0)
    r_grip = np.clip(-qpos[14] / _YAM_FINGER_OPEN_RANGE, 0.0, 1.0)
    out = np.empty(14, dtype=np.float32)
    out[:6]   = left_arm
    out[6]    = l_grip
    out[7:13] = right_arm
    out[13]   = r_grip
    return out


_DK1_FINGER_OPEN_RANGE = 0.045  # matches bimanual_dk1.py gripper lower=-0.045


_DK1_GRIP_LOWER = -0.045  # joint at action -1 (DK1: this is CLOSED — inverted vs YAM)
_DK1_GRIP_UPPER = 0.001   # joint at action +1 (DK1: this is OPEN)


def dk1_state_adapter(qpos: np.ndarray) -> np.ndarray:
    """bimanual_dk1 qpos (16-D) → MolmoAct2 YAM state (14-D).

    DK1 shares YAM's joint ordering (ManiSkill interleaves arms:
    [L1,R1,...,L6,R6, Lgl,Lgr,Rgl,Rgr]) and the identity arm map. BUT the DK1
    gripper joint convention is INVERTED vs YAM: DK1 is OPEN at the upper limit
    (+0.001) and CLOSED at the lower limit (-0.045), the opposite of YAM. So the
    open fraction (server wants 1=open) is (qpos - lower)/(upper - lower).
    """
    qpos = np.asarray(qpos, dtype=np.float32)
    if qpos.shape[-1] != 16:
        raise ValueError(f"dk1_state_adapter: expected (16,), got {qpos.shape}")
    left_arm  = qpos[[0, 2, 4, 6,  8, 10]]
    right_arm = qpos[[1, 3, 5, 7,  9, 11]]
    span = _DK1_GRIP_UPPER - _DK1_GRIP_LOWER
    l_grip = np.clip((qpos[12] - _DK1_GRIP_LOWER) / span, 0.0, 1.0)
    r_grip = np.clip((qpos[14] - _DK1_GRIP_LOWER) / span, 0.0, 1.0)
    out = np.empty(14, dtype=np.float32)
    out[:6]   = left_arm
    out[6]    = l_grip
    out[7:13] = right_arm
    out[13]   = r_grip
    return out


# Deployable grasp-center offset correction (lever 2). DK1's parallel gripper is
# physically different from YAM's, so at identical joints DK1's grasp center is
# ~11 mm off from where the YAM-trained policy expects it. We compensate with a
# constant arm-joint bias (a pure calibration constant — runs identically on the
# real robot, no IK, no URDF edits). Values from
# scripts/compute_arm_offset_bias.py (J_pos^+ @ R·t averaged over reach poses).
# Applied to BOTH arms only when DK1_ARM_BIAS is set (default off → faithful).
_DK1_ARM_BIAS = np.array(
    [0.023580, -0.045929, -0.010138, -0.006765, -0.012996, 0.000000],
    dtype=np.float32,
)


def _arm_bias() -> np.ndarray:
    """Return the per-joint arm bias from env DK1_ARM_BIAS (off → zeros).

    DK1_ARM_BIAS=1 → use the computed default; "a,b,c,d,e,f" → custom 6-vector.
    """
    v = os.environ.get("DK1_ARM_BIAS", "")
    if not v or v == "0":
        return np.zeros(6, dtype=np.float32)
    if "," in v:
        return np.asarray([float(x) for x in v.split(",")], dtype=np.float32)
    return _DK1_ARM_BIAS


def dk1_action_adapter(action: np.ndarray) -> np.ndarray:
    """YAM server action (14-D) → DK1 pd_joint_pos action (14-D).

    Arm joints pass through (identity map), plus an optional constant grasp-center
    bias (env DK1_ARM_BIAS, off by default). Gripper indices 6 and 13: the DK1
    gripper is INVERTED vs YAM (DK1 OPEN at the upper joint limit), so the
    mapping is the negation of YAM's: server [0,1] (1=open) → +1 (action upper
    = DK1 open), 0 (closed) → -1 (action lower = DK1 closed). i.e. out = 2g-1.
    """
    action = np.asarray(action, dtype=np.float32)
    if action.shape[-1] != 14:
        raise ValueError(f"dk1_action_adapter: expected (14,), got {action.shape}")
    out = action.copy()
    bias = _arm_bias()
    if bias.any():
        out[0:6]  += bias  # left arm
        out[7:13] += bias  # right arm
    for i in (6, 13):
        out[i] = 2.0 * float(action[i]) - 1.0
    return out


def yam_action_adapter(action: np.ndarray) -> np.ndarray:
    """YAM server action (14-D) → ManiSkill pd_joint_pos action (14-D).

    Arm joints pass through as absolute angles. Gripper indices 6 and 13
    are linearly mapped: server [0,1] (1=open) → ManiSkill [-1,1] (-1=open, +1=closed).
    """
    action = np.asarray(action, dtype=np.float32)
    if action.shape[-1] != 14:
        raise ValueError(f"yam_action_adapter: expected (14,), got {action.shape}")
    out = action.copy()
    for i in (6, 13):
        out[i] = 1.0 - 2.0 * float(action[i])
    return out


def extract_camera(obs: dict, maniskill_cam: str) -> np.ndarray:
    """Pull a uint8 RGB image for maniskill_cam out of a ManiSkill obs dict."""
    sensors = obs.get("sensor_data") or {}
    raw = None
    if maniskill_cam in sensors:
        data = sensors[maniskill_cam]
        raw = data.get("rgb") if isinstance(data, dict) else data
    elif maniskill_cam in obs:
        raw = obs[maniskill_cam]
    if raw is None:
        raise KeyError(
            f"Camera '{maniskill_cam}' not in obs. Available: {sorted(sensors.keys())}"
        )
    return _to_uint8(raw)


def extract_qpos(obs: dict) -> np.ndarray:
    """Pull the joint-position vector out of a ManiSkill obs dict."""
    agent = obs.get("agent")
    if isinstance(agent, dict) and "qpos" in agent:
        v = agent["qpos"]
    elif "joint_positions" in obs:
        v = obs["joint_positions"]
    elif "state" in obs:
        v = obs["state"]
    else:
        raise KeyError("Cannot find qpos in obs (tried agent/qpos, joint_positions, state)")
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu().numpy()
    v = np.asarray(v, dtype=np.float32)
    if v.ndim == 2 and v.shape[0] == 1:
        v = v[0]
    return v


def _to_uint8(img) -> np.ndarray:
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    img = np.asarray(img)
    if img.ndim == 4 and img.shape[0] == 1:
        img = img[0]
    if img.dtype != np.uint8:
        img = np.clip(img * 255 if img.max() <= 1 else img, 0, 255).astype(np.uint8)
    return img
