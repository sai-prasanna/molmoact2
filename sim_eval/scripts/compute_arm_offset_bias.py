"""Deployable form of the grasp-center offset correction.

measure_grasp_offset.py finds a constant tool-frame translation `t` (in DK1's
link6-7 frame) that best aligns DK1's grasp center with YAM's. Realising it by
moving the gripper subtree desyncs the sim from the real DK1 hardware (and froze
the joint when tried as a mount link). Instead we emulate the SAME grasp-center
shift with a constant ARM-JOINT bias dq, applied to the commanded joints in the
action adapter -- a pure calibration constant that runs identically on the real
robot (no per-step IK, no URDF edits).

At config q, shifting the tool by t (tool frame) is a world displacement
R_d(q) @ t. The joint correction that produces it (1st order) is
    dq(q) = J_pos(q)^+ @ (R_d(q) @ t),
J_pos = finite-difference position Jacobian of DK1's tool0. We average dq over
the reach configs (same set measure_grasp_offset uses) for a robust constant.

Run:  PYTHONPATH=. python -m sim_eval.scripts.compute_arm_offset_bias
"""
import numpy as np

from sim_eval.scripts.measure_grasp_offset import (
    dk1_link67_T, G_DK1_LOCAL, compute_mount_transform, CONFIGS,
)


def dk1_tool_pos(q):
    """World position of DK1's grasp center (tool0) at joint config q."""
    T = dk1_link67_T(np.asarray(q, float)) @ G_DK1_LOCAL
    return T[:3, 3]


def pos_jacobian(q, eps=1e-5):
    """Finite-difference 3x6 position Jacobian of the grasp center."""
    q = np.asarray(q, float)
    J = np.zeros((3, 6))
    for i in range(6):
        dq = q.copy(); dq[i] += eps
        J[:, i] = (dk1_tool_pos(dq) - dk1_tool_pos(q)) / eps
    return J


def main():
    M, xyz, rpy, resid, rms = compute_mount_transform()
    t = M[:3, 3]  # desired tool-frame translation (link6-7 frame), metres
    print(f"grasp-center correction t (link6-7 frame) = {t.round(6)}  |t|={np.linalg.norm(t)*1000:.2f} mm")

    # Average the per-config joint correction over the reaching poses (skip home:
    # the arm is folded there and never grasps from it).
    grasp_cfgs = {k: v for k, v in CONFIGS.items() if k != "home"}
    biases = []
    for name, q in grasp_cfgs.items():
        R_d = dk1_link67_T(np.asarray(q, float))[:3, :3]
        e_world = R_d @ t
        J = pos_jacobian(q)
        dq = np.linalg.pinv(J) @ e_world
        biases.append(dq)
        print(f"  {name:7s} dq(rad) = {dq.round(5)}  |dq|={np.linalg.norm(dq):.5f}")
    bias = np.mean(biases, axis=0)
    print("\n=== constant arm-joint bias (rad), apply to BOTH arms ===")
    print("DK1_ARM_BIAS =", ", ".join(f"{b:.6f}" for b in bias))
    print(f"  max |bias| = {np.abs(bias).max():.5f} rad = {np.degrees(np.abs(bias).max()):.3f} deg")


if __name__ == "__main__":
    main()
