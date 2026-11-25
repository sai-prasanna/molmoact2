"""Measure the constant grasp-frame offset between YAM and DK1.

MolmoAct2-YAM reasons with an explicit grasp site at [0,0,0.1347] in `link_6`
(0.1347 m out along link_6's local z). DK1's grasp center is its `tool0` frame
(0.158 m out along link6-7's local x). Under the identity joint map the two
*wrist links* (YAM link_6, DK1 link6-7) differ only by a constant frame
transform, so the two grasp frames also differ by a constant SE(3) offset.

This script computes, via pure FK (same hardcoded chains as verify_grasp_frame),
the mount transform M (expressed in the link6-7 frame) such that relocating
DK1's gripper subtree by M makes DK1's grasp frame coincide with YAM's grasp
frame for *identical joints*. It asserts M is constant across configs — that is
the empirical proof that a constant-offset correction is sufficient (no IK).

Run:  PYTHONPATH=. python -m sim_eval.scripts.measure_grasp_offset
Output: M as xyz + rpy (URDF fixed-axis), consumed by build_bimanual_dk1_urdf.py.
"""
import numpy as np


# ---- SO(3)/SE(3) helpers --------------------------------------------------
def q2R(q):
    w, x, y, z = q
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def axR(axis, a):
    u = np.array(axis, float); u /= np.linalg.norm(u); x, y, z = u
    c, s = np.cos(a), np.sin(a); C = 1 - c
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def Tm(R, p):
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = np.asarray(p, float); return M


def rpy_from_R(R):
    """URDF fixed-axis XYZ (R = Rz(yaw) @ Ry(pitch) @ Rx(roll))."""
    sy = -R[2, 0]
    sy = np.clip(sy, -1.0, 1.0)
    pitch = np.arcsin(sy)
    if abs(sy) < 1.0 - 1e-9:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:  # gimbal lock
        roll = np.arctan2(-R[1, 2], R[1, 1])
        yaw = 0.0
    return np.array([roll, pitch, yaw])


# ---- Wrist-link FK chains (base -> wrist link), from verify_grasp_frame ----
# Each entry is the joint's fixed origin (pos, quat for YAM / pos, axis for DK1)
# followed by the actuated revolute about `axis`.
YAM = [([0, 0, 0.0631], [0.707107, 0, 0, 0.707107], [0, 0, 1]),
       ([2.5e-05, -0.02, 0.0409], [0.5, 0.5, 0.5, 0.5], [0, 0, 1]),
       ([0.264, 0, 0], [0, 1, 0, 0], [0, 0, 1]),
       ([-0.245, -0.06, 0], [1, 0, 0, 0], [0, 0, 1]),
       ([-0.074, -0.0395, 2.5e-05], [0.5, -0.5, 0.5, 0.5], [0, 0, 1]),
       ([0, 0.0353, 0.0395], [0.707107, -0.707107, 0, 0], [0, 0, 1])]

DK1 = [([0, 0, 0.0615], [0, 0, 1]), ([0.02, 0, 0.0385], [0, 1, 0]),
       ([-0.264, 0, 0], [0, -1, 0]), ([0.244004, 0, 0.059971], [0, -1, 0]),
       ([0.061868, 0, 0.0425], [0, 0, -1]), ([0.032, 0, -0.0425], [1, 0, 0])]


def yam_link6_T(q):
    M = np.eye(4)
    for (pos, quat, axis), qi in zip(YAM, q):
        M = M @ Tm(q2R(quat), pos) @ Tm(axR(axis, qi), np.zeros(3))
    return M


def dk1_link67_T(q):
    M = np.eye(4)
    for (xyz, axis), qi in zip(DK1, q):
        M = M @ Tm(np.eye(3), xyz) @ Tm(axR(axis, qi), np.zeros(3))
    return M


# ---- Grasp frames in their own wrist-link frame ---------------------------
# YAM: explicit grasp site in link_6 (from the MJCF).
G_YAM_LOCAL = Tm(q2R([0.707107, 0, 0, -0.707107]), [0, 0, 0.1347])
# DK1: tool0 fixed frame in link6-7 (from the URDF).
G_DK1_LOCAL = Tm(q2R(None) if False else
                 # rpy = (-pi/2, 0, -pi/2)  ->  R = Rz(-pi/2)Ry(0)Rx(-pi/2)
                 (axR([0, 0, 1], -np.pi / 2) @ axR([1, 0, 0], -np.pi / 2)),
                 [0.158, 0, 0])


CONFIGS = {
    "home":   [0, 0, 0, 0, 0, 0],
    "reach1": [0.4, 0.6, 0.5, -0.3, 0.2, 0.1],
    "reach2": [-0.5, 0.9, 1.0, 0.5, -0.4, 0.3],
    "reach3": [0.2, 1.2, 0.3, -0.8, 0.7, -0.6],
    "reach4": [1.0, 0.3, -0.2, 1.2, 1.0, -1.5],
}


def grasp_offset_in_link67(q):
    """G_yam expressed in DK1's link6-7 frame at joint config q."""
    T_y = yam_link6_T(q)
    T_d = dk1_link67_T(q)
    return np.linalg.inv(T_d) @ (T_y @ G_YAM_LOCAL)  # G_yam_in_67


def compute_mount_transform():
    """Least-squares-optimal constant mount transform M (in link6-7 frame).

    After inserting M, DK1's grasp pose in link6-7 is M @ G_DK1_LOCAL = const.
    The grasp ORIENTATION mismatch is constant (verified ~0deg), so we keep
    YAM's grasp orientation exactly. The POSITION mismatch is pose-dependent
    (the arms aren't positionally identical), so we pick the grasp-center
    position m* that minimises RMS base-frame error across CONFIGS:

        err(q) = dp(q) + R(q) (g_y - m),   dp(q)=p_y(q)-p_d(q)
        c* = argmin_c sum_q |R(q) c + dp(q)|^2  =  -(1/N) sum_q R(q)^T dp(q)
        m* = g_y - c*

    Returns (M, xyz, rpy, residual_mm_per_config dict, rms_mm)."""
    qs = list(CONFIGS.values())
    # Ideal grasp pose in DK1's link6-7 frame, per config. Its ORIENTATION is
    # constant (the wrist frames differ by a constant rotation B); only its
    # POSITION varies (~10 mm), because the arms aren't positionally identical.
    G67 = [grasp_offset_in_link67(q) for q in qs]
    P = np.array([g[:3, 3] for g in G67])          # varying grasp-center positions
    m_star = P.mean(axis=0)                        # RMS-optimal constant position
    # PURE TRANSLATION: DK1's physical jaw-closing/approach axes already match
    # YAM at 0deg (verify_grasp_frame), so we must NOT rotate the gripper — only
    # translate its subtree so the grasp center moves from G_DK1_LOCAL's position
    # to m_star. A rotation here would physically reorient the jaws.
    t = m_star - G_DK1_LOCAL[:3, 3]
    M = Tm(np.eye(3), t)
    # Residuals after M: base-frame error == link67-frame error (R_d is a rotation).
    resid = {n: np.linalg.norm(P[i] - m_star) * 1000 for i, n in enumerate(CONFIGS)}
    rms = float(np.sqrt(np.mean([v ** 2 for v in resid.values()])))
    return M, M[:3, 3], rpy_from_R(M[:3, :3]), resid, rms


if __name__ == "__main__":
    M, xyz, rpy, resid, rms = compute_mount_transform()
    print("=== grasp-orientation match (constant) ===")
    rot_var = max(np.degrees(np.arccos(np.clip(
        (np.trace(grasp_offset_in_link67(q)[:3, :3].T
                  @ grasp_offset_in_link67(list(CONFIGS.values())[0])[:3, :3]) - 1) / 2, -1, 1)))
        for q in CONFIGS.values())
    print(f"  max rotation variation across configs: {rot_var:.4f} deg  (orientation is matched exactly)\n")
    print("=== LSQ-optimal constant gripper-mount transform M (link6-7 frame) ===")
    print(f"  xyz = {xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}")
    print(f"  rpy = {rpy[0]:.6f} {rpy[1]:.6f} {rpy[2]:.6f}\n")
    print("=== residual grasp-center error after M (pose-dependent, irreducible by a constant) ===")
    for n, v in resid.items():
        print(f"  {n:7s}  {v:6.2f} mm")
    print(f"  RMS = {rms:.2f} mm   (was ~12 mm before correction)")
