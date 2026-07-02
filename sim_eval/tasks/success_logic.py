"""Pure tensor success predicates for the bimanual pick-place tasks.

No ManiSkill / SAPIEN imports — these are unit-testable on CPU. Env code in
pickplace_base.py calls them from evaluate().
"""

import math
import torch


def object_in_footprint(
    p: torch.Tensor,
    cx: float,
    cy: float,
    half_x: float,
    half_y: float,
    z_lo: float,
    z_hi: float,
) -> torch.Tensor:
    """True where an object's center is inside the container XY footprint and z band.

    p: (b, 3) object center positions. Returns (b,) bool.
    Piling-tolerant: tests the center only, with a z band that extends above the
    rim, so objects that stack/shove inside still count.
    """
    dx = torch.abs(p[:, 0] - cx)
    dy = torch.abs(p[:, 1] - cy)
    inside_xy = (dx < half_x) & (dy < half_y)
    z_ok = (p[:, 2] > z_lo) & (p[:, 2] < z_hi)
    return inside_xy & z_ok


def blocks_stacked(
    positions: torch.Tensor,
    block_height: float,
    xy_tol: float,
    gap_lo_frac: float = 0.5,
    gap_hi_frac: float = 1.5,
) -> torch.Tensor:
    """True per env where N blocks form a single vertical stack.

    positions: (N, b, 3). Returns (b,).
    Criterion: sort blocks by z; every consecutive z-gap lies in
    [gap_lo_frac, gap_hi_frac] * block_height, and all blocks' XY lie within
    xy_tol of the stack mean XY.
    """
    n, b, _ = positions.shape
    z = positions[..., 2]                       # (N, b)
    z_sorted, _ = torch.sort(z, dim=0)          # (N, b)
    gaps = z_sorted[1:] - z_sorted[:-1]         # (N-1, b)
    gap_lo = gap_lo_frac * block_height
    gap_hi = gap_hi_frac * block_height
    gaps_ok = ((gaps > gap_lo) & (gaps < gap_hi)).all(dim=0)   # (b,)

    xy = positions[..., :2]                      # (N, b, 2)
    mean_xy = xy.mean(dim=0, keepdim=True)       # (1, b, 2)
    xy_dev = torch.linalg.norm(xy - mean_xy, dim=-1)  # (N, b)
    xy_ok = (xy_dev < xy_tol).all(dim=0)         # (b,)

    return gaps_ok & xy_ok


def rotate_vec_by_quat(quat_wxyz: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate a single 3-vector v by a batch of wxyz quaternions.

    quat_wxyz: (b, 4) w,x,y,z (assumed unit). v: (3,). Returns (b, 3).
    Uses v' = v + 2w (q_xyz x v) + 2 q_xyz x (q_xyz x v).
    """
    w = quat_wxyz[:, 0:1]                       # (b,1)
    xyz = quat_wxyz[:, 1:]                       # (b,3)
    vb = v.to(quat_wxyz).unsqueeze(0).expand(xyz.shape[0], 3)  # (b,3)
    t = 2.0 * torch.cross(xyz, vb, dim=-1)
    return vb + w * t + torch.cross(xyz, t, dim=-1)


def bar_tilt_ok(quat_wxyz: torch.Tensor, max_deg: float) -> torch.Tensor:
    """True where the bar's long (local +y) axis is within max_deg of horizontal.

    quat_wxyz: (b,4). Returns (b,) bool. Tilt = asin(|world_axis_z|).
    """
    axis = rotate_vec_by_quat(quat_wxyz, torch.tensor([0.0, 1.0, 0.0]))
    z = axis[:, 2].abs().clamp(0.0, 1.0)
    tilt = torch.asin(z)
    return tilt < math.radians(max_deg)


def placed_on_stand(
    bar_p: torch.Tensor,
    quat_wxyz: torch.Tensor,
    speed: torch.Tensor,
    cx: float, cy: float, half_x: float, half_y: float,
    z_lo: float, z_hi: float,
    max_tilt_deg: float, max_speed: float,
) -> torch.Tensor:
    """True where the bar rests level, settled, centered on the stand.

    bar_p: (b,3) center; quat_wxyz: (b,4); speed: (b,) linear speed. Returns (b,).
    """
    dx = (bar_p[:, 0] - cx).abs() < half_x
    dy = (bar_p[:, 1] - cy).abs() < half_y
    z_ok = (bar_p[:, 2] > z_lo) & (bar_p[:, 2] < z_hi)
    level = bar_tilt_ok(quat_wxyz, max_tilt_deg)
    settled = speed < max_speed
    return dx & dy & z_ok & level & settled


def lift_stage(
    bar_p: torch.Tensor,
    quat_wxyz: torch.Tensor,
    both_grasped: torch.Tensor,
    placed: torch.Tensor,
    z_lift: float,
    max_tilt_deg_lift: float,
) -> torch.Tensor:
    """Discrete progress stage per env: 0 on table, 1 lifted-level-both-arms, 2 placed.

    Returns (b,) long. `placed` overrides `lifted`.
    """
    lifted = (bar_p[:, 2] >= z_lift) & bar_tilt_ok(quat_wxyz, max_tilt_deg_lift) & both_grasped
    stage = torch.zeros(bar_p.shape[0], dtype=torch.long, device=bar_p.device)
    stage[lifted] = 1
    stage[placed] = 2
    return stage
