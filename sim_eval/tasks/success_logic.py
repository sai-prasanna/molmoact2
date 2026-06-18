"""Pure tensor success predicates for the bimanual pick-place tasks.

No ManiSkill / SAPIEN imports — these are unit-testable on CPU. Env code in
pickplace_base.py calls them from evaluate().
"""

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
