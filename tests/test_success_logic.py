import math
import torch

from sim_eval.tasks.success_logic import (
    object_in_footprint, blocks_stacked, rotate_vec_by_quat, bar_tilt_ok,
    placed_on_stand, lift_stage,
)


def test_object_inside_footprint_is_true():
    p = torch.tensor([[-0.15, 0.0, 0.03]])  # at container center, low z
    inside = object_in_footprint(p, cx=-0.15, cy=0.0, half_x=0.1, half_y=0.1, z_lo=0.0, z_hi=0.1)
    assert bool(inside[0]) is True


def test_object_outside_xy_is_false():
    p = torch.tensor([[-0.15, 0.5, 0.03]])  # far in +y, outside footprint
    inside = object_in_footprint(p, cx=-0.15, cy=0.0, half_x=0.1, half_y=0.1, z_lo=0.0, z_hi=0.1)
    assert bool(inside[0]) is False


def test_object_above_rim_is_false():
    p = torch.tensor([[-0.15, 0.0, 0.5]])  # inside XY but flying above z band
    inside = object_in_footprint(p, cx=-0.15, cy=0.0, half_x=0.1, half_y=0.1, z_lo=0.0, z_hi=0.1)
    assert bool(inside[0]) is False


def test_object_in_footprint_batched():
    p = torch.tensor([[-0.15, 0.0, 0.03], [-0.15, 0.5, 0.03]])
    inside = object_in_footprint(p, cx=-0.15, cy=0.0, half_x=0.1, half_y=0.1, z_lo=0.0, z_hi=0.1)
    assert inside.tolist() == [True, False]


def test_blocks_stacked_true():
    # three 5cm blocks stacked at same xy, z = 0.025, 0.075, 0.125
    positions = torch.tensor([
        [[-0.15, 0.0, 0.025]],
        [[-0.15, 0.0, 0.075]],
        [[-0.15, 0.0, 0.125]],
    ])  # (N=3, b=1, 3)
    ok = blocks_stacked(positions, block_height=0.05, xy_tol=0.03)
    assert bool(ok[0]) is True


def test_blocks_not_stacked_when_spread_in_xy():
    positions = torch.tensor([
        [[-0.15, 0.0, 0.025]],
        [[-0.15, 0.20, 0.075]],   # second block far away in y
        [[-0.15, 0.0, 0.125]],
    ])
    ok = blocks_stacked(positions, block_height=0.05, xy_tol=0.03)
    assert bool(ok[0]) is False


def test_blocks_not_stacked_when_all_on_table():
    positions = torch.tensor([
        [[-0.15, 0.00, 0.025]],
        [[-0.15, 0.10, 0.025]],   # all at same z, not stacked
        [[-0.15, 0.20, 0.025]],
    ])
    ok = blocks_stacked(positions, block_height=0.05, xy_tol=0.03)
    assert bool(ok[0]) is False


_LONG_AXIS = torch.tensor([0.0, 1.0, 0.0])
_IDENT = torch.tensor([[1.0, 0.0, 0.0, 0.0]])           # no rotation
# 90 deg about x maps local +y -> world +z (bar stood on end)
_ROT90X = torch.tensor([[math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0]])


def test_rotate_identity_is_noop():
    out = rotate_vec_by_quat(_IDENT, _LONG_AXIS)
    assert torch.allclose(out, _LONG_AXIS.unsqueeze(0), atol=1e-6)


def test_rotate_90x_maps_y_to_z():
    out = rotate_vec_by_quat(_ROT90X, _LONG_AXIS)[0]
    assert abs(out[2].item() - 1.0) < 1e-5 and abs(out[1].item()) < 1e-5


def test_tilt_ok_when_level():
    assert bool(bar_tilt_ok(_IDENT, max_deg=25.0)[0]) is True


def test_tilt_not_ok_when_on_end():
    assert bool(bar_tilt_ok(_ROT90X, max_deg=25.0)[0]) is False


def test_placed_on_stand_true_when_centered_level_settled():
    ok = placed_on_stand(
        bar_p=torch.tensor([[-0.15, 0.0, 0.14]]), quat_wxyz=_IDENT,
        speed=torch.tensor([0.0]),
        cx=-0.15, cy=0.0, half_x=0.06, half_y=0.05,
        z_lo=0.11, z_hi=0.17, max_tilt_deg=25.0, max_speed=0.05,
    )
    assert bool(ok[0]) is True


def test_placed_false_when_off_footprint():
    ok = placed_on_stand(
        bar_p=torch.tensor([[-0.15, 0.30, 0.14]]), quat_wxyz=_IDENT,
        speed=torch.tensor([0.0]),
        cx=-0.15, cy=0.0, half_x=0.06, half_y=0.05,
        z_lo=0.11, z_hi=0.17, max_tilt_deg=25.0, max_speed=0.05,
    )
    assert bool(ok[0]) is False


def test_placed_false_when_tilted_or_moving():
    tilted = placed_on_stand(
        bar_p=torch.tensor([[-0.15, 0.0, 0.14]]), quat_wxyz=_ROT90X,
        speed=torch.tensor([0.0]), cx=-0.15, cy=0.0, half_x=0.06, half_y=0.05,
        z_lo=0.11, z_hi=0.17, max_tilt_deg=25.0, max_speed=0.05)
    moving = placed_on_stand(
        bar_p=torch.tensor([[-0.15, 0.0, 0.14]]), quat_wxyz=_IDENT,
        speed=torch.tensor([1.0]), cx=-0.15, cy=0.0, half_x=0.06, half_y=0.05,
        z_lo=0.11, z_hi=0.17, max_tilt_deg=25.0, max_speed=0.05)
    assert bool(tilted[0]) is False and bool(moving[0]) is False


def test_lift_stage_zero_on_table():
    s = lift_stage(
        bar_p=torch.tensor([[-0.32, 0.0, 0.02]]), quat_wxyz=_IDENT,
        both_grasped=torch.tensor([False]), placed=torch.tensor([False]),
        z_lift=0.09, max_tilt_deg_lift=30.0)
    assert int(s[0]) == 0


def test_lift_stage_one_when_lifted_level_both_grasped():
    s = lift_stage(
        bar_p=torch.tensor([[-0.30, 0.0, 0.12]]), quat_wxyz=_IDENT,
        both_grasped=torch.tensor([True]), placed=torch.tensor([False]),
        z_lift=0.09, max_tilt_deg_lift=30.0)
    assert int(s[0]) == 1


def test_lift_stage_zero_when_lifted_but_single_arm():
    s = lift_stage(
        bar_p=torch.tensor([[-0.30, 0.0, 0.12]]), quat_wxyz=_IDENT,
        both_grasped=torch.tensor([False]), placed=torch.tensor([False]),
        z_lift=0.09, max_tilt_deg_lift=30.0)
    assert int(s[0]) == 0


def test_lift_stage_two_when_placed_overrides():
    s = lift_stage(
        bar_p=torch.tensor([[-0.15, 0.0, 0.14]]), quat_wxyz=_IDENT,
        both_grasped=torch.tensor([True]), placed=torch.tensor([True]),
        z_lift=0.09, max_tilt_deg_lift=30.0)
    assert int(s[0]) == 2
