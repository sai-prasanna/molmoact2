import torch

from sim_eval.tasks.success_logic import object_in_footprint, blocks_stacked


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
