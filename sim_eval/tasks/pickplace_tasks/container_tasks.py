# sim_eval/tasks/pickplace_tasks/container_tasks.py
"""Container pick-place tasks: place a set of YCB objects into one container.

Each is a thin subclass of BimanualPickPlaceEnv. Anchor/container values are
starting points tuned on GPU smoke (object reachability, non-overlap, and
large-object fit — the cracker box in particular is large).
"""

from mani_skill.utils.registration import register_env

from ..pickplace_base import BimanualPickPlaceEnv

# Reusable container presets (tune on GPU smoke).
_BLACK = [0.05, 0.05, 0.05, 1.0]
_BROWN = [0.55, 0.35, 0.18, 1.0]
_GRAY = [0.6, 0.6, 0.62, 1.0]


@register_env("BimanualBox-v1", max_episode_steps=1200)
class BimanualBoxEnv(BimanualPickPlaceEnv):
    """Place a can and three boxes into the black box."""
    OBJECTS = [
        ("002_master_chef_can", (-0.34, 0.20), "left"),
        ("004_sugar_box",       (-0.24, 0.28), "left"),
        ("003_cracker_box",     (-0.34, -0.20), "right"),
        ("008_pudding_box",     (-0.24, -0.28), "right"),
    ]
    CONTAINER = {"kind": "box", "inner_half": 0.15, "pos": (-0.12, 0.0),
                 "height": 0.07, "wall": 0.01, "color": _BLACK}
    INSTRUCTION = "Place the can, cracker box, sugar box, and pudding box into the black box."


@register_env("BimanualToy-v1", max_episode_steps=1200)
class BimanualToyEnv(BimanualPickPlaceEnv):
    """Place three toys into the box."""
    OBJECTS = [
        ("077_rubiks_cube",   (-0.30, 0.22), "left"),
        ("073-a_lego_duplo",  (-0.30, -0.18), "right"),
        ("072-a_toy_airplane",(-0.24, -0.30), "right"),
    ]
    CONTAINER = {"kind": "box", "inner_half": 0.12, "pos": (-0.13, 0.0),
                 "height": 0.07, "wall": 0.01, "color": _BROWN}
    INSTRUCTION = "Place the Rubik's cube, the lego block, and the toy airplane into the box."


@register_env("BimanualCup-v1", max_episode_steps=1200)
class BimanualCupEnv(BimanualPickPlaceEnv):
    """Place two cups, then a fork and spoon, into the container."""
    OBJECTS = [
        ("065-a_cups", (-0.32, 0.20), "left"),
        ("065-b_cups", (-0.24, 0.28), "left"),
        ("030_fork",   (-0.32, -0.20), "right"),
        ("031_spoon",  (-0.24, -0.28), "right"),
    ]
    CONTAINER = {"kind": "box", "inner_half": 0.12, "pos": (-0.13, 0.0),
                 "height": 0.06, "wall": 0.01, "color": _GRAY}
    INSTRUCTION = "Place the cups, then the fork and spoon, into the container."


@register_env("BimanualMed-v1", max_episode_steps=1200)
class BimanualMedEnv(BimanualPickPlaceEnv):
    """Place the medicine boxes into the container."""
    OBJECTS = [
        ("009_gelatin_box", (-0.30, 0.22), "left"),
        ("008_pudding_box", (-0.30, -0.22), "right"),
    ]
    CONTAINER = {"kind": "box", "inner_half": 0.11, "pos": (-0.13, 0.0),
                 "height": 0.06, "wall": 0.01, "color": _GRAY}
    INSTRUCTION = "Place the medicine boxes into the container."


@register_env("BimanualTool-v1", max_episode_steps=1200)
class BimanualToolEnv(BimanualPickPlaceEnv):
    """Place the screwdriver, wrench, clamp, and scissors into the black box."""
    OBJECTS = [
        ("043_phillips_screwdriver", (-0.34, 0.20), "left"),
        ("050_medium_clamp",         (-0.24, 0.28), "left"),
        ("042_adjustable_wrench",    (-0.34, -0.20), "right"),
        ("037_scissors",             (-0.24, -0.28), "right"),
    ]
    CONTAINER = {"kind": "box", "inner_half": 0.14, "pos": (-0.12, 0.0),
                 "height": 0.06, "wall": 0.01, "color": _BLACK}
    INSTRUCTION = "Place the screwdriver, wrench, clamp, and scissors into the black box."


@register_env("BimanualGrocery-v1", max_episode_steps=1200)
class BimanualGroceryEnv(BimanualPickPlaceEnv):
    """Place the cracker box, soup can, and sugar box into the basket."""
    OBJECTS = [
        ("003_cracker_box",     (-0.32, 0.22), "left"),
        ("005_tomato_soup_can", (-0.32, -0.18), "right"),
        ("004_sugar_box",       (-0.24, -0.30), "right"),
    ]
    CONTAINER = {"kind": "box", "inner_half": 0.15, "pos": (-0.12, 0.0),
                 "height": 0.07, "wall": 0.01, "color": _BROWN}
    INSTRUCTION = "Place the cracker box, soup can, and sugar box into the basket."
