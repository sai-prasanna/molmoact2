# sim_eval/tasks/pickplace_tasks/utensil_tray.py
"""Utensil-tray task: place fork, spoon, and knife into a compartmented tray.

Uses the base _build_tray container (kind="tray") and the default
center-in-footprint success. The tray's interior dividers are physical structure
for fidelity; success only requires each utensil's center inside the tray
footprint (piling-tolerant, like the box tasks).
"""

from mani_skill.utils.registration import register_env

from ..pickplace_base import BimanualPickPlaceEnv

_TRAY_WHITE = [0.9, 0.9, 0.9, 1.0]


@register_env("BimanualUtensilTray-v1", max_episode_steps=1200)
class BimanualUtensilTrayEnv(BimanualPickPlaceEnv):
    """Place the fork, spoon, and knife into the tray."""

    OBJECTS = [
        ("030_fork",  (-0.30, 0.22), "left"),
        ("031_spoon", (-0.30, -0.18), "right"),
        ("032_knife", (-0.24, -0.30), "right"),
    ]
    CONTAINER = {
        "kind": "tray",
        "inner_half": (0.10, 0.13),   # (hx, hy): long along y to hold utensils
        "pos": (-0.13, 0.0),
        "height": 0.04,
        "wall": 0.008,
        "color": _TRAY_WHITE,
        "n_compartments": 3,
        "divider": 0.004,
    }
    INSTRUCTION = "Place the fork, spoon, and knife into the tray."
