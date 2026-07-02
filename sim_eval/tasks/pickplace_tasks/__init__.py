"""MolmoAct2 pick-place task suite. Importing each module registers its env."""

from .container_tasks import (
    BimanualBoxEnv,
    BimanualToyEnv,
    BimanualCupEnv,
    BimanualMedEnv,
    BimanualToolEnv,
    BimanualGroceryEnv,
)
from .block_stack import BimanualBlockStackEnv
from .utensil_tray import BimanualUtensilTrayEnv
from .lift_place import BimanualLiftPlaceStandEnv

__all__ = [
    "BimanualBoxEnv",
    "BimanualToyEnv",
    "BimanualCupEnv",
    "BimanualMedEnv",
    "BimanualToolEnv",
    "BimanualGroceryEnv",
    "BimanualBlockStackEnv",
    "BimanualUtensilTrayEnv",
    "BimanualLiftPlaceStandEnv",
]
