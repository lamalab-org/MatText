import numpy as np
from pymatgen.core import Structure
from math import sqrt
from itertools import combinations
from sklearn.neighbors import KDTree
from mp_api.client import MPRester
import pandas as pd
import os
from structllm.utils import structllm_storage
import pystow

_MIXING_RULES = {}
_ATOM_PARAMETERS = {}

_df_uff_energy: pd.DataFrame = pystow.ensure_csv(
    structllm_storage,
    url="https://www.dropbox.com/scl/fi/fnla7a64yjpk9ca65d69y/element_to_energy.csv?rlkey=d97gp2idkbgdemncv1cys7hye&dl=1",
)

_df_uff_xi: pd.DataFrame = pystow.ensure_csv(
    structllm_storage,
    url="https://www.dropbox.com/scl/fi/hz7wdi1ug31gdfddxb600/element_to_xi.csv?rlkey=wdwtowjx2jyp9m636ed4z60ki&dl=1",
)

_df_uff_z: pd.DataFrame = pystow.ensure_csv(
    structllm_storage,
    url="https://www.dropbox.com/scl/fi/ieqcfyw6g5zsa4memcb5f/element_to_zi.csv?rlkey=nn307r5k0h4yuph6pjmk44omr&dl=1",
)


def register(name: str, store: dict):
    """
    Decorator to register a mixing rule function
    """

    def add_to_dict(func):
        store[name] = func
        return func

    return add_to_dict


@register("lorentz", _MIXING_RULES)
def lorentz_mixing(parameters: np.array) -> float:
    return np.mean(parameters)


@register("berthelot", _MIXING_RULES)
def berthelot_mixing(parameters: np.array) -> float:
    return sqrt(np.prod(parameters))


@register("dender_halsey", _MIXING_RULES)
def dender_halsey_mixing(parameters: np.array) -> float:
    return 2 * np.mean(parameters) / np.sum(parameters)


@register("average_of_squares", _MIXING_RULES)
def average_of_squares_mixing(parameters: np.array) -> float:
    return np.mean(parameters**2)
