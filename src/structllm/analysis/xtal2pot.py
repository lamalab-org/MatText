import functools
import operator
from itertools import combinations
from math import sqrt

import numpy as np
import pandas as pd
import pystow
from pymatgen.core import Structure
from sklearn.neighbors import KDTree

from structllm.utils import structllm_storage

_MIXING_RULES = {}
_ATOM_PARAMETERS = {}
_GEOMETRY_POTENTIALS = {}

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

_ATOM_PARAMETERS["uff_energy"] = dict(
    zip(_df_uff_energy["Element"], _df_uff_energy["energy"])
)

_ATOM_PARAMETERS["uff_xi"] = dict(zip(_df_uff_xi["Element"], _df_uff_xi["xi"]))

_ATOM_PARAMETERS["uff_z"] = dict(zip(_df_uff_z["Element"], _df_uff_z["zi"]))


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


def composition_potential(
    composition: dict,
    mixing_rule: callable,
    atomic_parameters: dict,
    interaction_order: int = 1,
) -> float:
    """E_\\mathrm{comp}=\\sum_{k=1}^k w_k n_k+\\sum_{i_1=1}^N \\sum_{i_2=1}^N \\cdots \\sum_{i_n=1}^N \\sum_{k_1 \neq k_2 \neq \\cdots \neq k_n}^k U_{i_1 i_2 \\cdots i_n}^{\\left(k_1 k_2 \\cdots k_n\right)}

    Args:
        composition (dict): Composition of the structure
        atomic_parameters (dict, optional): Atomic parameters.
        interaction_order (int, optional): Interaction order.
            Interaction order 1 means that the potential energy is the sum of atomic contributions. Interaction order 2 means that the potential energy is the sum of pairwise atomic contributions. Defaults to 1.
    """

    composition_energy = 0

    atom_list = [[atom] * count for atom, count in composition.items()]
    atom_list = functools.reduce(operator.iadd, atom_list, [])
    for combination in combinations(atom_list, interaction_order):
        atomic_parameters_of_combination = [
            atomic_parameters[atom] for atom in combination
        ]

        composition_energy += mixing_rule(atomic_parameters_of_combination)

    return composition_energy


@register("lennard_jones", _GEOMETRY_POTENTIALS)
def lennard_jones(r: float, epsilon: float = 1, sigma: float = 1) -> float:
    return 4.0 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)


def geometry_potential(
    struct, interaction_order: int = 2, potential: callable = lennard_jones
) -> float:
    """Calculate the potential energy of a structure based on its geometry

    Args:
        struct (Structure): pymatgen Structure object
        interaction_order (int, optional): Interaction order.
            Interaction order 2 mean that the nearest neighbor interactions are considered.
            Interaction order 3 means that the nearest and next-nearest neighbor interactions are considered.
            Defaults to 2.
        potential (callable, optional): Potential function to use. Defaults to lennard_jones.

    Returns:
        float: Potential energy of the structure
    """
    coords = struct.cart_coords
    potential_energy = 0
    kd_tree = KDTree(coords)
    for i in range(len(coords)):
        coord = coords[i]
        dist, ind = kd_tree.query(coord.reshape(1, -1), k=interaction_order)

        for dist_ in dist[0]:
            if dist_ > 0:
                potential_energy += potential(dist_)

    return potential_energy


class Xtal2Pot:
    def __init__(
        self,
        mixing_rule: str = "lorentz",
        geometry_potential: str = "lennard_jones",
        composition_interaction_order: int = 1,
        geometry_interaction_order: int = 2,
        atomic_parameter: str = "uff_energy",
    ):
        self.mixing_rule = _MIXING_RULES[mixing_rule]
        self.geometry_potential = _GEOMETRY_POTENTIALS[geometry_potential]
        self.composition_interaction_order = composition_interaction_order
        self.geometry_interaction_order = geometry_interaction_order
        self.atomic_parameter = _ATOM_PARAMETERS[atomic_parameter]

    def get_potential(self, struct: Structure, alpha: float = 0.5) -> float:
        composition_energy = composition_potential(
            struct.composition.as_dict(),
            self.mixing_rule,
            self.atomic_parameter,
            self.composition_interaction_order,
        )

        geometry_energy = geometry_potential(
            struct, self.geometry_interaction_order, self.geometry_potential
        )

        return alpha * composition_energy + (1 - alpha) * geometry_energy
