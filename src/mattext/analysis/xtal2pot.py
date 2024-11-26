import functools
import operator
from itertools import combinations
from math import sqrt
from typing import Callable

import numpy as np
import pandas as pd
from pymatgen.core import Structure
from sklearn.neighbors import KDTree

from mattext.utils import mattext_storage

_MIXING_RULES = {}
_ATOM_PARAMETERS = {}
_GEOMETRY_POTENTIALS = {}

_df_uff_energy: pd.DataFrame = mattext_storage.ensure_csv(
    url="https://www.dropbox.com/scl/fi/fnla7a64yjpk9ca65d69y/element_to_energy.csv?rlkey=d97gp2idkbgdemncv1cys7hye&dl=1",
    read_csv_kwargs={"sep": ","},
)

_df_uff_xi: pd.DataFrame = mattext_storage.ensure_csv(
    url="https://www.dropbox.com/scl/fi/hz7wdi1ug31gdfddxb600/element_to_xi.csv?rlkey=wdwtowjx2jyp9m636ed4z60ki&dl=1",
    read_csv_kwargs={"sep": ","},
)

_df_uff_z: pd.DataFrame = mattext_storage.ensure_csv(
    url="https://www.dropbox.com/scl/fi/ieqcfyw6g5zsa4memcb5f/element_to_zi.csv?rlkey=nn307r5k0h4yuph6pjmk44omr&dl=1",
    read_csv_kwargs={"sep": ","},
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


def get_atomic_parameters(
    atomic_parameters: dict,
    atom: str,
):
    try:
        return atomic_parameters[atom]
    except KeyError:
        return np.mean(list(atomic_parameters.values()))


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

    atom_list = [[atom] * int(count) for atom, count in composition.items()]
    atom_list = functools.reduce(operator.iadd, atom_list, [])
    for combination in combinations(atom_list, interaction_order):
        atomic_parameters_of_combination = [
            get_atomic_parameters(atomic_parameters, atom) for atom in combination
        ]
        composition_energy += mixing_rule(atomic_parameters_of_combination)

    return composition_energy


@register("lennard_jones", _GEOMETRY_POTENTIALS)
def lennard_jones(r: float, epsilon: float = 1, sigma: float = 1) -> float:
    return 4.0 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)


@register("rasp_potential", _GEOMETRY_POTENTIALS)
def rasp_geometric_potential(
    x_coords: np.ndarray, y_coords: np.ndarray, z_coords: np.ndarray
) -> float:
    """
    RASP-compatible geometric potential using only attention-like operations.
    Robust version that handles array indexing safely.
    """

    def kqv(k: np.ndarray, q: np.ndarray, v: np.ndarray, pred) -> np.ndarray:
        """
        Simulates attention-like mechanism with safe indexing.
        Now passes indices instead of values to pred function.
        """
        s = len(k)
        A = np.zeros((s, s), dtype=bool)

        # Pass indices to pred instead of values
        for i in range(s):
            for j in range(s):
                A[i, j] = pred(j, i)  # passing indices instead of values

        # return np.dot(A, v)
        out = np.dot(A, v)
        norm = np.dot(A, np.ones(len(A)))
        return np.divide(out, norm, out=np.zeros_like(v), where=(norm != 0))

    def get_pair_energy(idx1: int, idx2: int) -> float:
        """
        Compute pairwise energy using array indices instead of direct values.
        """
        distance_potential = {
            1: 3.0,  # very close - highly repulsive
            2: 2.0,  # repulsive
            3: -4.0,  # attractive (equilibrium)
            4: -3.0,  # weakly attractive
            5: -2.0,  # negligible interaction
            6: -2.0,  # no interaction
            7: -1.0,
            8: 0.0,
            9: 0.0,
            10: 0.0,
            11: 0.0,
        }

        # Safely compute distances using indices
        dx = abs(x_coords[idx1] - x_coords[idx2])
        dy = abs(y_coords[idx1] - y_coords[idx2])
        dz = abs(z_coords[idx1] - z_coords[idx2])

        dist = dx + dy + dz
        return distance_potential.get(min(dist, 8), 0.0)

    # Ensure inputs are numpy arrays and properly shaped
    x_coords = np.asarray(x_coords, dtype=np.int32)
    y_coords = np.asarray(y_coords, dtype=np.int32)
    z_coords = np.asarray(z_coords, dtype=np.int32)

    if len(x_coords) == 0:
        return 0.0

    # First attention operation - compute pairwise energies
    # Now passing indices to get_pair_energy through the pred function
    energies = kqv(
        k=np.arange(len(x_coords)),  # use indices instead of coordinates
        q=np.arange(len(x_coords)),
        v=np.ones(len(x_coords), dtype=float),
        pred=lambda idx_k, idx_q: get_pair_energy(
            idx_k, idx_q
        ),  # clip between -10 and 10,
    )

    if len(energies) == 0:
        return 0.0
    total_energy = kqv(
        k=np.arange(len(x_coords)),
        q=np.ones(len(x_coords)),  # Query all positions
        v=energies,
        pred=lambda k, q: True,  # Include all interactions
    )
    return float(total_energy[0])


def geometry_potential(
    struct: Structure, interaction_order: int = 2, potential: Callable = lennard_jones
) -> float:
    """Calculate the potential energy of a structure based on its geometry."""

    def to_int_coords(coords: np.ndarray) -> np.ndarray:
        """Safely convert coordinates to integer representation."""
        return np.clip((coords * 9).astype(np.int32), -127, 127)

    if potential == rasp_geometric_potential:
        fractional_coords = struct.frac_coords
        x_coords = to_int_coords(fractional_coords[:, 0])
        y_coords = to_int_coords(fractional_coords[:, 1])
        z_coords = to_int_coords(fractional_coords[:, 2])
        return potential(x_coords, y_coords, z_coords)

    # Original implementation for other potentials
    coords = struct.cart_coords
    potential_energy = 0.0
    kd_tree = KDTree(coords)

    for i in range(len(coords)):
        coord = coords[i]
        dist, ind = kd_tree.query(
            coord.reshape(1, -1), k=min(interaction_order, len(coords))
        )

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

    def get_potential(self, struct: Structure) -> float:
        composition_energy = composition_potential(
            struct.composition.as_dict(),
            self.mixing_rule,
            self.atomic_parameter,
            self.composition_interaction_order,
        )

        geometry_energy = geometry_potential(
            struct, self.geometry_interaction_order, self.geometry_potential
        )
        return composition_energy, geometry_energy

    def get_total_energy(self, struct: Structure, alphas: list[float] = [0.5]):
        energies = {}
        for alpha in alphas:
            composition_energy, geometry_energy = self.get_potential(struct)
            total_energy = alpha * composition_energy + (1 - alpha) * geometry_energy
            energies[alpha] = total_energy
        return energies
