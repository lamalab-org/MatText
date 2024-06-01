from mattext.analysis.xtal2pot import Xtal2Pot
from pymatgen.core import Structure, Lattice


def test_xtal2pot():
    bcc_fe = Structure(Lattice.cubic(2.8), ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    xtal2pot = Xtal2Pot()
    potential = xtal2pot.get_potential(bcc_fe)
    assert isinstance(potential, float)
