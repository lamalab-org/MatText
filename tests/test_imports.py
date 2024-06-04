from mattext.tokenizer import CifTokenizer, CompositionTokenizer, CrysllmTokenizer, NumTokenizer, RobocrysTokenizer, SliceTokenizer, SmilesTokenizer
from mattext.representations import TextRep
from pymatgen.core import Lattice, Structure, Molecule


def test_textrep():
    coords = [[0, 0, 0], [0.75,0.5,0.75]]
    lattice = Lattice.from_parameters(a=3.84, b=3.84, c=3.84, alpha=120,
                                    beta=90, gamma=60)
    struct = Structure(lattice, ["Si", "Si"], coords)

    textrep = TextRep.from_input(struct)

    slices = test_textrep.get_slices()
    assert isinstance(slices, str)