from invcryrep.invcryrep import InvCryRep


def give_slice(structure):
    backend = InvCryRep(check_results=True)
    return backend.structure2SLICES(structure)
