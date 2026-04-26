#!/usr/bin/env python3
import trexio as t
from pyscf import gto, scf
from pyscf.tools import molden
from pyscf.tools import trexio as pyscf_trexio

mol = gto.M(atom="N 0 0 0; N 0 0 1.098", basis="cc-pvqz", symmetry=True)
mf = scf.RHF(mol).run()

# Extract MO symmetry labels from PySCF (e.g. "A1g", "E1ux", ...)
irrep_map = dict(zip(mol.irrep_id, mol.irrep_name))
orbsym = [irrep_map[i] for i in mf.orbsym]

# Write TREXIO containers
pyscf_trexio.to_trexio(mf, "n2", backend="text")
pyscf_trexio.to_trexio(mf, "n2.h5", backend="hdf5")

# Append symmetry labels (not written by pyscf_trexio.to_trexio)
for path, backend in [("n2", t.TREXIO_TEXT), ("n2.h5", t.TREXIO_HDF5)]:
    with t.File(path, "u", back_end=backend) as f:
        t.write_mo_symmetry(f, list(orbsym))

molden.from_scf(mf, "n2.molden")
