from pyscf import gto, scf
from pyscf.tools import molden, trexio

mol = gto.M(atom="N 0 0 0; N 0 0 1.098", basis="cc-pvqz", symmetry=True)
mf = scf.RHF(mol).run()

trexio.to_trexio(mf, "n2", backend="text")
trexio.to_trexio(mf, "n2.h5", backend="hdf5")
molden.from_scf(mf, "n2.molden")
