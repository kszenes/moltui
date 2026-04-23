# for now requires installation of pyscf-forge
from pyscf import gto, scf
from pyscf.tools import cubegen, molden

mol = gto.M(
    atom="""
    C   0.000   1.396   0.000
    C   1.209   0.698   0.000
    C   1.209  -0.698   0.000
    C   0.000  -1.396   0.000
    C  -1.209  -0.698   0.000
    C  -1.209   0.698   0.000
    H   0.000   2.479   0.000
    H   2.147   1.240   0.000
    H   2.147  -1.240   0.000
    H   0.000  -2.479   0.000
    H  -2.147  -1.240   0.000
    H  -2.147   1.240   0.000
    """,
    basis="sto-3g",
    symmetry=True,
)

mf = scf.RHF(mol).run()

# compare 12th mo
mo = 12

with open("benzene.molden", "w") as f:
    molden.header(mol, f)
    molden.orbital_coeff(mol, f, mf.mo_coeff, ene=mf.mo_energy, occ=mf.mo_occ)

cubegen.orbital(mol, f"benzene_mo_{mo}.cube", mf.mo_coeff[:, mo])
