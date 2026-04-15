from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .elements import Atom, Molecule, get_element
from .parsers import BOHR_TO_ANGSTROM, CubeData


@dataclass
class MoldenData:
    molecule: Molecule
    mo_energies: np.ndarray
    mo_occupations: np.ndarray
    mo_symmetries: list[str]
    n_mos: int
    homo_idx: int
    _pyscf_mol: Any = field(repr=False)
    _mo_coeff: np.ndarray = field(repr=False)


def parse_molden_atoms(filepath: str | Path) -> Molecule:
    """Parse just the atoms from a molden file (no pyscf needed)."""
    filepath = Path(filepath)
    atoms = []
    in_atoms = False
    is_angstrom = False

    with open(filepath) as f:
        for line in f:
            if "[Atoms]" in line:
                in_atoms = True
                is_angstrom = "Angs" in line
                continue
            if line.startswith("[") and in_atoms:
                break
            if in_atoms and line.strip():
                parts = line.split()
                symbol = parts[0]
                x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
                if not is_angstrom:
                    x *= BOHR_TO_ANGSTROM
                    y *= BOHR_TO_ANGSTROM
                    z *= BOHR_TO_ANGSTROM
                atoms.append(Atom(element=get_element(symbol), position=np.array([x, y, z])))

    mol = Molecule(atoms=atoms, bonds=[])
    mol.detect_bonds()
    return mol


def load_molden_data(filepath: str | Path) -> MoldenData:
    """Load full molden data including MO coefficients (requires pyscf)."""
    try:
        from pyscf.tools import molden as molden_tools
    except ImportError:
        raise ImportError(
            "pyscf is required for orbital evaluation from molden files. "
            "Install with: pip install pyscf"
        )

    filepath = Path(filepath)
    result = molden_tools.load(str(filepath))
    pyscf_mol, mo_energy, mo_coeff, mo_occ = result[0], result[1], result[2], result[3]
    irrep_labels = result[4] if len(result) > 4 else []

    # Convert atoms
    atoms = []
    for i in range(pyscf_mol.natm):
        symbol = pyscf_mol.atom_pure_symbol(i)
        coord = pyscf_mol.atom_coord(i) * BOHR_TO_ANGSTROM
        atoms.append(Atom(element=get_element(symbol), position=coord))

    molecule = Molecule(atoms=atoms, bonds=[])
    molecule.detect_bonds()

    # Find HOMO
    occ_indices = np.where(mo_occ > 0.5)[0]
    homo_idx = int(occ_indices[-1]) if len(occ_indices) > 0 else 0

    return MoldenData(
        molecule=molecule,
        mo_energies=np.asarray(mo_energy),
        mo_occupations=np.asarray(mo_occ),
        mo_symmetries=list(irrep_labels) if irrep_labels is not None else [],
        n_mos=mo_coeff.shape[1],
        homo_idx=homo_idx,
        _pyscf_mol=pyscf_mol,
        _mo_coeff=np.asarray(mo_coeff),
    )


def evaluate_mo(
    molden_data: MoldenData,
    mo_index: int,
    grid_shape: tuple[int, int, int] = (60, 60, 60),
    padding: float = 5.0,
) -> CubeData:
    """Evaluate a molecular orbital on a 3D grid. Returns CubeData."""
    mol = molden_data._pyscf_mol
    coords = mol.atom_coords()  # Bohr

    padding_bohr = padding / BOHR_TO_ANGSTROM
    min_c = coords.min(axis=0) - padding_bohr
    max_c = coords.max(axis=0) + padding_bohr

    nx, ny, nz = grid_shape
    x = np.linspace(min_c[0], max_c[0], nx)
    y = np.linspace(min_c[1], max_c[1], ny)
    z = np.linspace(min_c[2], max_c[2], nz)

    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    grid_points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    ao = mol.eval_gto("GTOval_sph", grid_points)
    mo_vec = molden_data._mo_coeff[:, mo_index]
    mo_vals = ao @ mo_vec

    origin = min_c  # Bohr
    axes = np.diag(
        [
            (max_c[0] - min_c[0]) / (nx - 1),
            (max_c[1] - min_c[1]) / (ny - 1),
            (max_c[2] - min_c[2]) / (nz - 1),
        ]
    )

    return CubeData(
        molecule=molden_data.molecule,
        origin=origin,
        axes=axes,
        n_points=grid_shape,
        data=mo_vals.reshape(grid_shape),
    )
