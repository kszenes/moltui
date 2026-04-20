from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .elements import Atom, Molecule, get_element
from .gto import MoldenBasis, eval_gto, parse_molden
from .parsers import BOHR_TO_ANGSTROM, CubeData


@dataclass
class MoldenData:
    molecule: Molecule
    mo_energies: np.ndarray
    mo_occupations: np.ndarray
    mo_symmetries: list[str]
    mo_spins: list[str]
    n_mos: int
    homo_idx: int
    _basis: MoldenBasis = field(repr=False)


def parse_molden_atoms(filepath: str | Path) -> Molecule:
    """Parse just the atoms from a molden file."""
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
    """Load full molden data including MO coefficients."""
    filepath = Path(filepath)
    basis = parse_molden(filepath)
    file_text = filepath.read_text().lower()

    # Reject vibrational-only Molden files up front with a clear message.
    has_normal_modes = any(
        tag in file_text for tag in ("[fr-coord]", "[fr-norm-coord]", "[freq]", "[n_freq]")
    )
    has_required_orbital_sections = all(tag in file_text for tag in ("[atoms]", "[gto]", "[mo]"))
    if has_normal_modes and not has_required_orbital_sections:
        raise ValueError(
            "This Molden file only contains normal modes and not geometries or orbitals."
        )

    # Build Molecule
    atoms = []
    for i, symbol in enumerate(basis.atom_symbols):
        coord = basis.atom_coords_bohr[i] * BOHR_TO_ANGSTROM
        atoms.append(Atom(element=get_element(symbol), position=coord))

    molecule = Molecule(atoms=atoms, bonds=[])
    molecule.detect_bonds()

    # Find HOMO
    occ_indices = np.where(basis.mo_occupations > 0.5)[0]
    homo_idx = int(occ_indices[-1]) if len(occ_indices) > 0 else 0

    return MoldenData(
        molecule=molecule,
        mo_energies=basis.mo_energies,
        mo_occupations=basis.mo_occupations,
        mo_symmetries=basis.mo_symmetries,
        mo_spins=basis.mo_spins,
        n_mos=basis.mo_coefficients.shape[1],
        homo_idx=homo_idx,
        _basis=basis,
    )


def evaluate_mo(
    molden_data: MoldenData,
    mo_index: int,
    grid_shape: tuple[int, int, int] = (60, 60, 60),
    padding: float = 5.0,
) -> CubeData:
    """Evaluate a molecular orbital on a 3D grid. Returns CubeData."""
    basis = molden_data._basis
    coords = basis.atom_coords_bohr

    padding_bohr = padding / BOHR_TO_ANGSTROM
    min_c = coords.min(axis=0) - padding_bohr
    max_c = coords.max(axis=0) + padding_bohr

    nx, ny, nz = grid_shape
    x = np.linspace(min_c[0], max_c[0], nx)
    y = np.linspace(min_c[1], max_c[1], ny)
    z = np.linspace(min_c[2], max_c[2], nz)

    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    grid_points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    ao = eval_gto(basis.shells, grid_points, basis.spherical)
    mo_vec = basis.mo_coefficients[:, mo_index]
    mo_vals = ao @ mo_vec

    origin = min_c
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
