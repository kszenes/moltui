from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .elements import Atom, Molecule, get_element, get_element_by_number

BOHR_TO_ANGSTROM = 0.529177249


@dataclass
class CubeData:
    molecule: Molecule
    origin: np.ndarray  # (3,) in Bohr
    axes: np.ndarray  # (3, 3) step vectors in Bohr
    n_points: tuple[int, int, int]
    data: np.ndarray  # (n1, n2, n3) volumetric data


def parse_xyz(filepath: str | Path) -> Molecule:
    filepath = Path(filepath)
    with open(filepath) as f:
        lines = f.readlines()

    n_atoms = int(lines[0].strip())
    # line 1 is comment, skip
    atoms = []
    for line in lines[2 : 2 + n_atoms]:
        parts = line.split()
        symbol = parts[0]
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        atoms.append(Atom(element=get_element(symbol), position=np.array([x, y, z])))

    mol = Molecule(atoms=atoms, bonds=[])
    mol.detect_bonds()
    return mol


def parse_cube(filepath: str | Path) -> Molecule:
    cube_data = parse_cube_data(filepath)
    return cube_data.molecule


def parse_cube_data(filepath: str | Path) -> CubeData:
    filepath = Path(filepath)
    with open(filepath) as f:
        # Lines 0-1: comments
        f.readline()
        f.readline()

        # Line 2: n_atoms, origin
        parts = f.readline().split()
        raw_natoms = int(parts[0])
        n_atoms = abs(raw_natoms)
        has_mo = raw_natoms < 0
        origin = np.array([float(parts[1]), float(parts[2]), float(parts[3])])

        # Lines 3-5: grid dimensions and step vectors
        n_points = []
        axes = np.zeros((3, 3))
        for i in range(3):
            parts = f.readline().split()
            n_points.append(int(parts[0]))
            axes[i] = [float(parts[1]), float(parts[2]), float(parts[3])]

        # Atom lines
        atoms = []
        for _ in range(n_atoms):
            parts = f.readline().split()
            atomic_number = int(parts[0])
            x = float(parts[2]) * BOHR_TO_ANGSTROM
            y = float(parts[3]) * BOHR_TO_ANGSTROM
            z = float(parts[4]) * BOHR_TO_ANGSTROM
            atoms.append(
                Atom(
                    element=get_element_by_number(atomic_number),
                    position=np.array([x, y, z]),
                )
            )

        # Skip MO line if present
        if has_mo:
            f.readline()

        # Read all remaining data
        data_text = f.read()

    values = np.array(data_text.split(), dtype=np.float64)
    data = values.reshape(n_points[0], n_points[1], n_points[2])

    mol = Molecule(atoms=atoms, bonds=[])
    mol.detect_bonds()

    return CubeData(
        molecule=mol,
        origin=origin,
        axes=axes,
        n_points=(n_points[0], n_points[1], n_points[2]),
        data=data,
    )


def load_molecule(filepath: str | Path) -> Molecule:
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()
    if suffix == ".xyz":
        return parse_xyz(filepath)
    elif suffix == ".cube":
        return parse_cube(filepath)
    elif suffix == ".molden":
        from .molden import parse_molden_atoms

        return parse_molden_atoms(filepath)
    else:
        raise ValueError(
            f"Unsupported file format: {suffix}. Use .xyz, .cube, or .molden"
        )
