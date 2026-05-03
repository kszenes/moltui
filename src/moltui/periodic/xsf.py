from __future__ import annotations

from pathlib import Path

import numpy as np

from moltui.elements import Atom, Molecule, get_element, get_element_by_number


def parse_xsf(filepath: str | Path) -> Molecule:
    """Parse an XSF structure containing PRIMVEC and PRIMCOORD blocks."""
    lines = Path(filepath).read_text().splitlines()
    lattice: np.ndarray | None = None
    atoms: list[Atom] = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        upper = stripped.upper()
        if not stripped or stripped.startswith("#"):
            i += 1
            continue
        if upper == "PRIMVEC":
            if i + 3 >= len(lines):
                raise ValueError("XSF PRIMVEC block is incomplete")
            lattice = np.array(
                [[float(tok) for tok in lines[i + j].split()[:3]] for j in range(1, 4)],
                dtype=np.float64,
            )
            i += 4
            continue
        if upper == "PRIMCOORD":
            if i + 1 >= len(lines):
                raise ValueError("XSF PRIMCOORD block is incomplete")
            header = lines[i + 1].split()
            if not header:
                raise ValueError("XSF PRIMCOORD missing atom count")
            n_atoms = int(header[0])
            start = i + 2
            if len(lines) < start + n_atoms:
                raise ValueError("XSF PRIMCOORD ended before all atoms were read")
            atoms = []
            for line in lines[start : start + n_atoms]:
                parts = line.split()
                if len(parts) < 4:
                    raise ValueError("Invalid XSF atom line")
                token = parts[0]
                try:
                    element = get_element_by_number(int(token))
                except ValueError:
                    element = get_element(token)
                pos = np.array(
                    [float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64
                )
                atoms.append(Atom(element=element, position=pos))
            i = start + n_atoms
            continue
        i += 1

    if not atoms:
        raise ValueError("No atoms found in XSF file")
    mol = Molecule(atoms=atoms, bonds=[], lattice=lattice)
    mol.detect_bonds_auto()
    return mol
