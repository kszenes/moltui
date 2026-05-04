from __future__ import annotations

from pathlib import Path

import numpy as np

from ..elements import Atom, Molecule
from ..parser_utils import read_scalar_grid_values, resolve_element_token
from ..volumetric import VolumetricData


def _parse_structure_lines(lines: list[str]) -> Molecule:
    lattice: np.ndarray | None = None
    atoms: list[Atom] = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        upper = stripped.upper()
        if not stripped or stripped.startswith("#"):
            i += 1
            continue
        keyword = upper.split()[0]
        if keyword == "PRIMVEC":
            if i + 3 >= len(lines):
                raise ValueError("XSF PRIMVEC block is incomplete")
            lattice = np.array(
                [[float(tok) for tok in lines[i + j].split()[:3]] for j in range(1, 4)],
                dtype=np.float64,
            )
            i += 4
            continue
        if keyword == "PRIMCOORD":
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
                element = resolve_element_token(parts[0])
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


def parse_xsf(filepath: str | Path) -> Molecule:
    """Parse an XSF structure containing PRIMVEC and PRIMCOORD blocks."""
    return _parse_structure_lines(Path(filepath).read_text().splitlines())


def parse_xsf_volumetric_data(filepath: str | Path) -> VolumetricData:
    """Parse the first XSF 3D datagrid along with the structure."""
    lines = Path(filepath).read_text().splitlines()
    molecule = _parse_structure_lines(lines)

    i = 0
    while i < len(lines):
        upper = lines[i].strip().upper()
        if upper.startswith("DATAGRID_3D") or upper.startswith("BEGIN_DATAGRID_3D"):
            if i + 5 >= len(lines):
                raise ValueError("XSF DATAGRID_3D block is incomplete")
            try:
                n_points = tuple(int(tok) for tok in lines[i + 1].split()[:3])
                if len(n_points) != 3:
                    raise ValueError
                origin = np.array([float(tok) for tok in lines[i + 2].split()[:3]])
                spans = np.array(
                    [[float(tok) for tok in lines[i + j].split()[:3]] for j in range(3, 6)],
                    dtype=np.float64,
                )
            except ValueError as exc:
                raise ValueError("Invalid XSF DATAGRID_3D header") from exc

            try:
                data, _ = read_scalar_grid_values(
                    lines,
                    i + 6,
                    n_points,
                    stop_prefixes=("END_DATAGRID_3D",),
                )
            except ValueError as exc:
                raise ValueError(
                    "XSF DATAGRID_3D ended before all scalar values were read"
                ) from exc
            axes = np.array([spans[k] / max(n_points[k] - 1, 1) for k in range(3)])
            return VolumetricData(
                molecule=molecule,
                origin=origin,
                axes=axes,
                n_points=n_points,  # type: ignore[arg-type]
                data=data,
                periodic=molecule.lattice is not None,
            )
        i += 1

    raise ValueError("No XSF DATAGRID_3D block found")
