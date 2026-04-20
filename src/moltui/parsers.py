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


def parse_xyz_trajectory(filepath: str | Path) -> list[Molecule]:
    """Parse an XYZ file with one or more frames. Returns one Molecule per frame.

    Accepts the standard multi-frame XYZ format: each frame is
    <atom_count>\\n<comment>\\n<atom_count lines of "SYMBOL X Y Z">, concatenated.
    """
    filepath = Path(filepath)
    with open(filepath) as f:
        lines = f.readlines()

    frames: list[Molecule] = []
    i = 0
    n = len(lines)
    while i < n:
        # Skip blank lines between or trailing frames
        if not lines[i].strip():
            i += 1
            continue
        n_atoms = int(lines[i].strip())
        # lines[i+1] is comment, skip
        atoms = []
        for line in lines[i + 2 : i + 2 + n_atoms]:
            parts = line.split()
            symbol = parts[0]
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            atoms.append(Atom(element=get_element(symbol), position=np.array([x, y, z])))
        mol = Molecule(atoms=atoms, bonds=[])
        mol.detect_bonds()
        frames.append(mol)
        i += 2 + n_atoms
    return frames


def parse_xyz(filepath: str | Path) -> Molecule:
    return parse_xyz_trajectory(filepath)[0]


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


def _zmat_to_cartesian(
    symbols: list[str],
    refs: list[tuple[int, ...]],
    values: list[tuple[float, ...]],
) -> list[np.ndarray]:
    """Convert Z-matrix internal coordinates to Cartesian positions.

    Each entry in refs/values corresponds to the atom at that index:
      atom 0: no refs/values (placed at origin)
      atom 1: (ref_atom,) / (distance,)
      atom 2: (ref_atom, angle_atom) / (distance, angle_deg)
      atom 3+: (ref_atom, angle_atom, dihedral_atom) / (distance, angle_deg, dihedral_deg)
    """
    coords: list[np.ndarray] = []
    for i in range(len(symbols)):
        if i == 0:
            coords.append(np.array([0.0, 0.0, 0.0]))
        elif i == 1:
            r = values[i][0]
            coords.append(np.array([r, 0.0, 0.0]))
        elif i == 2:
            r = values[i][0]
            angle = np.radians(values[i][1])
            ref_a = refs[i][0]
            ref_b = refs[i][1]
            # Place along the ref_a -> ref_b direction, rotated by angle
            d = coords[ref_a] - coords[ref_b]
            d_norm = d / (np.linalg.norm(d) + 1e-15)
            # Pick a perpendicular vector
            if abs(d_norm[1]) < 0.9:
                perp = np.cross(d_norm, np.array([0.0, 1.0, 0.0]))
            else:
                perp = np.cross(d_norm, np.array([1.0, 0.0, 0.0]))
            perp /= np.linalg.norm(perp) + 1e-15
            pos = coords[ref_a] + r * (-d_norm * np.cos(angle) + perp * np.sin(angle))
            coords.append(pos)
        else:
            r = values[i][0]
            angle = np.radians(values[i][1])
            dihedral = np.radians(values[i][2])
            ref_a = refs[i][0]  # bonded to this atom
            ref_b = refs[i][1]  # angle vertex
            ref_c = refs[i][2]  # dihedral reference

            ab = coords[ref_b] - coords[ref_a]
            ab /= np.linalg.norm(ab) + 1e-15
            bc = coords[ref_c] - coords[ref_b]

            # Build local frame: n = ab direction, d2 perpendicular in abc plane
            n = ab
            bc_perp = bc - np.dot(bc, n) * n
            bc_perp_norm = np.linalg.norm(bc_perp)
            if bc_perp_norm < 1e-10:
                # Degenerate: pick arbitrary perpendicular
                if abs(n[1]) < 0.9:
                    d2 = np.cross(n, np.array([0.0, 1.0, 0.0]))
                else:
                    d2 = np.cross(n, np.array([1.0, 0.0, 0.0]))
                d2 /= np.linalg.norm(d2)
            else:
                d2 = bc_perp / bc_perp_norm
            d3 = np.cross(n, d2)

            pos = coords[ref_a] + r * (
                -n * np.cos(angle)
                + d2 * np.sin(angle) * np.cos(dihedral)
                + d3 * np.sin(angle) * np.sin(dihedral)
            )
            coords.append(pos)
    return coords


def parse_zmat(filepath: str | Path) -> Molecule:
    """Parse a Z-matrix file into a Molecule.

    Supports both inline numeric values and named variables with a
    variables section separated by a blank line.
    """
    filepath = Path(filepath)
    with open(filepath) as f:
        text = f.read()

    # Split into atom lines and optional variables section
    sections = text.strip().split("\n\n")
    atom_lines = [l.strip() for l in sections[0].strip().splitlines() if l.strip()]

    # Parse variables if present
    variables: dict[str, float] = {}
    for section in sections[1:]:
        for line in section.strip().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                name, val = line.split("=", 1)
                variables[name.strip()] = float(val.strip())

    def _resolve(token: str) -> float:
        """Resolve a token to a float, looking up variables if needed."""
        try:
            return float(token)
        except ValueError:
            # Handle negative variable references like -a1
            if token.startswith("-") and token[1:] in variables:
                return -variables[token[1:]]
            return variables[token]

    symbols: list[str] = []
    refs: list[tuple[int, ...]] = []
    values: list[tuple[float, ...]] = []

    for i, line in enumerate(atom_lines):
        parts = line.split()
        sym = parts[0]
        # Strip numeric suffix from labels like C1, H3
        sym_clean = ""
        for ch in sym:
            if ch.isalpha():
                sym_clean += ch
            else:
                break
        symbols.append(sym_clean)

        if i == 0:
            refs.append(())
            values.append(())
        elif i == 1:
            ref_a = int(parts[1]) - 1
            dist = _resolve(parts[2])
            refs.append((ref_a,))
            values.append((dist,))
        elif i == 2:
            ref_a = int(parts[1]) - 1
            dist = _resolve(parts[2])
            ref_b = int(parts[3]) - 1
            ang = _resolve(parts[4])
            refs.append((ref_a, ref_b))
            values.append((dist, ang))
        else:
            ref_a = int(parts[1]) - 1
            dist = _resolve(parts[2])
            ref_b = int(parts[3]) - 1
            ang = _resolve(parts[4])
            ref_c = int(parts[5]) - 1
            dih = _resolve(parts[6])
            refs.append((ref_a, ref_b, ref_c))
            values.append((dist, ang, dih))

    positions = _zmat_to_cartesian(symbols, refs, values)
    atoms = [Atom(element=get_element(sym), position=pos) for sym, pos in zip(symbols, positions)]
    mol = Molecule(atoms=atoms, bonds=[])
    mol.detect_bonds()
    return mol


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
    elif suffix in (".zmat", ".zmatrix"):
        return parse_zmat(filepath)
    elif suffix == ".gbw":
        raise ValueError(
            ".gbw files must be opened via the moltui command, not load_molecule(). "
            "Use: moltui <file.gbw>"
        )
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Use .xyz, .cube, .molden, or .gbw")
