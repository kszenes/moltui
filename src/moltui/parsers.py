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


@dataclass
class XYZTrajectory:
    molecule: Molecule
    frames: np.ndarray  # (n_frames, n_atoms, 3) in Angstrom


@dataclass
class HessData:
    molecule: Molecule
    frequencies: np.ndarray | None = None  # (n_modes,)
    normal_modes: np.ndarray | None = None  # (n_modes, n_atoms, 3) in Angstrom


def _parse_float(token: str) -> float:
    return float(token.replace("D", "E").replace("d", "e"))


def _parse_orca_hess_sections(text: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("$"):
            current = stripped[1:].strip().lower()
            sections.setdefault(current, [])
            continue
        if current is not None:
            sections[current].append(raw_line)
    return sections


def _parse_orca_hess_block_matrix(
    lines: list[str], n_rows: int, n_cols: int, section_name: str
) -> np.ndarray:
    matrix = np.zeros((n_rows, n_cols), dtype=np.float64)
    filled = np.zeros((n_rows, n_cols), dtype=bool)
    col_indices: list[int] | None = None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        tokens = stripped.split()

        is_header = True
        for tok in tokens:
            try:
                int(tok)
            except ValueError:
                is_header = False
                break

        if is_header:
            col_indices = [int(tok) for tok in tokens]
            continue

        if col_indices is None:
            raise ValueError(f"Invalid ${section_name} matrix block header")

        try:
            row_idx = int(tokens[0])
        except ValueError as exc:
            raise ValueError(f"Invalid ${section_name} row index: {tokens[0]!r}") from exc
        values = [_parse_float(tok) for tok in tokens[1:]]

        if len(values) != len(col_indices):
            raise ValueError(
                f"Invalid ${section_name} matrix row width for row {row_idx}: "
                f"expected {len(col_indices)}, got {len(values)}"
            )

        for col_idx, value in zip(col_indices, values):
            if row_idx < 0 or row_idx >= n_rows:
                raise ValueError(f"Invalid ${section_name} matrix row index: {row_idx}")
            if col_idx < 0 or col_idx >= n_cols:
                raise ValueError(f"Invalid ${section_name} matrix column index: {col_idx}")
            matrix[row_idx, col_idx] = value
            filled[row_idx, col_idx] = True

    if not np.all(filled):
        raise ValueError(f"Incomplete ${section_name} matrix data")

    return matrix


def parse_orca_hess_data(filepath: str | Path) -> HessData:
    filepath = Path(filepath)
    sections = _parse_orca_hess_sections(filepath.read_text())

    atom_lines = [
        line.strip()
        for line in sections.get("atoms", [])
        if line.strip() and not line.strip().startswith("#")
    ]
    if not atom_lines:
        raise ValueError("ORCA Hessian file missing $atoms section")

    try:
        n_atoms = int(atom_lines[0].split()[0])
    except (IndexError, ValueError) as exc:
        raise ValueError("Invalid $atoms atom-count line in ORCA Hessian file") from exc

    if len(atom_lines) < n_atoms + 1:
        raise ValueError("Incomplete $atoms section in ORCA Hessian file")

    atoms: list[Atom] = []
    for atom_line in atom_lines[1 : n_atoms + 1]:
        parts = atom_line.split()
        if len(parts) < 4:
            raise ValueError("Invalid $atoms entry in ORCA Hessian file")
        symbol = parts[0]
        x, y, z = (_parse_float(parts[-3]), _parse_float(parts[-2]), _parse_float(parts[-1]))
        coords = np.array([x, y, z], dtype=np.float64) * BOHR_TO_ANGSTROM
        atoms.append(Atom(element=get_element(symbol), position=coords))

    molecule = Molecule(atoms=atoms, bonds=[])
    molecule.detect_bonds()

    frequencies: np.ndarray | None = None
    freq_lines = [
        line.strip()
        for line in sections.get("vibrational_frequencies", [])
        if line.strip() and not line.strip().startswith("#")
    ]
    if freq_lines:
        try:
            n_freq = int(freq_lines[0].split()[0])
        except (IndexError, ValueError) as exc:
            raise ValueError("Invalid $vibrational_frequencies count in ORCA Hessian file") from exc
        parsed_freqs: list[float] = []
        for line in freq_lines[1:]:
            parts = line.split()
            if len(parts) == 1:
                parsed_freqs.append(_parse_float(parts[0]))
            else:
                parsed_freqs.append(_parse_float(parts[1]))
            if len(parsed_freqs) >= n_freq:
                break
        if len(parsed_freqs) < n_freq:
            raise ValueError("Incomplete $vibrational_frequencies section in ORCA Hessian file")
        frequencies = np.array(parsed_freqs[:n_freq], dtype=np.float64)

    normal_modes: np.ndarray | None = None
    mode_lines = [
        line.strip()
        for line in sections.get("normal_modes", [])
        if line.strip() and not line.strip().startswith("#")
    ]
    if mode_lines:
        dims = mode_lines[0].split()
        if len(dims) < 2:
            raise ValueError("Invalid $normal_modes dimensions in ORCA Hessian file")
        try:
            n_rows = int(dims[0])
            n_cols = int(dims[1])
        except ValueError as exc:
            raise ValueError("Invalid $normal_modes dimensions in ORCA Hessian file") from exc

        if n_rows != 3 * n_atoms:
            raise ValueError(
                "ORCA Hessian normal modes do not match atom count "
                f"(rows={n_rows}, expected={3 * n_atoms})"
            )

        mode_matrix = _parse_orca_hess_block_matrix(mode_lines[1:], n_rows, n_cols, "normal_modes")
        normal_modes = mode_matrix.T.reshape(n_cols, n_atoms, 3) * BOHR_TO_ANGSTROM
        if frequencies is not None:
            frequencies = frequencies[: normal_modes.shape[0]]

    return HessData(
        molecule=molecule,
        frequencies=frequencies,
        normal_modes=normal_modes,
    )


def parse_xyz(filepath: str | Path) -> Molecule:
    return parse_xyz_trajectory(filepath).molecule


def parse_xyz_trajectory(filepath: str | Path) -> XYZTrajectory:
    filepath = Path(filepath)
    with open(filepath) as f:
        lines = f.readlines()

    frames: list[np.ndarray] = []
    symbols_ref: list[str] | None = None
    idx = 0
    while idx < len(lines):
        while idx < len(lines) and not lines[idx].strip():
            idx += 1
        if idx >= len(lines):
            break

        try:
            n_atoms = int(lines[idx].strip())
        except ValueError as exc:
            raise ValueError(f"Invalid XYZ frame header at line {idx + 1}") from exc
        frame_start = idx + 2
        frame_end = frame_start + n_atoms
        if frame_end > len(lines):
            raise ValueError("Unexpected end of XYZ file while reading frame atoms")

        frame_symbols: list[str] = []
        frame_coords: list[list[float]] = []
        for line in lines[frame_start:frame_end]:
            parts = line.split()
            if len(parts) < 4:
                raise ValueError("Invalid XYZ atom line; expected: <symbol> <x> <y> <z>")
            frame_symbols.append(parts[0])
            frame_coords.append([float(parts[1]), float(parts[2]), float(parts[3])])

        if symbols_ref is None:
            symbols_ref = frame_symbols
        else:
            if len(frame_symbols) != len(symbols_ref):
                raise ValueError("All XYZ frames must have the same atom count")
            if frame_symbols != symbols_ref:
                raise ValueError("All XYZ frames must preserve atom ordering and symbols")

        frames.append(np.array(frame_coords, dtype=np.float64))
        idx = frame_end

    if not frames or symbols_ref is None:
        raise ValueError("Empty XYZ file")

    first_frame = frames[0]
    atoms = [
        Atom(element=get_element(symbol), position=first_frame[i].copy())
        for i, symbol in enumerate(symbols_ref)
    ]
    mol = Molecule(atoms=atoms, bonds=[])
    mol.detect_bonds()
    return XYZTrajectory(molecule=mol, frames=np.stack(frames, axis=0))


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
    elif suffix in (".fchk", ".fch"):
        from .fchk import parse_fchk_atoms

        return parse_fchk_atoms(filepath)
    elif suffix == ".hess":
        return parse_orca_hess_data(filepath).molecule
    elif suffix in (".zmat", ".zmatrix"):
        return parse_zmat(filepath)
    elif suffix == ".gbw":
        raise ValueError(
            ".gbw files must be opened via the moltui command, not load_molecule(). "
            "Use: moltui <file.gbw>"
        )
    else:
        from .trexio_support import is_trexio_path, load_molecule_from_trexio

        if is_trexio_path(filepath):
            return load_molecule_from_trexio(filepath)
        raise ValueError(
            f"Unsupported file format: {suffix}. Use .xyz, .cube, .molden, .hess, .gbw, "
            "or TREXIO (.h5, .hdf5, .trexio; install optional extra: trexio)"
        )
