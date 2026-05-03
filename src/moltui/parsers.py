import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .elements import ELEMENTS, Atom, Element, Molecule, get_element, get_element_by_number


class CIFParseWarning(UserWarning):
    """Warning emitted by parse_cif for non-fatal interpretation issues."""


BOHR_TO_ANGSTROM = 0.529177249


@dataclass
class VolumetricData:
    molecule: Molecule
    origin: np.ndarray  # (3,) in Angstrom
    axes: np.ndarray  # (3, 3) grid step vectors in Angstrom
    n_points: tuple[int, int, int]
    data: np.ndarray  # (n1, n2, n3) volumetric data
    periodic: bool | tuple[bool, bool, bool] = False


@dataclass
class CubeData:
    molecule: Molecule
    origin: np.ndarray  # (3,) in Bohr
    axes: np.ndarray  # (3, 3) step vectors in Bohr
    n_points: tuple[int, int, int]
    data: np.ndarray  # (n1, n2, n3) volumetric data
    periodic: bool | tuple[bool, bool, bool] = False

    def to_volumetric_data(self) -> VolumetricData:
        return VolumetricData(
            molecule=self.molecule,
            origin=self.origin * BOHR_TO_ANGSTROM,
            axes=self.axes * BOHR_TO_ANGSTROM,
            n_points=self.n_points,
            data=self.data,
            periodic=self.periodic,
        )


@dataclass
class XYZTrajectory:
    molecule: Molecule
    frames: np.ndarray  # (n_frames, n_atoms, 3) in Angstrom
    lattice: np.ndarray | None = None  # (3, 3) in Angstrom, or None


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
    """Parse the first frame of an XYZ/extXYZ file as a molecule."""
    filepath = Path(filepath)
    with open(filepath) as f:
        lines = f.readlines()

    idx = 0
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    if idx >= len(lines):
        raise ValueError("Empty XYZ file")

    try:
        n_atoms = int(lines[idx].strip())
    except ValueError as exc:
        raise ValueError(f"Invalid XYZ frame header at line {idx + 1}") from exc

    comment_idx = idx + 1
    frame_start = idx + 2
    frame_end = frame_start + n_atoms
    if frame_end > len(lines):
        raise ValueError("Unexpected end of XYZ file while reading frame atoms")

    species_col = 0
    pos_col = 1
    lattice = None
    pbc = None
    comment_line = lines[comment_idx] if comment_idx < len(lines) else ""
    metadata = _parse_xyz_comment_metadata(comment_line)
    if "Lattice" in metadata:
        lattice = _parse_lattice(metadata["Lattice"])
    if "pbc" in metadata:
        pbc = _parse_pbc(metadata["pbc"])
    if "Properties" in metadata:
        species_col, pos_col = _parse_properties_spec(metadata["Properties"])

    atoms: list[Atom] = []
    for line in lines[frame_start:frame_end]:
        parts = line.split()
        if len(parts) <= max(species_col, pos_col + 2):
            raise ValueError(f"Invalid XYZ atom line: {line!r}")
        element = _resolve_species(parts[species_col])
        try:
            pos = np.array(
                [
                    float(parts[pos_col]),
                    float(parts[pos_col + 1]),
                    float(parts[pos_col + 2]),
                ],
                dtype=np.float64,
            )
        except ValueError as exc:
            raise ValueError(f"Invalid XYZ coordinates in line: {line!r}") from exc
        atoms.append(Atom(element, pos))

    molecule = Molecule(atoms=atoms, bonds=[], lattice=lattice, pbc=pbc)
    molecule.detect_bonds_auto()
    return molecule


def _parse_xyz_comment_metadata(line: str) -> dict[str, str]:
    """Parse `key=value` pairs from an extended-XYZ comment line.

    Handles double-quoted values containing spaces (e.g. `Lattice="..."`) and
    bare values (e.g. `energy=-15.5`). Returns an empty dict for plain XYZ
    comments with no `=`.
    """
    matches = re.findall(r'(\w+)=(?:"([^"]*)"|(\S+))', line.rstrip("\n\r"))
    return {key: quoted or bare for key, quoted, bare in matches}


def _parse_lattice(value: str) -> np.ndarray:
    parts = value.split()
    if len(parts) != 9:
        raise ValueError(f"Lattice must contain 9 floats, got {len(parts)}: {value!r}")
    return np.array([float(p) for p in parts], dtype=np.float64).reshape(3, 3)


def _parse_pbc(value: str) -> tuple[bool, bool, bool]:
    tokens = value.split()
    if len(tokens) == 1:
        tokens = tokens * 3
    if len(tokens) != 3:
        raise ValueError(f"pbc must contain 1 or 3 boolean values, got {len(tokens)}: {value!r}")
    true_values = {"t", "true", "1"}
    false_values = {"f", "false", "0"}
    parsed: list[bool] = []
    for token in tokens:
        lower = token.lower()
        if lower in true_values:
            parsed.append(True)
        elif lower in false_values:
            parsed.append(False)
        else:
            raise ValueError(f"Invalid pbc boolean value: {token!r}")
    return (parsed[0], parsed[1], parsed[2])


def _parse_properties_spec(value: str) -> tuple[int, int]:
    """Parse a Properties=... spec into (species_col, pos_col).

    Returns 0-based column indices for the species token and the first of the
    three position tokens. Raises ValueError if species or pos cannot be
    located.
    """
    triples = value.split(":")
    if len(triples) % 3 != 0:
        raise ValueError(f"Malformed Properties spec: {value!r}")

    species_col: int | None = None
    pos_col: int | None = None
    col = 0
    for k in range(0, len(triples), 3):
        name = triples[k]
        type_code = triples[k + 1]
        try:
            count = int(triples[k + 2])
        except ValueError as exc:
            raise ValueError(f"Malformed Properties spec: {value!r}") from exc
        if name in ("species", "Z") and count == 1 and type_code in ("S", "I"):
            species_col = col
        elif name == "pos" and count == 3 and type_code == "R":
            pos_col = col
        col += count

    if species_col is None or pos_col is None:
        raise ValueError(f"Properties spec missing species or pos field: {value!r}")
    return species_col, pos_col


def _resolve_species(token: str) -> Element:
    """Resolve an atom token (element symbol or atomic number) to an Element."""
    try:
        z = int(token)
    except ValueError:
        return get_element(token)
    return get_element_by_number(z)


def parse_xyz_trajectory(filepath: str | Path) -> XYZTrajectory:
    filepath = Path(filepath)
    with open(filepath) as f:
        lines = f.readlines()

    frames: list[np.ndarray] = []
    symbols_ref: list[str] | None = None
    lattice: np.ndarray | None = None
    pbc: tuple[bool, bool, bool] | None = None
    species_col = 0
    pos_col = 1
    is_first_frame = True
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
        comment_idx = idx + 1
        frame_start = idx + 2
        frame_end = frame_start + n_atoms
        if frame_end > len(lines):
            raise ValueError("Unexpected end of XYZ file while reading frame atoms")

        if is_first_frame:
            comment_line = lines[comment_idx] if comment_idx < len(lines) else ""
            metadata = _parse_xyz_comment_metadata(comment_line)
            if "Lattice" in metadata:
                lattice = _parse_lattice(metadata["Lattice"])
            if "pbc" in metadata:
                pbc = _parse_pbc(metadata["pbc"])
            if "Properties" in metadata:
                species_col, pos_col = _parse_properties_spec(metadata["Properties"])
            is_first_frame = False

        frame_symbols: list[str] = []
        frame_coords: list[list[float]] = []
        min_cols = max(species_col, pos_col + 2) + 1
        for line in lines[frame_start:frame_end]:
            parts = line.split()
            if len(parts) < min_cols:
                raise ValueError(
                    f"Invalid XYZ atom line; expected at least {min_cols} columns: {line!r}"
                )
            frame_symbols.append(parts[species_col])
            frame_coords.append(
                [
                    float(parts[pos_col]),
                    float(parts[pos_col + 1]),
                    float(parts[pos_col + 2]),
                ]
            )

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
        Atom(element=_resolve_species(symbol), position=first_frame[i].copy())
        for i, symbol in enumerate(symbols_ref)
    ]
    mol = Molecule(atoms=atoms, bonds=[], lattice=lattice, pbc=pbc)
    mol.detect_bonds_auto()
    return XYZTrajectory(molecule=mol, frames=np.stack(frames, axis=0), lattice=lattice)


def _cif_fractional_to_cartesian_matrix(
    a: float, b: float, c: float, alpha: float, beta: float, gamma: float
) -> np.ndarray:
    """Build M such that cart = frac @ M (rows are lattice vectors)."""
    alpha_r = np.radians(alpha)
    beta_r = np.radians(beta)
    gamma_r = np.radians(gamma)
    cos_a, cos_b, cos_g = np.cos(alpha_r), np.cos(beta_r), np.cos(gamma_r)
    sin_g = np.sin(gamma_r)

    bx = b * cos_g
    by = b * sin_g
    cx = c * cos_b
    cy = c * (cos_a - cos_b * cos_g) / sin_g
    cz_sq = c * c - cx * cx - cy * cy
    cz = np.sqrt(max(cz_sq, 0.0))
    return np.array([[a, 0.0, 0.0], [bx, by, 0.0], [cx, cy, cz]], dtype=np.float64)


def _strip_cif_value(token: str) -> str:
    """Strip CIF uncertainty parens and surrounding quotes."""
    token = token.strip()
    if (token.startswith("'") and token.endswith("'")) or (
        token.startswith('"') and token.endswith('"')
    ):
        token = token[1:-1]
    paren = token.find("(")
    if paren != -1:
        token = token[:paren]
    return token


def _cif_float(token: str) -> float:
    return float(_strip_cif_value(token))


def _split_cif_tokens(line: str) -> list[str]:
    """Split a CIF data line, respecting single/double quotes."""
    tokens: list[str] = []
    i = 0
    n = len(line)
    while i < n:
        c = line[i]
        if c.isspace():
            i += 1
            continue
        if c in ("'", '"'):
            quote = c
            j = i + 1
            while j < n and line[j] != quote:
                j += 1
            tokens.append(line[i + 1 : j])
            i = j + 1
        else:
            j = i
            while j < n and not line[j].isspace():
                j += 1
            tokens.append(line[i:j])
            i = j
    return tokens


def _parse_symop(op: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse a CIF symmetry operation like ``"1/2-x,1/2+y,-z"``.

    Returns a ``(rot, trans)`` pair such that the operation maps a fractional
    coordinate ``v`` to ``rot @ v + trans``. Whitespace, ``*`` separators
    between coefficients and variables (``2*x``), and divisions on either
    side of a variable (``x/2`` or ``1/2*x``) are accepted.
    """
    parts = op.replace("'", "").replace('"', "").lower().split(",")
    if len(parts) != 3:
        raise ValueError(f"symop must have 3 components: {op!r}")
    rot = np.zeros((3, 3), dtype=np.float64)
    trans = np.zeros(3, dtype=np.float64)
    for row, raw in enumerate(parts):
        s = raw.replace(" ", "").replace("\t", "")
        if not s:
            raise ValueError(f"empty component in symop: {op!r}")
        i = 0
        while i < len(s):
            sign = 1.0
            if s[i] == "+":
                i += 1
            elif s[i] == "-":
                sign = -1.0
                i += 1
            num_start = i
            while i < len(s) and (s[i].isdigit() or s[i] == "."):
                i += 1
            num_str = s[num_start:i]
            denom = 1.0
            if i < len(s) and s[i] == "/":
                i += 1
                d_start = i
                while i < len(s) and s[i].isdigit():
                    i += 1
                if i == d_start:
                    raise ValueError(f"missing denominator in symop component: {raw!r}")
                denom = float(s[d_start:i])
            if i < len(s) and s[i] == "*":
                i += 1
            var: str | None = None
            if i < len(s) and s[i] in ("x", "y", "z"):
                var = s[i]
                i += 1
                if i < len(s) and s[i] == "/":
                    i += 1
                    d_start = i
                    while i < len(s) and s[i].isdigit():
                        i += 1
                    if i == d_start:
                        raise ValueError(f"missing denominator in symop component: {raw!r}")
                    denom *= float(s[d_start:i])
            if num_str == "" and var is None:
                raise ValueError(f"unparsable symop component: {raw!r}")
            magnitude = float(num_str) if num_str else 1.0
            coef = sign * magnitude / denom
            if var is None:
                trans[row] += coef
            else:
                col = "xyz".index(var)
                rot[row, col] += coef
    return rot, trans


def _apply_symops(
    fracs: np.ndarray,
    symbols: list[str],
    symops: list[tuple[np.ndarray, np.ndarray]],
    tol: float = 1e-4,
) -> tuple[np.ndarray, list[str]]:
    """Expand a fractional-coord atom set by a list of symmetry operations.

    Atoms produced by different operations that map to the same fractional
    site (modulo the lattice) are deduplicated. Returns ``(fracs, symbols)``
    for the expanded set; original atoms are guaranteed to appear first.
    """
    if not symops:
        return fracs, list(symbols)
    expanded_fracs: list[np.ndarray] = []
    expanded_symbols: list[str] = []
    seen_keys: set[tuple[int, int, int]] = set()
    scale = int(round(1.0 / tol))
    for rot, trans in symops:
        for row, sym in zip(fracs, symbols):
            mapped = rot @ row + trans
            mapped_mod = mapped - np.floor(mapped)
            # Snap near-integer-1 components to 0 to avoid (0, 0, 1.0 - eps)
            # collapsing to a different cell from the canonical (0, 0, 0).
            mapped_mod = np.where(mapped_mod > 1.0 - tol, 0.0, mapped_mod)
            key = tuple(int(round(c * scale)) % scale for c in mapped_mod)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            expanded_fracs.append(mapped_mod)
            expanded_symbols.append(sym)
    return np.array(expanded_fracs, dtype=np.float64), expanded_symbols


def _element_from_cif_label(label: str) -> str:
    """Extract an element symbol from a CIF atom label like 'C1', 'Fe2+', 'ca1'.

    Tries a two-letter element symbol (case-insensitive) first and falls back
    to the single first letter, accepting both ``Ca1`` and ``ca1`` forms.
    """
    label = label.strip()
    if not label:
        raise ValueError("Empty CIF atom label")
    if len(label) >= 2 and label[0].isalpha() and label[1].isalpha():
        candidate = label[0].upper() + label[1].lower()
        if candidate in ELEMENTS:
            return candidate
    return label[0].upper()


def _clean_cif_lines(raw_lines: list[str]) -> list[str]:
    """Strip comments and collapse ``;``-delimited multi-line text fields.

    A multi-line text block opens with a line starting with ``;`` (column 1)
    and closes with the next such line. The whole block represents a single
    CIF value, so we collapse it to one double-quoted token appended to the
    preceding output line (or emitted alone if no preceding line exists).
    """
    lines: list[str] = []
    in_semi = False
    semi_buf: list[str] = []
    for raw in raw_lines:
        line = raw.rstrip("\r\n")
        if in_semi:
            if line.startswith(";"):
                content = " ".join(s.strip() for s in semi_buf if s.strip())
                token = '"' + content.replace('"', "'") + '"'
                if lines:
                    lines[-1] = lines[-1] + " " + token
                else:
                    lines.append(token)
                in_semi = False
                semi_buf = []
            else:
                semi_buf.append(line)
            continue
        if line.startswith(";"):
            in_semi = True
            rest = line[1:]
            if rest.strip():
                semi_buf.append(rest)
            continue
        stripped = line.split("#", 1)[0].rstrip()
        if stripped.strip():
            lines.append(stripped)
    return lines


def _parse_cif_cell(lines: list[str]) -> dict[str, float]:
    cell: dict[str, float] = {}
    cell_keys = {
        "_cell_length_a": "a",
        "_cell_length_b": "b",
        "_cell_length_c": "c",
        "_cell_angle_alpha": "alpha",
        "_cell_angle_beta": "beta",
        "_cell_angle_gamma": "gamma",
    }
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        lower = stripped.lower()
        matched_key = next((key for key in cell_keys if lower.startswith(key)), None)
        if matched_key is None:
            i += 1
            continue
        tokens = _split_cif_tokens(stripped)
        if len(tokens) >= 2:
            cell[cell_keys[matched_key]] = _cif_float(tokens[1])
        else:
            i += 1
            cell[cell_keys[matched_key]] = _cif_float(lines[i].strip())
        i += 1
    return cell


def _skip_cif_loop_rows(lines: list[str], start_idx: int) -> int:
    i = start_idx
    while i < len(lines):
        s = lines[i].strip()
        if not s or s.lower() == "loop_" or s.startswith("_") or s.startswith("data_"):
            break
        i += 1
    return i


def _parse_cif_symmetry(
    headers: list[str], lines: list[str], start_idx: int
) -> tuple[list[tuple[np.ndarray, np.ndarray]], int]:
    symop_header_names = (
        "_symmetry_equiv_pos_as_xyz",
        "_space_group_symop_operation_xyz",
    )
    symop_col = next((idx for idx, h in enumerate(headers) if h in symop_header_names), None)
    if symop_col is None:
        return [], start_idx

    symops: list[tuple[np.ndarray, np.ndarray]] = []
    i = start_idx
    while i < len(lines):
        s = lines[i].strip()
        if not s or s.lower() == "loop_" or s.startswith("_") or s.startswith("data_"):
            break
        tokens = _split_cif_tokens(s)
        if len(tokens) == len(headers):
            op_str = tokens[symop_col]
        elif len(tokens) >= symop_col + 1:
            op_str = tokens[symop_col]
        else:
            i += 1
            continue
        rot, trans = _parse_symop(op_str)
        symops.append((rot, trans))
        i += 1
    return symops, i


def _parse_cif_atoms(
    headers: list[str], lines: list[str], start_idx: int, cell: dict[str, float]
) -> tuple[list[Atom], list[str], list[np.ndarray], int]:
    atom_site_headers = [h for h in headers if h.startswith("_atom_site_")]
    if not atom_site_headers or len(atom_site_headers) != len(headers):
        return [], [], [], start_idx

    col_index = {h: idx for idx, h in enumerate(headers)}
    label_idx = col_index.get("_atom_site_label")
    symbol_idx = col_index.get("_atom_site_type_symbol")
    fx_idx = col_index.get("_atom_site_fract_x")
    fy_idx = col_index.get("_atom_site_fract_y")
    fz_idx = col_index.get("_atom_site_fract_z")
    cx_idx = col_index.get("_atom_site_cartn_x")
    cy_idx = col_index.get("_atom_site_cartn_y")
    cz_idx = col_index.get("_atom_site_cartn_z")
    occupancy_idx = col_index.get("_atom_site_occupancy")

    coord_idx: tuple[int, int, int] | None = None
    use_fractional = False
    if fx_idx is not None and fy_idx is not None and fz_idx is not None:
        coord_idx = (fx_idx, fy_idx, fz_idx)
        use_fractional = True
    elif cx_idx is not None and cy_idx is not None and cz_idx is not None:
        coord_idx = (cx_idx, cy_idx, cz_idx)
    else:
        return [], [], [], _skip_cif_loop_rows(lines, start_idx)

    lattice = np.eye(3)
    if use_fractional:
        missing = [k for k in ("a", "b", "c", "alpha", "beta", "gamma") if k not in cell]
        if missing:
            raise ValueError(
                f"CIF file uses fractional coordinates but is missing cell parameters: {missing}"
            )
        lattice = _cif_fractional_to_cartesian_matrix(
            cell["a"],
            cell["b"],
            cell["c"],
            cell["alpha"],
            cell["beta"],
            cell["gamma"],
        )

    atoms: list[Atom] = []
    frac_symbols: list[str] = []
    frac_coords: list[np.ndarray] = []
    warned_fractional_occupancy = False
    i = start_idx
    while i < len(lines):
        s = lines[i].strip()
        if not s or s.lower() == "loop_" or s.startswith("data_") or s.startswith("_"):
            break
        tokens = _split_cif_tokens(s)
        if len(tokens) != len(headers):
            raise ValueError(f"CIF atom_site row has {len(tokens)} tokens, expected {len(headers)}")

        if symbol_idx is not None:
            symbol = _strip_cif_value(tokens[symbol_idx])
        elif label_idx is not None:
            symbol = _element_from_cif_label(_strip_cif_value(tokens[label_idx]))
        else:
            raise ValueError("CIF atom_site loop missing label and type_symbol")

        symbol_clean = ""
        for ch in symbol:
            if ch.isalpha():
                symbol_clean += ch
            else:
                break
        if not symbol_clean:
            raise ValueError(f"Could not derive element from CIF symbol {symbol!r}")

        if occupancy_idx is not None and not warned_fractional_occupancy:
            occ_token = _strip_cif_value(tokens[occupancy_idx])
            if occ_token not in ("?", "."):
                try:
                    occupancy = _cif_float(occ_token)
                except ValueError:
                    occupancy = 1.0
                if not np.isclose(occupancy, 1.0):
                    warnings.warn(
                        "CIF fractional occupancies/disorder are displayed as full atoms; "
                        "occupancy metadata is ignored.",
                        CIFParseWarning,
                        stacklevel=2,
                    )
                    warned_fractional_occupancy = True

        assert coord_idx is not None
        ix, iy, iz = coord_idx
        coords = np.array(
            [_cif_float(tokens[ix]), _cif_float(tokens[iy]), _cif_float(tokens[iz])],
            dtype=np.float64,
        )
        pos = coords @ lattice if use_fractional else coords
        atoms.append(Atom(element=get_element(symbol_clean), position=pos))
        if use_fractional:
            frac_symbols.append(symbol_clean)
            frac_coords.append(coords)
        i += 1

    return atoms, frac_symbols, frac_coords, i


def parse_cif(filepath: str | Path) -> Molecule:
    """Parse a minimal CIF file into a Molecule.

    Supports cell parameters (_cell_length_a/b/c, _cell_angle_alpha/beta/gamma)
    and a single atom_site loop with fractional or Cartesian coordinates.
    """
    filepath = Path(filepath)
    with open(filepath) as f:
        lines = _clean_cif_lines(f.readlines())

    cell = _parse_cif_cell(lines)

    atoms: list[Atom] = []
    frac_atoms_symbols: list[str] = []
    frac_atoms_coords: list[np.ndarray] = []
    symops: list[tuple[np.ndarray, np.ndarray]] = []
    hm_name: str | None = _extract_hm_space_group_name(lines)

    i = 0
    while i < len(lines):
        if lines[i].strip().lower() == "loop_":
            i += 1
            headers: list[str] = []
            while i < len(lines) and lines[i].strip().startswith("_"):
                headers.append(lines[i].strip().lower())
                i += 1

            loop_symops, next_i = _parse_cif_symmetry(headers, lines, i)
            if loop_symops:
                symops.extend(loop_symops)
                i = next_i
                continue

            loop_atoms, loop_symbols, loop_fracs, next_i = _parse_cif_atoms(headers, lines, i, cell)
            if loop_atoms:
                atoms.extend(loop_atoms)
                frac_atoms_symbols.extend(loop_symbols)
                frac_atoms_coords.extend(loop_fracs)
                i = next_i
                continue

            i = _skip_cif_loop_rows(lines, i)
            continue

        i += 1

    if not atoms:
        raise ValueError("No atoms found in CIF file")

    mol_lattice: np.ndarray | None = None
    if all(k in cell for k in ("a", "b", "c", "alpha", "beta", "gamma")):
        mol_lattice = _cif_fractional_to_cartesian_matrix(
            cell["a"],
            cell["b"],
            cell["c"],
            cell["alpha"],
            cell["beta"],
            cell["gamma"],
        )

    # Apply symmetry expansion when we have ops, fractional atoms, and a cell.
    if symops and mol_lattice is not None and frac_atoms_coords:
        # Drop the trivial single-identity case to avoid duplicating atoms via
        # near-zero rounding in dedupe; the expansion path is a no-op anyway.
        nontrivial = any(
            not (np.allclose(rot, np.eye(3)) and np.allclose(trans, 0.0)) for rot, trans in symops
        )
        if nontrivial or len(symops) > 1:
            fracs = np.array(frac_atoms_coords, dtype=np.float64)
            expanded_fracs, expanded_symbols = _apply_symops(fracs, frac_atoms_symbols, symops)
            atoms = [
                Atom(element=get_element(sym), position=frac @ mol_lattice)
                for sym, frac in zip(expanded_symbols, expanded_fracs)
            ]

    if hm_name is not None and not symops and hm_name.replace(" ", "").upper() not in ("P1", "P-1"):
        warnings.warn(
            f"CIF declares space group {hm_name!r} but provides no symmetry "
            "operations; structure shown is the asymmetric unit only (treated as P1).",
            CIFParseWarning,
            stacklevel=2,
        )

    mol = Molecule(atoms=atoms, bonds=[], lattice=mol_lattice)
    mol.detect_bonds_auto()
    return mol


def _extract_hm_space_group_name(lines: list[str]) -> str | None:
    """Return the Hermann-Mauguin space group name from a CIF, if present."""
    keys = (
        "_symmetry_space_group_name_h-m",
        "_space_group_name_h-m_alt",
        "_space_group_name_h-m",
    )
    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()
        for key in keys:
            if lower.startswith(key):
                tokens = _split_cif_tokens(stripped)
                if len(tokens) >= 2:
                    return _strip_cif_value(" ".join(tokens[1:]))
                return None
    return None


def parse_cube(filepath: str | Path) -> Molecule:
    cube_data = parse_cube_data(filepath)
    return cube_data.molecule


def parse_cube_data(filepath: str | Path, periodic: bool = False) -> CubeData:
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

    n_points_tuple = (n_points[0], n_points[1], n_points[2])
    lattice = None
    pbc = None
    if periodic:
        lattice = np.array(
            [n_points_tuple[i] * axes[i] * BOHR_TO_ANGSTROM for i in range(3)],
            dtype=np.float64,
        )
        # Cube files do not carry explicit PBC flags. Match ASE's convention:
        # derive periodicity per axis from whether the corresponding cell
        # vector is nonzero. This supports 2D slices/slabs represented by a
        # zero grid step vector.
        pbc = tuple(bool(np.any(np.abs(vec) > 1e-12)) for vec in lattice)
    mol = Molecule(
        atoms=atoms,
        bonds=[],
        lattice=lattice,
        pbc=pbc,
    )
    mol.detect_bonds_auto()

    return CubeData(
        molecule=mol,
        origin=origin,
        axes=axes,
        n_points=n_points_tuple,
        data=data,
        periodic=periodic,
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
    return parse_zmat_text(text)


def parse_zmat_text(text: str) -> Molecule:
    """Parse Z-matrix content (geometry + optional variables) into a Molecule.

    Sections (geometry, variables) are separated by blank lines. Atoms beyond
    the third reference earlier atoms by 1-based line index; values may be
    inline floats or named variables defined after a blank line as
    `name = value`.
    """
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


def parse_poscar(filepath: str | Path) -> Molecule:
    from .periodic.vasp import parse_poscar as _parse_poscar

    return _parse_poscar(filepath)


def parse_xsf(filepath: str | Path) -> Molecule:
    from .periodic.xsf import parse_xsf as _parse_xsf

    return _parse_xsf(filepath)


def parse_xsf_volumetric_data(filepath: str | Path) -> VolumetricData:
    from .periodic.xsf import parse_xsf_volumetric_data as _parse_xsf_volumetric_data

    return _parse_xsf_volumetric_data(filepath)


def parse_vasp_volumetric_data(filepath: str | Path) -> VolumetricData:
    from .periodic.vasp import parse_vasp_volumetric_data as _parse_vasp_volumetric_data

    return _parse_vasp_volumetric_data(filepath)


def _is_poscar_like_path(filepath: Path) -> bool:
    return filepath.name.lower() in ("poscar", "contcar") or filepath.suffix.lower() in (
        ".vasp",
        ".poscar",
    )


def load_molecule(filepath: str | Path) -> Molecule:
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()
    if suffix in (".xyz", ".extxyz"):
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
    elif suffix == ".cif":
        return parse_cif(filepath)
    elif _is_poscar_like_path(filepath):
        return parse_poscar(filepath)
    elif suffix == ".xsf":
        return parse_xsf(filepath)
    elif suffix == ".gbw":
        raise ValueError(
            ".gbw files must be opened via the moltui command, not load_molecule(). "
            "Use: moltui <file.gbw>"
        )
    else:
        from .qc_inputs import (
            QC_INPUT_AMBIGUOUS_SUFFIXES,
            detect_qc_input_by_extension,
            parse_qc_input,
            sniff_qc_input,
        )

        qc_kind = detect_qc_input_by_extension(filepath)
        if qc_kind is not None:
            return parse_qc_input(filepath, qc_kind)
        if suffix in QC_INPUT_AMBIGUOUS_SUFFIXES or suffix == "":
            sniffed = sniff_qc_input(filepath)
            if sniffed is not None:
                return parse_qc_input(filepath, sniffed)
            raise ValueError(f"Could not identify QC input format from contents of {filepath!s}")
        from .trexio_support import is_trexio_path, load_molecule_from_trexio

        if is_trexio_path(filepath):
            return load_molecule_from_trexio(filepath)
        raise ValueError(
            f"Unsupported file format: {suffix}. Use .xyz, .extxyz, .cube, .molden, "
            ".hess, .cif, POSCAR/CONTCAR/.vasp, .xsf, .gbw, a QC input "
            "(Orca, Q-Chem, Gaussian, NWChem, Turbomole, Molcas, Molpro, MRCC, "
            "CFOUR, Psi4, GAMESS, or Jaguar), or TREXIO (.h5, .hdf5, .trexio; "
            "install optional extra: trexio)"
        )
