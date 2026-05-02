"""Geometry-only parsers and dispatch for QC input files.

Supported formats:
- Orca (`.inp`)
- Q-Chem (`.in`, `.qcin`)
- Gaussian (`.com`, `.gjf`)
- NWChem (`.nw`, `.nwi`)
- Turbomole (`coord` file)
- Molcas / OpenMolcas (`.input`)
- Molpro (`.com`, `.inp`)
- MRCC (`MINP`)
- CFOUR (`ZMAT`)
- Psi4 (`.dat`)
- GAMESS (`.inp`)
- Jaguar (`.in`)

Each parser returns a `Molecule` with coordinates in Angstrom. Dispatch
helpers (`detect_qc_input_by_extension`, `sniff_qc_input`, `parse_qc_input`)
are the single source of truth shared by `parsers.load_molecule` and
`app._detect_filetype`.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np

from .elements import (
    Atom,
    Element,
    Molecule,
    get_element,
    get_element_by_number,
    get_element_from_tag,
)

BOHR_TO_ANGSTROM = 0.529177249
_BOHR_UNITS = {"bohr", "bohrs", "au", "a.u.", "atomic"}
_ANGSTROM_UNITS = {"ang", "angs", "angstrom", "angstroms"}

ElementResolver = Callable[[str], Element]


def _make_molecule(atoms: list[Atom]) -> Molecule:
    mol = Molecule(atoms=atoms, bonds=[])
    mol.detect_bonds()
    return mol


def _scale_molecule(mol: Molecule, factor: float) -> Molecule:
    """Scale molecule coordinates in-place and rebuild bonds."""
    if factor != 1.0:
        for atom in mol.atoms:
            atom.position *= factor
        mol.bonds = []
        mol.detect_bonds()
    return mol


def _unit_factor(unit: str | None, *, default_bohr: bool = False) -> float:
    """Return coordinate factor to Angstrom for common unit spellings."""
    if unit is None:
        return BOHR_TO_ANGSTROM if default_bohr else 1.0
    normalized = unit.strip().strip(",;").lower()
    if normalized in _BOHR_UNITS:
        return BOHR_TO_ANGSTROM
    if normalized in _ANGSTROM_UNITS:
        return 1.0
    return BOHR_TO_ANGSTROM if default_bohr else 1.0


def _bohr_factor(enabled: bool) -> float:
    return BOHR_TO_ANGSTROM if enabled else 1.0


def _unit_factor_from_tokens(tokens: Iterable[str], *, default_bohr: bool = False) -> float:
    for token in tokens:
        normalized = token.strip().strip(",;").lower()
        if normalized in _BOHR_UNITS | _ANGSTROM_UNITS:
            return _unit_factor(normalized, default_bohr=default_bohr)
    return BOHR_TO_ANGSTROM if default_bohr else 1.0


def _split_fields(line: str) -> list[str]:
    """Split atom lines on whitespace plus comma/semicolon separators."""
    return [p for p in re.split(r"[,;\s]+", line.strip()) if p]


def _parse_xyz_atom_line(line: str) -> tuple[str, float, float, float] | None:
    parts = line.split()
    if len(parts) < 4:
        return None
    sym = parts[0]
    try:
        x = float(parts[1])
        y = float(parts[2])
        z = float(parts[3])
    except ValueError:
        return None
    return sym, x, y, z


def _atom_from_fields(
    fields: list[str],
    *,
    symbol_index: int = 0,
    coord_start: int = 1,
    factor: float = 1.0,
    element_resolver: ElementResolver = get_element,
) -> Atom | None:
    if len(fields) <= max(symbol_index, coord_start + 2):
        return None
    try:
        x = float(fields[coord_start]) * factor
        y = float(fields[coord_start + 1]) * factor
        z = float(fields[coord_start + 2]) * factor
    except ValueError:
        return None
    return Atom(element=element_resolver(fields[symbol_index]), position=np.array([x, y, z]))


def _cartesian_atoms_from_lines(
    lines: Iterable[str],
    *,
    factor: float = 1.0,
    element_resolver: ElementResolver = get_element,
    split_fields: bool = False,
    symbol_index: int = 0,
    coord_start: int = 1,
    skip_prefixes: tuple[str, ...] = (),
    stop_prefixes: tuple[str, ...] = (),
) -> list[Atom]:
    atoms: list[Atom] = []
    for raw in lines:
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        low = s.lower()
        if stop_prefixes and low.startswith(stop_prefixes):
            break
        if skip_prefixes and low.startswith(skip_prefixes):
            continue
        fields = _split_fields(s) if split_fields else s.split()
        atom = _atom_from_fields(
            fields,
            symbol_index=symbol_index,
            coord_start=coord_start,
            factor=factor,
            element_resolver=element_resolver,
        )
        if atom is not None:
            atoms.append(atom)
    return atoms


def _molecule_from_cartesian_lines(
    lines: Iterable[str],
    *,
    factor: float = 1.0,
    element_resolver: ElementResolver = get_element,
    split_fields: bool = False,
    symbol_index: int = 0,
    coord_start: int = 1,
    error: str,
    skip_prefixes: tuple[str, ...] = (),
    stop_prefixes: tuple[str, ...] = (),
) -> Molecule:
    atoms = _cartesian_atoms_from_lines(
        lines,
        factor=factor,
        element_resolver=element_resolver,
        split_fields=split_fields,
        symbol_index=symbol_index,
        coord_start=coord_start,
        skip_prefixes=skip_prefixes,
        stop_prefixes=stop_prefixes,
    )
    if not atoms:
        raise ValueError(error)
    return _make_molecule(atoms)


def _zmat_text(geom_lines: Iterable[str], var_lines: Iterable[str] = ()) -> str:
    text = "\n".join(geom_lines)
    variables = list(var_lines)
    if variables:
        text += "\n\n" + "\n".join(variables)
    return text


def _parse_zmat_text_scaled(
    geom_lines: Iterable[str],
    var_lines: Iterable[str] = (),
    *,
    factor: float = 1.0,
) -> Molecule:
    mol = parse_zmat_text_local(_zmat_text(geom_lines, var_lines))
    return _scale_molecule(mol, factor)


def _split_inline_zmat_variables(lines: Iterable[str]) -> tuple[list[str], list[str]]:
    geom_lines: list[str] = []
    var_lines: list[str] = []
    for line in lines:
        if "=" in line:
            var_lines.append(line)
        else:
            geom_lines.append(line)
    return geom_lines, var_lines


def _split_blank_sections(text: str) -> list[list[str]]:
    sections = re.split(r"\n[ \t]*\n", text.strip())
    return [
        [line.strip() for line in section.splitlines() if line.strip()]
        for section in sections
        if section.strip()
    ]


def _is_charge_multiplicity_line(line: str) -> bool:
    parts = line.replace(",", " ").split()
    if len(parts) < 2:
        return False
    try:
        int(parts[0])
        int(parts[1])
    except ValueError:
        return False
    return True


def _element_from_symbol_or_number(token: str) -> Element:
    try:
        return get_element_by_number(int(token))
    except ValueError:
        return get_element(token)


def parse_turbomole_coord(filepath: str | Path) -> Molecule:
    """Parse a Turbomole `coord` file.

    Coordinates are in Bohr by default; `$coord angs` switches to Angstrom.
    """
    text = Path(filepath).read_text()
    factor = BOHR_TO_ANGSTROM
    in_block = False
    block_lines: list[str] = []
    for raw in text.splitlines():
        s = raw.strip()
        if not in_block:
            low = s.lower()
            if low == "$coord" or low.startswith("$coord "):
                in_block = True
                factor = _unit_factor_from_tokens(low.split()[1:], default_bohr=True)
            continue
        if s.startswith("$"):
            break
        block_lines.append(s)
    return _molecule_from_cartesian_lines(
        block_lines,
        factor=factor,
        symbol_index=3,
        coord_start=0,
        error=f"Turbomole coord: no atoms found in {filepath!s}",
    )


def parse_orca_input(filepath: str | Path) -> Molecule:
    """Parse the geometry block of an Orca input file.

    Supported geometry forms:
      * `* xyz CHARGE MULT ... *` — Cartesian coordinates (default).
      * `* xyzfile CHARGE MULT path.xyz` — external reference, resolved
        relative to the input file's directory.
      * `* gzmt CHARGE MULT ... *` — Gaussian-flavor Z-matrix.
      * `* int CHARGE MULT ... *` — Orca-flavor Z-matrix (always 7 fields
        per atom: `Sym refA refB refC dist angle dihedral`, with `0`
        placeholders for unused refs on atoms 1-3).

    A `! Bohrs` (or `Bohr`) keyword anywhere in the keyword section switches
    distances (Cartesian or Z-matrix) to atomic units; angles remain in
    degrees. The `%coords ... pardef ... end end` block form and `{var}`
    parameter substitution are not supported.
    """
    filepath = Path(filepath)
    text = filepath.read_text()

    if re.search(r"(?mi)^\s*%compound\b", text):
        raise ValueError("Orca input: `%compound` scripts are not supported")
    if re.search(r"(?mi)^\s*%paras\b", text):
        raise ValueError("Orca input: `%paras` parameter blocks are not supported")
    if re.search(r"(?mi)^\s*%coords\b", text):
        raise ValueError("Orca input: `%coords ... end` block-form geometry is not supported")

    bohr = bool(re.search(r"(?mi)^\s*!.*\bbohrs?\b", text))
    factor = _bohr_factor(bohr)

    mode = "xyz"
    in_block = False
    block_lines: list[str] = []
    for raw in text.splitlines():
        s = raw.strip()
        if not in_block:
            low = s.lower()
            if not low.startswith("*"):
                continue
            tokens = s.replace("*", " * ").split()
            tokens_low = [t.lower() for t in tokens]
            if "xyzfile" in tokens_low:
                ref = tokens[-1]
                ref_path = (filepath.parent / ref).resolve()
                if not ref_path.exists():
                    raise ValueError(f"Orca xyzfile reference not found: {ref_path!s}")
                from .parsers import parse_xyz

                return parse_xyz(ref_path)
            if "gzmt" in tokens_low:
                mode = "gzmt"
                in_block = True
            elif "int" in tokens_low or "internal" in tokens_low:
                mode = "int"
                in_block = True
            elif "xyz" in tokens_low:
                mode = "xyz"
                in_block = True
            continue
        if s.startswith("*"):
            break
        if not s or s.startswith("#"):
            continue
        block_lines.append(s)

    if not block_lines:
        raise ValueError(f"Orca input: empty geometry block in {filepath!s}")

    if mode == "xyz":
        return _molecule_from_cartesian_lines(
            block_lines,
            factor=factor,
            error=f"Orca input: no atoms found in {filepath!s}",
        )

    if mode == "gzmt":
        return _parse_zmat_text_scaled(block_lines, factor=factor)

    # mode == "int": Orca 7-field-per-line Z-matrix.
    from .parsers import _zmat_to_cartesian

    symbols: list[str] = []
    refs: list[tuple[int, ...]] = []
    values: list[tuple[float, ...]] = []
    for i, s in enumerate(block_lines):
        parts = s.split()
        if len(parts) < 7:
            raise ValueError(
                f"Orca *int: expected 7 fields per atom (Sym A B C dist ang dih), "
                f"got {len(parts)} in line {s!r}"
            )
        symbols.append(parts[0])
        try:
            ref1 = int(parts[1]) - 1
            ref2 = int(parts[2]) - 1
            ref3 = int(parts[3]) - 1
            dist = float(parts[4]) * factor
            angle = float(parts[5])
            dih = float(parts[6])
        except ValueError as exc:
            raise ValueError(f"Orca *int: malformed atom line {s!r}") from exc
        if i == 0:
            refs.append(())
            values.append(())
        elif i == 1:
            refs.append((ref1,))
            values.append((dist,))
        elif i == 2:
            refs.append((ref1, ref2))
            values.append((dist, angle))
        else:
            refs.append((ref1, ref2, ref3))
            values.append((dist, angle, dih))

    positions = _zmat_to_cartesian(symbols, refs, values)
    return _make_molecule(
        [Atom(element=get_element(sym), position=pos) for sym, pos in zip(symbols, positions)]
    )


def parse_qchem_input(filepath: str | Path) -> Molecule:
    """Parse the `$molecule ... $end` block of a Q-Chem input file.

    Supports Cartesian coordinates and Z-matrix-with-variables (the same
    `name=value` substitution form Gaussian Z-matrices use). A `$rem` block
    setting `INPUT_BOHR  true` (or `1`/`yes`) switches Cartesian coordinates
    from Angstrom to atomic units.
    """
    filepath = Path(filepath)
    text = filepath.read_text()
    bohr = bool(
        re.search(
            r"(?mi)^\s*input_bohr\s+(?:true|1|yes|t)\b",
            text,
        )
    )
    factor = _bohr_factor(bohr)

    block_lines: list[str] = []
    in_block = False
    seen_header = False
    for raw in text.splitlines():
        s = raw.strip()
        low = s.lower()
        if not in_block:
            if low == "$molecule":
                in_block = True
                seen_header = False
            continue
        if low == "$end":
            break
        if not s or s.startswith("!"):
            continue
        if not seen_header:
            seen_header = True
            if low == "read" or low.startswith("read "):
                raise ValueError("Q-Chem input: `read` is not supported")
            # Charge / multiplicity line — skip.
            continue
        block_lines.append(s)

    if not block_lines:
        raise ValueError(f"Q-Chem input: empty `$molecule` block in {filepath!s}")

    atoms = _cartesian_atoms_from_lines(block_lines, factor=factor)
    if atoms:
        return _make_molecule(atoms)

    # Z-matrix fallback. Q-Chem allows `name = value` variable definitions
    # inside `$molecule` after the Z-matrix lines; split them out so
    # `parse_zmat_text` sees the canonical "geometry, blank line, vars" shape.
    geom_lines, var_lines = _split_inline_zmat_variables(block_lines)
    try:
        return _parse_zmat_text_scaled(geom_lines, var_lines)
    except Exception as exc:
        raise ValueError(
            f"Q-Chem input: no Cartesian atoms found in {filepath!s}; "
            f"Z-matrix fallback failed: {exc}"
        ) from exc


def parse_gaussian_input(filepath: str | Path) -> Molecule:
    """Parse the molecule specification of a Gaussian input file (`.com`/`.gjf`).

    Skips Link 0 (`%...`) and route (`#...`) sections, then the title block,
    then reads charge/multiplicity and Cartesian coordinates. Z-matrix style
    inputs are not supported here (use `parse_zmat`).
    """
    text = Path(filepath).read_text()
    if re.search(r"(?mi)^\s*--\s*Link1\s*--\s*$", text):
        raise ValueError("Gaussian input: multi-job `--Link1--` inputs are not supported")
    lines = text.splitlines()
    n = len(lines)
    i = 0

    while i < n and (not lines[i].strip() or lines[i].lstrip().startswith("%")):
        i += 1
    if i >= n or not lines[i].lstrip().startswith("#"):
        raise ValueError(f"Gaussian input: route section (`# ...`) not found in {filepath!s}")
    route_start = i
    while i < n and lines[i].strip():
        i += 1
    route = " ".join(lines[route_start:i])
    bohr = bool(re.search(r"(?i)\bunits?\s*=\s*(?:bohr|au)\b", route))
    factor = _bohr_factor(bohr)
    while i < n and not lines[i].strip():
        i += 1
    while i < n and lines[i].strip():
        i += 1
    while i < n and not lines[i].strip():
        i += 1
    if i >= n:
        raise ValueError(f"Gaussian input: charge/multiplicity not found in {filepath!s}")

    if not _is_charge_multiplicity_line(lines[i]):
        raise ValueError(f"Gaussian input: invalid charge/multiplicity line: {lines[i]!r}")
    i += 1

    geom_start = i
    atoms: list[Atom] = []
    while i < n and lines[i].strip():
        parts = _split_fields(lines[i])
        # "Sym x y z" or "Sym 0 x y z" (frozen flag) or "Sym(label) x y z"
        coord_start: int | None = None
        if len(parts) >= 5:
            try:
                int(parts[1])
                coord_start = 2
            except ValueError:
                coord_start = 1
        elif len(parts) >= 4:
            coord_start = 1
        if coord_start is not None:
            atom = _atom_from_fields(
                parts,
                symbol_index=0,
                coord_start=coord_start,
                factor=factor,
            )
            if atom is not None:
                atoms.append(atom)
        i += 1

    if atoms:
        return _make_molecule(atoms)

    # No Cartesian atoms — try Z-matrix style (atom lines reference earlier
    # atoms by 1-based index, possibly with a `name = value` variables block
    # separated by a blank line).
    geom_lines = []
    j = geom_start
    while j < n and lines[j].strip():
        geom_lines.append(lines[j])
        j += 1
    if geom_lines and any(re.search(r"\d", l) for l in geom_lines):
        # Look for a variables section after the geometry.
        var_block: list[str] = []
        k = j
        while k < n and not lines[k].strip():
            k += 1
        while k < n and lines[k].strip():
            var_block.append(lines[k])
            k += 1
        if var_block:
            # Convert Gaussian's `name value` form to `name=value` if needed.
            vars_norm = [(re.sub(r"^(\S+)\s+(\S+)\s*$", r"\1=\2", l)) for l in var_block]
        else:
            vars_norm = []
        try:
            return _parse_zmat_text_scaled(geom_lines, vars_norm)
        except Exception as exc:
            raise ValueError(
                f"Gaussian input: no Cartesian atoms found in {filepath!s}; "
                f"Z-matrix fallback failed: {exc}"
            ) from exc
    raise ValueError(f"Gaussian input: no atoms found in {filepath!s}")


def parse_zmat_text_local(text: str) -> Molecule:
    """Lazy wrapper around `parsers.parse_zmat_text` (avoids module-load cycle)."""
    from .parsers import parse_zmat_text

    return parse_zmat_text(text)


_NWCHEM_GEOM_DIRECTIVES = (
    "zmatrix",
    "zcoord",
    "constants",
    "variables",
    "center",
    "nocenter",
    "autosym",
    "noautosym",
    "autoz",
    "noautoz",
    "symmetry",
    "load",
    "print",
    "adjust",
    "system",
    "bqbq",
)


def parse_nwchem_input(filepath: str | Path) -> Molecule:
    """Parse the `geometry ... end` block of an NWChem input file."""
    text = Path(filepath).read_text()
    in_geom = False
    factor = 1.0
    block_lines: list[str] = []
    for raw in text.splitlines():
        s = raw.strip()
        low = s.lower()
        if not in_geom:
            if low.startswith("geometry") and (
                len(low) == len("geometry") or not low[len("geometry")].isalnum()
            ):
                in_geom = True
                if any(unit in low for unit in (" bohr", " atomic", " au")) or low.endswith(" au"):
                    factor = BOHR_TO_ANGSTROM
            continue
        if low == "end":
            break
        block_lines.append(s)
    return _molecule_from_cartesian_lines(
        block_lines,
        factor=factor,
        element_resolver=get_element_from_tag,
        skip_prefixes=_NWCHEM_GEOM_DIRECTIVES,
        error=f"NWChem input: no atoms found in {filepath!s}",
    )


_MOLCAS_BOHR_TOKENS = ("bohr", "a.u.", "au", "atomic")
_MOLCAS_UNSUPPORTED_TRANSFORMS = ("trans", "rot", "scale")


def parse_molcas_input(filepath: str | Path) -> Molecule:
    """Parse the `Coord` directive of a Molcas / OpenMolcas input file.

    Supports both `Coord = path/to/file.xyz` external references and inline
    XYZ-formatted blocks following a bare `Coord` (or any 4-char-stem
    abbreviation: `Coor`, `Coordinates`, ...) keyword. Defaults to Angstrom;
    the inline block's "comment" line may carry `bohr` / `a.u.` / `au` to
    switch units. A top-level `Bohr` directive in `&GATEWAY` does the same.
    The geometry-transform tokens `TRANS` / `ROT` / `SCALE` in the comment
    line are not supported and raise an error.
    """
    filepath = Path(filepath)
    text = filepath.read_text()
    bohr_global = bool(re.search(r"(?mi)^\s*bohr\s*$", text))
    lines = text.splitlines()
    n = len(lines)
    for i, raw in enumerate(lines):
        s = raw.strip()
        low = s.lower()
        # OpenMolcas keywords match by their first 4 characters (e.g. `COOR`).
        if not low.startswith("coor"):
            continue
        # Strip the keyword's alphabetic stem before looking for `=` / value.
        k = 0
        while k < len(s) and s[k].isalpha():
            k += 1
        rest = s[k:].lstrip()
        if rest.startswith("="):
            ref = rest[1:].strip()
            if not ref:
                continue
            ref_path = (filepath.parent / ref).resolve()
            if not ref_path.exists():
                raise ValueError(f"Molcas Coord reference not found: {ref_path!s}")
            from .parsers import parse_xyz

            return parse_xyz(ref_path)
        # Bare `Coord` — inline XYZ block on following lines.
        j = i + 1
        while j < n and not lines[j].strip():
            j += 1
        if j >= n:
            continue
        try:
            natoms = int(lines[j].strip())
        except ValueError:
            continue
        # Standard XYZ has count + comment + atoms, but Molcas inline blocks
        # sometimes omit the comment line entirely. First check the line
        # after the count for transform/unit tokens (those mean it IS a
        # comment line); otherwise decide based on whether it looks like
        # an atom line.
        next_line = lines[j + 1] if j + 1 < n else ""
        next_tokens = re.findall(r"[a-z.]+", next_line.lower())
        for tok in _MOLCAS_UNSUPPORTED_TRANSFORMS:
            if tok in next_tokens:
                raise ValueError(
                    f"Molcas input: `{tok.upper()}` geometry-transform tokens are not supported"
                )
        comment_bohr = any(tok in next_tokens for tok in _MOLCAS_BOHR_TOKENS)
        has_comment = comment_bohr or _parse_xyz_atom_line(next_line) is None
        factor = _bohr_factor(bohr_global or comment_bohr)
        j += 2 if has_comment else 1
        if j + natoms > n:
            raise ValueError(f"Molcas input: inline Coord block truncated (need {natoms} atoms)")
        atoms: list[Atom] = []
        for kk in range(natoms):
            atom = _atom_from_fields(lines[j + kk].split(), factor=factor)
            if atom is None:
                raise ValueError(f"Molcas input: malformed atom line: {lines[j + kk]!r}")
            atoms.append(atom)
        return _make_molecule(atoms)
    raise ValueError(f"Molcas input: no Coord directive found in {filepath!s}")


_MOLPRO_GEOM_RE = re.compile(r"\b(?:geometry|geom)\s*=\s*\{", re.IGNORECASE)


def parse_molpro_input(filepath: str | Path) -> Molecule:
    """Parse the `geometry={ ... }` block of a Molpro input file.

    If the block contains an XYZ-style payload (atom-count, comment, atoms),
    it is read as Angstrom. Otherwise the block is read as Cartesian
    coordinates: in Bohr by default, in Angstrom if a top-level `angstrom`
    directive precedes the block or if the block opens with an `angstrom`
    line.
    """
    text = Path(filepath).read_text()
    match = _MOLPRO_GEOM_RE.search(text)
    if match is None:
        raise ValueError(f"Molpro input: no geometry={{...}} block in {filepath!s}")
    block_start = match.end()
    block_end = text.find("}", block_start)
    if block_end == -1:
        raise ValueError(f"Molpro input: unterminated geometry block in {filepath!s}")

    raw_block = [l.strip() for l in text[block_start:block_end].splitlines()]
    non_comment = [l for l in raw_block if not l.startswith("!")]
    non_blank = [l for l in non_comment if l]
    if not non_blank:
        raise ValueError(f"Molpro input: empty geometry block in {filepath!s}")

    is_xyz_style = False
    atom_lines: list[str] = []
    try:
        natoms = int(non_blank[0])
        # XYZ-style: count, comment line (possibly blank), then `natoms` atoms.
        count_idx = non_comment.index(non_blank[0])
        atom_start = count_idx + 2  # skip count + comment
        if natoms > 0 and atom_start + natoms <= len(non_comment):
            candidates = [non_comment[atom_start + k] for k in range(natoms)]
            if all(_parse_xyz_atom_line(l) is not None for l in candidates):
                atom_lines = candidates
                is_xyz_style = True
    except (ValueError, IndexError):
        pass

    if is_xyz_style:
        factor = 1.0
    else:
        # Determine units. Molpro defaults to Bohr unless `angstrom` directive
        # is set globally, or appears as the first token in the block.
        global_angstrom = bool(
            re.search(
                r"^[ \t]*angstroms?[ \t;]*$",
                text[: match.start()],
                re.MULTILINE | re.IGNORECASE,
            )
        )
        first_parts = _split_fields(non_blank[0].lower())
        first = first_parts[0] if first_parts else ""
        if first in ("angstrom", "angstroms"):
            atom_lines = non_blank[1:]
            factor = 1.0
        elif first == "bohr":
            atom_lines = non_blank[1:]
            factor = BOHR_TO_ANGSTROM
        else:
            atom_lines = non_blank
            factor = 1.0 if global_angstrom else BOHR_TO_ANGSTROM

    atoms = _cartesian_atoms_from_lines(atom_lines, factor=factor, split_fields=True)
    if atoms:
        return _make_molecule(atoms)

    # Z-matrix fallback. Molpro supports numeric and label atom references;
    # variables are defined elsewhere in the file as `name=value [unit]` lines.
    return _parse_molpro_zmatrix(atom_lines, text, match.start(), filepath, factor)


_MOLPRO_VARIABLE_RE = re.compile(
    r"(?mi)^\s*([A-Za-z_]\w*)\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\b"
)


def _parse_molpro_zmatrix(
    atom_lines: list[str],
    full_text: str,
    geom_block_start: int,
    filepath: str | Path,
    factor: float,
) -> Molecule:
    """Parse a Molpro Z-matrix.

    Molpro inputs may reference atoms by 1-based index or by label. Dummy
    centers are kept while constructing coordinates, then omitted from the
    returned molecule.
    """
    from .parsers import _zmat_to_cartesian

    variables: dict[str, float] = {}
    pre = full_text[:geom_block_start]
    block_end = full_text.find("}", geom_block_start)
    post = full_text[block_end + 1 :] if block_end != -1 else ""
    vars_text = pre + "\n" + post
    for m in _MOLPRO_VARIABLE_RE.finditer(vars_text):
        variables[m.group(1)] = float(m.group(2))

    def resolve(token: str) -> float:
        try:
            return float(token)
        except ValueError:
            if token.startswith("-") and token[1:] in variables:
                return -variables[token[1:]]
            return variables[token]

    def clean_symbol(label: str) -> str:
        return "".join(ch for ch in label if ch.isalpha())

    def is_dummy(label: str) -> bool:
        return clean_symbol(label).lower() in {"q", "x"}

    def resolve_ref(
        token: str,
        label_to_idx: dict[str, int],
        max_numeric_ref: int,
        line: str,
    ) -> int:
        try:
            ref = int(token)
        except ValueError:
            if token in label_to_idx:
                return label_to_idx[token] - 1
            raise ValueError(f"Molpro Z-matrix: unknown atom label {token!r} on line {line!r}")
        if ref <= 0 or ref > max_numeric_ref:
            raise ValueError(f"Molpro Z-matrix: invalid atom reference {token!r} on line {line!r}")
        return ref - 1

    label_to_idx: dict[str, int] = {}
    symbols: list[str] = []
    dummy_flags: list[bool] = []
    refs: list[tuple[int, ...]] = []
    values: list[tuple[float, ...]] = []
    for i, line in enumerate(atom_lines):
        parts = _split_fields(line)
        if not parts:
            continue
        label = parts[0]
        label_to_idx.setdefault(label, len(label_to_idx) + 1)
        symbols.append(clean_symbol(label))
        dummy_flags.append(is_dummy(label))

        try:
            if i == 0:
                refs.append(())
                values.append(())
            elif i == 1:
                refs.append((resolve_ref(parts[1], label_to_idx, i, line),))
                values.append((resolve(parts[2]) * factor,))
            elif i == 2:
                refs.append(
                    (
                        resolve_ref(parts[1], label_to_idx, i, line),
                        resolve_ref(parts[3], label_to_idx, i, line),
                    )
                )
                values.append((resolve(parts[2]) * factor, resolve(parts[4])))
            else:
                refs.append(
                    (
                        resolve_ref(parts[1], label_to_idx, i, line),
                        resolve_ref(parts[3], label_to_idx, i, line),
                        resolve_ref(parts[5], label_to_idx, i, line),
                    )
                )
                values.append((resolve(parts[2]) * factor, resolve(parts[4]), resolve(parts[6])))
        except (IndexError, KeyError, ValueError) as exc:
            raise ValueError(
                f"Molpro input: no Cartesian atoms parsed from {filepath!s}; "
                f"Z-matrix fallback failed: {exc}"
            ) from exc

    try:
        positions = _zmat_to_cartesian(symbols, refs, values)
        atoms = [
            Atom(element=get_element(sym), position=pos)
            for sym, pos, dummy in zip(symbols, positions, dummy_flags)
            if not dummy
        ]
        if not atoms:
            raise ValueError("no non-dummy atoms parsed")
        return _make_molecule(atoms)
    except Exception as exc:
        raise ValueError(
            f"Molpro input: no Cartesian atoms parsed from {filepath!s}; "
            f"Z-matrix fallback failed: {exc}"
        ) from exc


def _mrcc_keyword_value(text: str, key: str) -> str | None:
    """Find `key=value` (case-insensitive) on its own line; return value or None."""
    m = re.search(rf"(?mi)^\s*{re.escape(key)}\s*=\s*(\S+)\s*$", text)
    return m.group(1).strip() if m else None


def parse_mrcc_input(filepath: str | Path) -> Molecule:
    """Parse the geometry of an MRCC `MINP` input file.

    Supports `geom=xyz`, `geom=zmat`, and `geom=tmol` modes. The `geom=mol`
    (V2000 MDL) variant is not supported. Default units depend on the mode
    (`xyz`/`tmol` → Angstrom, `zmat` → Angstrom); a top-level
    `unit=bohr|angs|angstrom` line overrides.
    """
    filepath = Path(filepath)
    text = filepath.read_text()

    mode_value = _mrcc_keyword_value(text, "geom")
    if mode_value is None:
        if re.search(r"(?mi)^\s*geom\s*$", text):
            mode = "zmat"
        else:
            raise ValueError(f"MRCC input: no `geom=` keyword in {filepath!s}")
    else:
        mode = mode_value.lower()

    unit_raw = _mrcc_keyword_value(text, "unit")
    unit = (unit_raw or "").lower()
    bohr = unit in ("bohr", "au", "a.u.", "atomic")

    # Locate the body following the geom= keyword; body ends at the next
    # `key=value` line at column 0 or end-of-file.
    geom_match = re.search(r"(?mi)^\s*geom\s*(?:=\s*\S+)?\s*$", text)
    if geom_match is None:
        raise ValueError(f"MRCC input: malformed `geom` keyword in {filepath!s}")
    after = text[geom_match.end() :]
    body_end = re.search(r"(?m)^[A-Za-z_][A-Za-z0-9_]*\s*=", after)
    body = after[: body_end.start()] if body_end else after

    if mode == "xyz":
        return _mrcc_parse_xyz_body(body, bohr, filepath)
    if mode == "tmol":
        return _mrcc_parse_tmol_body(body, bohr)
    if mode == "zmat":
        return _mrcc_parse_zmat_body(body)
    if mode == "mol":
        raise ValueError(f"MRCC `geom=mol` (V2000 MDL) is not supported; got {filepath!s}")
    raise ValueError(f"MRCC input: unknown geom mode {mode!r}")


def _mrcc_parse_xyz_body(body: str, bohr: bool, filepath: Path) -> Molecule:
    """Parse `geom=xyz` body: count, comment, then count atom lines."""
    lines = [l.strip() for l in body.splitlines()]
    non_blank_idx = [i for i, l in enumerate(lines) if l]
    if not non_blank_idx:
        raise ValueError(f"MRCC input: empty xyz body in {filepath!s}")
    try:
        natoms = int(lines[non_blank_idx[0]])
    except ValueError as exc:
        raise ValueError(
            f"MRCC input: expected atom count in xyz body, got {lines[non_blank_idx[0]]!r}"
        ) from exc
    # Skip the count and one comment line, then take natoms atom lines
    start = non_blank_idx[0] + 2
    factor = _bohr_factor(bohr)
    atoms: list[Atom] = []
    for k in range(natoms):
        if start + k >= len(lines):
            raise ValueError(f"MRCC xyz: truncated body (need {natoms} atoms)")
        line = lines[start + k]
        if not line:
            raise ValueError("MRCC xyz: unexpected blank inside atom block")
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"MRCC xyz: malformed atom line {line!r}")
        atom = _atom_from_fields(
            parts,
            factor=factor,
            element_resolver=_element_from_symbol_or_number,
        )
        if atom is None:
            raise ValueError(f"MRCC xyz: bad coords on line {line!r}")
        atoms.append(atom)
    return _make_molecule(atoms)


def _mrcc_parse_tmol_body(body: str, bohr: bool) -> Molecule:
    """Parse `geom=tmol` body: Turbomole-style `x y z element` lines."""
    factor = _bohr_factor(bohr)
    return _molecule_from_cartesian_lines(
        body.splitlines(),
        factor=factor,
        symbol_index=3,
        coord_start=0,
        stop_prefixes=("$",),
        error="MRCC tmol: no atoms parsed",
    )


def _mrcc_parse_zmat_body(body: str) -> Molecule:
    """Parse `geom=zmat` body via the shared Z-matrix routine."""
    from .parsers import parse_zmat_text

    return parse_zmat_text(body.strip())


_CFOUR_BLOCK_RE = re.compile(r"\*\s*(?:CFOUR|ACES2)\s*\(", re.IGNORECASE)


def parse_cfour_input(filepath: str | Path) -> Molecule:
    """Parse the geometry section of a CFOUR `ZMAT` input file.

    Layout: title line, geometry block, blank line, optional `name=value`
    variables, blank line, `*CFOUR(...)` keyword block. Default geometry
    mode is Z-matrix; `COORDINATES=CARTESIAN` (or `CART`) inside the
    keyword block selects Cartesian. Default units are Angstrom; `UNITS=BOHR`
    (or `AU`) overrides.
    """
    filepath = Path(filepath)
    text = filepath.read_text()

    cfour_match = _CFOUR_BLOCK_RE.search(text)
    if cfour_match is None:
        raise ValueError(f"CFOUR input: no `*CFOUR(...)` keyword block in {filepath!s}")
    closing = text.find(")", cfour_match.end())
    if closing == -1:
        raise ValueError("CFOUR input: unterminated `*CFOUR(...)` block")
    cfour_kv: dict[str, str] = {}
    for token in _split_fields(text[cfour_match.end() : closing]):
        if "=" in token:
            k, v = token.split("=", 1)
            cfour_kv[k.strip().lower()] = v.strip().lower()
    coord_mode = cfour_kv.get("coordinates") or cfour_kv.get("coord", "internal")
    units = cfour_kv.get("units", "angstrom")
    factor = _unit_factor(units)

    pre = text[: cfour_match.start()]
    # Split on blank lines: section 0 = title + geometry, section 1 = variables.
    section_lines = _split_blank_sections(pre)
    if not section_lines:
        raise ValueError(f"CFOUR input: empty geometry section in {filepath!s}")
    geom_lines = section_lines[0][1:]  # skip title
    var_lines = section_lines[1] if len(section_lines) > 1 else []
    if not geom_lines:
        raise ValueError(f"CFOUR input: empty geometry block in {filepath!s}")

    if coord_mode in ("cartesian", "cart"):
        return _molecule_from_cartesian_lines(
            geom_lines,
            factor=factor,
            error="CFOUR input: no Cartesian atoms parsed",
        )

    # Z-matrix mode (default).
    return _parse_zmat_text_scaled(geom_lines, var_lines, factor=factor)


_PSI4_MOLECULE_RE = re.compile(r"\bmolecule(?:\s+\w+)?\s*\{", re.IGNORECASE)
_PSI4_DIRECTIVES = (
    "symmetry",
    "no_reorient",
    "noreorient",
    "no_com",
    "nocom",
    "pubchem",
)
_PSI4_GHOST_RE = re.compile(r"^[Gg][Hh]\((.+)\)$")


def _psi4_element_from_tag(sym_raw: str) -> Element:
    ghost = _PSI4_GHOST_RE.match(sym_raw)
    if ghost:
        sym_raw = ghost.group(1)
    if "@" in sym_raw:
        sym_raw = sym_raw.split("@", 1)[0]
    return get_element(sym_raw)


def parse_psi4_input(filepath: str | Path) -> Molecule:
    """Parse the first `molecule { ... }` block of a Psi4 input file.

    Supports Cartesian and Z-matrix-with-variables (the `name=value`
    definitions inside the block become Z-matrix variables). Units default
    to Angstrom; an inline `units bohr|au` line switches. Fragment
    separators (`--`) are treated as concatenation (each fragment may
    carry its own `charge multiplicity` line, which is skipped). Ghost
    (`Gh(O)`) and isotope (`O@18.0`) decorations resolve to the underlying
    element.
    """
    text = Path(filepath).read_text()
    match = _PSI4_MOLECULE_RE.search(text)
    if match is None:
        raise ValueError(f"Psi4 input: no `molecule {{ ... }}` block in {filepath!s}")
    block_start = match.end()
    block_end = text.find("}", block_start)
    if block_end == -1:
        raise ValueError(f"Psi4 input: unterminated molecule block in {filepath!s}")
    body = text[block_start:block_end]

    factor = 1.0
    for raw in body.splitlines():
        s = raw.strip().lower()
        if s.startswith("units"):
            tokens = s.split()
            if len(tokens) >= 2 and tokens[1] in ("bohr", "au", "a.u.", "atomic"):
                factor = BOHR_TO_ANGSTROM
            break

    atom_or_zmat_lines: list[str] = []
    var_lines: list[str] = []
    for raw in body.splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        low = s.lower()
        if s == "--":
            continue
        if low.startswith("units"):
            continue
        if low.startswith(_PSI4_DIRECTIVES):
            continue
        if "=" in s:
            var_lines.append(s)
            continue
        if _is_charge_multiplicity_line(s):
            continue
        atom_or_zmat_lines.append(s)

    atoms = _cartesian_atoms_from_lines(
        atom_or_zmat_lines,
        factor=factor,
        element_resolver=_psi4_element_from_tag,
    )
    if atoms:
        return _make_molecule(atoms)

    # Z-matrix fallback: standard `name=value` substitution.
    try:
        return _parse_zmat_text_scaled(atom_or_zmat_lines, var_lines)
    except KeyError as exc:
        missing = str(exc.args[0]) if exc.args else "unknown"
        raise ValueError(
            f"Psi4 input: dynamic molecule variables are not supported (unresolved {missing!r})"
        ) from exc
    except Exception as exc:
        raise ValueError(
            f"Psi4 input: no Cartesian atoms parsed from {filepath!s}; "
            f"Z-matrix fallback failed: {exc}"
        ) from exc


_GAMESS_DATA_RE = re.compile(r"\$DATA\b", re.IGNORECASE)
_GAMESS_END_RE = re.compile(r"\$END\b", re.IGNORECASE)
_GAMESS_UNITS_RE = re.compile(r"\bUNITS\s*=\s*(\w+)", re.IGNORECASE)


def parse_gamess_input(filepath: str | Path) -> Molecule:
    """Parse the geometry from a GAMESS (US) input file's `$DATA` group.

    Supports `COORD=CART` and `COORD=UNIQUE` with `C1` symmetry only. The
    second column of each atom line is the **nuclear charge** (a float),
    not an element symbol — the element is inferred from that column.
    Units default to Angstrom; `UNITS=BOHR` (or `AU`) inside `$CONTRL`
    overrides.

    Non-`C1` symmetry inputs (which require additional master-frame cards)
    raise an error rather than silently mis-parse.
    """
    filepath = Path(filepath)
    text = filepath.read_text()

    data_match = _GAMESS_DATA_RE.search(text)
    if data_match is None:
        raise ValueError(f"GAMESS input: no `$DATA` group in {filepath!s}")
    after = text[data_match.end() :]
    end_match = _GAMESS_END_RE.search(after)
    if end_match is None:
        raise ValueError("GAMESS input: `$DATA` group not terminated by `$END`")
    block = after[: end_match.start()]

    units_match = _GAMESS_UNITS_RE.search(text)
    bohr = bool(units_match and units_match.group(1).lower() in ("bohr", "au", "a.u.", "atomic"))
    factor = _bohr_factor(bohr)

    lines = block.splitlines()
    idx = 0
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    if idx + 1 >= len(lines):
        raise ValueError(f"GAMESS input: `$DATA` group too short in {filepath!s}")
    # idx = title (skip), idx+1 = point group
    group = lines[idx + 1].strip().split()[0].upper() if lines[idx + 1].strip() else "C1"
    if group != "C1":
        raise ValueError(f"GAMESS input: only `C1` symmetry is supported (got {group!r})")

    atoms: list[Atom] = []
    for line in lines[idx + 2 :]:
        s = line.strip()
        if not s:
            continue
        parts = s.split()
        if len(parts) < 5 or not parts[0][:1].isalpha():
            # Basis-set sub-block lines — skip.
            continue
        try:
            nuc = float(parts[1])
            x = float(parts[2]) * factor
            y = float(parts[3]) * factor
            z = float(parts[4]) * factor
        except ValueError:
            continue
        elem = get_element_by_number(int(round(nuc)))
        atoms.append(Atom(element=elem, position=np.array([x, y, z])))

    if not atoms:
        raise ValueError(f"GAMESS input: no atoms parsed from {filepath!s}")
    return _make_molecule(atoms)


_JAGUAR_ZMAT_OPEN_RE = re.compile(r"^\s*&\s*zmat\b", re.IGNORECASE | re.MULTILINE)
_JAGUAR_ZMAT_CLOSE_RE = re.compile(r"^\s*&\s*$", re.MULTILINE)
_JAGUAR_IUNIT_RE = re.compile(r"\biunit\s*=\s*(\d+)", re.IGNORECASE)


def parse_jaguar_input(filepath: str | Path) -> Molecule:
    """Parse the first `&zmat ... &` block of a Jaguar (Schrödinger) input file.

    Cartesian-only: Z-matrix-with-variables and multi-`&zmat` inputs are not
    supported (the parser uses the first `&zmat` block and raises if any
    atom line lacks four numeric columns). Atom tags like `O1`/`H2` resolve
    to elements via the leading symbol prefix. Default units are Angstrom;
    `iunit=0` in the `&gen` block switches to Bohr.

    Note: the official Jaguar manual is paywalled; this parser is based on
    the open-source Schrödinger Python API source. Pre-validate exotic
    inputs before relying on the result.
    """
    filepath = Path(filepath)
    text = filepath.read_text()

    open_match = _JAGUAR_ZMAT_OPEN_RE.search(text)
    if open_match is None:
        raise ValueError(f"Jaguar input: no `&zmat` section in {filepath!s}")
    after = text[open_match.end() :]
    close_match = _JAGUAR_ZMAT_CLOSE_RE.search(after)
    if close_match is None:
        raise ValueError("Jaguar input: `&zmat` not terminated by `&`")
    body = after[: close_match.start()]

    iunit_match = _JAGUAR_IUNIT_RE.search(text)
    factor = 1.0
    if iunit_match and iunit_match.group(1) == "0":
        factor = BOHR_TO_ANGSTROM

    atoms: list[Atom] = []
    for raw in body.splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 4:
            continue
        atom = _atom_from_fields(parts, factor=factor)
        if atom is None:
            raise ValueError(
                f"Jaguar input: non-numeric coords on line {s!r} — "
                f"Z-matrix-with-variables is not supported (Cartesian only)"
            )
        atoms.append(atom)

    if not atoms:
        raise ValueError(f"Jaguar input: no atoms parsed from {filepath!s}")
    return _make_molecule(atoms)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

QC_INPUT_KINDS = (
    "orca-input",
    "qchem-input",
    "gaussian-input",
    "nwchem-input",
    "turbomole-input",
    "molcas-input",
    "molpro-input",
    "mrcc-input",
    "cfour-input",
    "psi4-input",
    "gamess-input",
    "jaguar-input",
)

QC_INPUT_EXTENSION_MAP: dict[str, str] = {
    ".gjf": "gaussian-input",
    ".nw": "nwchem-input",
    ".nwi": "nwchem-input",
    ".qcin": "qchem-input",
}

QC_INPUT_AMBIGUOUS_SUFFIXES = (
    ".com",
    ".inp",
    ".in",
    ".input",
    ".minp",
    ".mrcc",
    ".dat",
)

_QC_INPUT_PARSERS = {
    "orca-input": parse_orca_input,
    "qchem-input": parse_qchem_input,
    "gaussian-input": parse_gaussian_input,
    "nwchem-input": parse_nwchem_input,
    "turbomole-input": parse_turbomole_coord,
    "molcas-input": parse_molcas_input,
    "molpro-input": parse_molpro_input,
    "mrcc-input": parse_mrcc_input,
    "cfour-input": parse_cfour_input,
    "psi4-input": parse_psi4_input,
    "gamess-input": parse_gamess_input,
    "jaguar-input": parse_jaguar_input,
}


def detect_qc_input_by_extension(path: Path) -> str | None:
    """Return a QC input kind based on extension/filename only, without reading the file.

    Matches `coord` (Turbomole), `MINP` (MRCC), and the unambiguous extension
    map. For ambiguous extensions (`.com`, `.inp`, `.in`, `.input`), use
    `sniff_qc_input` instead.
    """
    name_lower = path.name.lower()
    if name_lower == "coord" and path.is_file():
        return "turbomole-input"
    if name_lower == "minp" and path.is_file():
        return "mrcc-input"
    if name_lower == "zmat" and path.is_file():
        return "cfour-input"
    return QC_INPUT_EXTENSION_MAP.get(path.suffix.lower())


def sniff_qc_input(path: Path) -> str | None:
    """Inspect the first ~200 lines of `path` and identify a QC input format.

    Returns one of `QC_INPUT_KINDS` or `None` if no marker is found. Decisive
    markers (`$coord`, `$molecule`, `***`, `&GATEWAY`, `geometry={`,
    `* xyz`/`xyzfile`) short-circuit; otherwise weaker signals (`!` keyword
    lines, `%` Link 0 lines, `#`-route, NWChem top-level directives) are
    accumulated and the most common wins.
    """
    signals: dict[str, int] = {}

    def bump(fmt: str) -> None:
        signals[fmt] = signals.get(fmt, 0) + 1

    with open(path) as f:
        for i, raw in enumerate(f):
            if i >= 200:
                break
            s = raw.strip()
            if not s:
                continue
            low = s.lower()

            if low == "$coord" or low.startswith("$coord "):
                return "turbomole-input"
            if low == "$molecule" or low == "$rem":
                return "qchem-input"
            if s.startswith("***") and not s.startswith("****"):
                # Molpro title line is `***,title`/`*** title`. Reject 4+
                # asterisk markers (Gaussian-output dividers, Orca's
                # `****END OF INPUT****`).
                trailing = s[3:].strip()
                if trailing.startswith(",") or (trailing and trailing[0].isalnum()):
                    return "molpro-input"
            if low.startswith("*cfour(") or low.startswith("*aces2("):
                return "cfour-input"
            if re.match(r"^\s*molecule(?:\s+\w+)?\s*\{", low):
                return "psi4-input"
            if re.match(r"^\s*\$(?:data|contrl|basis|system|scf)\b", low):
                return "gamess-input"
            if re.match(r"^\s*&\s*(?:zmat|gen)\b", low):
                return "jaguar-input"
            if low.startswith("&gateway") or low.startswith("&seward"):
                return "molcas-input"
            compact = low.replace(" ", "")
            if "geometry={" in compact or "geom={" in compact:
                return "molpro-input"
            if low.startswith("* xyz") or low.startswith("*xyz") or " xyzfile" in low:
                return "orca-input"
            if re.match(r"\s*geom\s*=\s*(?:xyz|zmat|tmol|mol)\s*$", low):
                return "mrcc-input"

            if low.startswith("memory,"):
                bump("molpro-input")
            elif s.startswith("!") and len(s) >= 2 and not s.startswith("!!"):
                bump("orca-input")
            elif s.startswith("%") and "=" in s:
                bump("gaussian-input")
            elif s.startswith("#") and (
                low.startswith("# ")
                or low.startswith("#p ")
                or low.startswith("#t ")
                or low.startswith("#n ")
            ):
                bump("gaussian-input")
            else:
                tokens = low.split()
                if tokens and tokens[0] == "geometry":
                    bump("nwchem-input")
                elif tokens and tokens[0] in ("start", "title", "echo"):
                    bump("nwchem-input")

    if not signals:
        return None
    return max(signals, key=signals.__getitem__)


def parse_qc_input(filepath: str | Path, kind: str) -> Molecule:
    """Dispatch parsing to the parser registered for `kind`."""
    parser = _QC_INPUT_PARSERS.get(kind)
    if parser is None:
        raise ValueError(f"Unknown QC input kind: {kind!r}")
    return parser(Path(filepath))
