from __future__ import annotations

from pathlib import Path

import numpy as np

from moltui.elements import Atom, Molecule, get_element, get_element_by_number


def _parse_float(token: str) -> float:
    return float(token.replace("D", "E").replace("d", "e"))


def _is_int_line(tokens: list[str]) -> bool:
    if not tokens:
        return False
    try:
        for token in tokens:
            int(token)
    except ValueError:
        return False
    return True


def _scale_lattice(raw_lattice: np.ndarray, scale_tokens: list[str]) -> tuple[np.ndarray, float]:
    if len(scale_tokens) != 1:
        raise ValueError("VASP parser supports a single POSCAR scale factor")
    scale = _parse_float(scale_tokens[0])
    if scale < 0:
        target_volume = abs(scale)
        raw_volume = abs(float(np.linalg.det(raw_lattice)))
        if raw_volume <= 0:
            raise ValueError("Cannot apply negative POSCAR volume scale to zero-volume lattice")
        factor = (target_volume / raw_volume) ** (1.0 / 3.0)
    else:
        factor = scale
    return raw_lattice * factor, factor


def _symbols_from_comment(comment: str, counts: list[int]) -> list[str]:
    symbols: list[str] = []
    for token in comment.replace(",", " ").split():
        alpha = "".join(ch for ch in token if ch.isalpha())
        if not alpha:
            continue
        elem = get_element(alpha)
        if elem.atomic_number != 0:
            symbols.append(elem.symbol)
    if len(symbols) >= len(counts):
        return symbols[: len(counts)]
    # VASP 4 files do not contain an unambiguous species line. Keep atom count
    # and positions useful, while making the limitation explicit in symbols.
    return ["X"] * len(counts)


def parse_poscar(filepath: str | Path) -> Molecule:
    """Parse a VASP POSCAR/CONTCAR-like structure file.

    Returned coordinates and row-vector lattice are in Angstrom. VASP 4
    atom-count-only files are supported by deriving species from the comment
    line when possible; otherwise unknown ``X`` atoms are used.
    """
    path = Path(filepath)
    lines = path.read_text().splitlines()
    if len(lines) < 8:
        raise ValueError("POSCAR/CONTCAR file is too short")

    comment = lines[0].strip()
    scale_tokens = lines[1].split()
    try:
        raw_lattice = np.array(
            [[_parse_float(tok) for tok in lines[i].split()[:3]] for i in range(2, 5)],
            dtype=np.float64,
        )
    except (ValueError, IndexError) as exc:
        raise ValueError("Invalid POSCAR lattice vectors") from exc
    if raw_lattice.shape != (3, 3):
        raise ValueError("Invalid POSCAR lattice vectors")
    lattice, cart_scale = _scale_lattice(raw_lattice, scale_tokens)

    idx = 5
    first = lines[idx].split()
    if _is_int_line(first):
        counts = [int(tok) for tok in first]
        symbols = _symbols_from_comment(comment, counts)
        idx += 1
    else:
        symbols = first
        idx += 1
        if idx >= len(lines):
            raise ValueError("POSCAR missing atom counts")
        count_tokens = lines[idx].split()
        if not _is_int_line(count_tokens):
            raise ValueError("POSCAR atom-count line must contain integers")
        counts = [int(tok) for tok in count_tokens]
        idx += 1

    if len(symbols) != len(counts):
        raise ValueError("POSCAR symbol and count lines have different lengths")
    if any(c < 0 for c in counts):
        raise ValueError("POSCAR atom counts must be non-negative")

    if idx >= len(lines):
        raise ValueError("POSCAR missing coordinate mode")
    if lines[idx].strip().lower().startswith("s"):
        idx += 1
        if idx >= len(lines):
            raise ValueError("POSCAR missing coordinate mode after Selective dynamics")

    mode = lines[idx].strip().lower()
    if not mode:
        raise ValueError("POSCAR missing coordinate mode")
    direct = mode.startswith("d")
    cartesian = mode.startswith("c") or mode.startswith("k")
    if not direct and not cartesian:
        raise ValueError(f"Unsupported POSCAR coordinate mode: {lines[idx]!r}")
    idx += 1

    expanded_symbols = [
        sym for sym, count in zip(symbols, counts, strict=True) for _ in range(count)
    ]
    n_atoms = len(expanded_symbols)
    if len(lines) < idx + n_atoms:
        raise ValueError("POSCAR ended before all atom coordinates were read")

    atoms: list[Atom] = []
    for atom_idx, symbol in enumerate(expanded_symbols):
        parts = lines[idx + atom_idx].split()
        if len(parts) < 3:
            raise ValueError("Invalid POSCAR coordinate line")
        coords = np.array([_parse_float(parts[0]), _parse_float(parts[1]), _parse_float(parts[2])])
        pos = coords @ lattice if direct else coords * cart_scale
        if symbol.strip().isdigit():
            element = get_element_by_number(int(symbol))
        else:
            element = get_element(symbol)
        atoms.append(Atom(element=element, position=pos.astype(np.float64)))

    mol = Molecule(atoms=atoms, bonds=[], lattice=lattice)
    mol.detect_bonds_auto()
    return mol
