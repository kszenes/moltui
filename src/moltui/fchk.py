"""Gaussian formatted-checkpoint (.fchk / .fch) reader.

Maps the fixed-format Fortran sections into the same :class:`~moltui.gto.GtoBasis`
representation that :mod:`moltui.gto` builds from Molden, so the existing AO
evaluation and orbital pipeline work unchanged.

Only sections needed for geometry + MO visualisation are parsed; energies,
gradients, Hessians, and post-SCF densities are ignored.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .elements import Atom, Molecule, get_element, get_element_by_number
from .gto import BOHR_TO_ANGSTROM, GtoBasis, PrimShell
from .molden import OrbitalData


@dataclass
class _Section:
    type_code: str  # "I" (int), "R" (real), "C", "L", "H"
    is_array: bool
    scalar: int | float | str | None
    array: np.ndarray | None


def _parse_float(token: str) -> float:
    return float(token.replace("D", "E").replace("d", "e"))


def _read_sections(path: Path) -> dict[str, _Section]:
    """Parse an fchk into a dict ``{label: _Section}``.

    fchk lines have one of two shapes:

    - scalar:  ``<label up to col 40><type code><value>``
    - array:   ``<label up to col 40><type code> N=<count>`` followed by
               5 reals/line (E16.8) or 6 ints/line (I12).

    Strings (``C``, ``H``) are read but stored as raw text — we never use them.
    """
    sections: dict[str, _Section] = {}
    with open(path) as f:
        # Skip the first two header lines (title + route/method/basis).
        f.readline()
        f.readline()
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].rstrip("\n")
        if len(line) < 43:
            i += 1
            continue
        label = line[:40].strip()
        # The type code lives at column 43 (0-indexed 43..44).
        type_code = line[43:45].strip()
        rest = line[47:].strip()
        if type_code not in ("I", "R", "C", "L", "H"):
            i += 1
            continue

        if rest.startswith("N="):
            count = int(rest[2:].strip())
            i += 1
            tokens: list[str] = []
            while len(tokens) < count and i < len(lines):
                tokens.extend(lines[i].split())
                i += 1
            tokens = tokens[:count]
            if type_code == "I":
                arr = np.array([int(t) for t in tokens], dtype=np.int64)
            elif type_code == "R":
                arr = np.array([_parse_float(t) for t in tokens], dtype=np.float64)
            else:
                # Strings: store as object array; not used.
                arr = np.array(tokens, dtype=object)
            sections[label] = _Section(type_code, True, None, arr)
        else:
            if type_code == "I":
                value: int | float | str = int(rest)
            elif type_code == "R":
                value = _parse_float(rest)
            else:
                value = rest
            sections[label] = _Section(type_code, False, value, None)
            i += 1

    return sections


def _require_array(sections: dict[str, _Section], label: str) -> np.ndarray:
    sec = sections.get(label)
    if sec is None or sec.array is None:
        raise ValueError(f"fchk file missing required section: {label!r}")
    return sec.array


def _require_scalar(sections: dict[str, _Section], label: str) -> int | float:
    sec = sections.get(label)
    if sec is None or sec.scalar is None:
        raise ValueError(f"fchk file missing required scalar: {label!r}")
    assert isinstance(sec.scalar, (int, float))
    return sec.scalar


def parse_fchk_atoms(filepath: str | Path) -> Molecule:
    """Read just the geometry from a Gaussian fchk file."""
    sections = _read_sections(Path(filepath))
    z = _require_array(sections, "Atomic numbers").astype(int)
    coords_bohr = _require_array(sections, "Current cartesian coordinates").reshape(-1, 3)
    if coords_bohr.shape[0] != z.shape[0]:
        raise ValueError("fchk atom count and coordinate count disagree")
    atoms = [
        Atom(
            element=get_element_by_number(int(z[i])),
            position=coords_bohr[i] * BOHR_TO_ANGSTROM,
        )
        for i in range(z.shape[0])
    ]
    mol = Molecule(atoms=atoms, bonds=[])
    mol.detect_bonds()
    return mol


def _build_shells(
    shell_types: np.ndarray,
    nprim: np.ndarray,
    shell_to_atom: np.ndarray,
    exponents: np.ndarray,
    contractions: np.ndarray,
    sp_contractions: np.ndarray | None,
    atom_coords_bohr: np.ndarray,
) -> tuple[list[PrimShell], list[int]]:
    """Build PrimShells from fchk shell tables.

    Also returns ``ao_layout``: one entry per *AO column* in the fchk MO matrix,
    holding the index in the returned shell list. Used to map Gaussian's
    SP-shell layout (1 S followed by 3 P, all under one shell) onto our split
    representation, and (in the future) to handle any Gaussian-specific
    intra-shell ordering differences.
    """
    shells: list[PrimShell] = []
    ao_to_shell: list[int] = []
    prim_offset = 0

    for s_idx, s_type in enumerate(shell_types):
        np_s = int(nprim[s_idx])
        atom_idx = int(shell_to_atom[s_idx]) - 1  # fchk uses 1-based
        center = atom_coords_bohr[atom_idx]
        exps = exponents[prim_offset : prim_offset + np_s]
        coeffs = contractions[prim_offset : prim_offset + np_s]

        if int(s_type) == -1:
            # SP shell: split into one S + one P sharing the exponents.
            if sp_contractions is None:
                raise ValueError("fchk has SP shell but no P(S=P) Contraction coefficients block")
            p_coeffs = sp_contractions[prim_offset : prim_offset + np_s]
            s_shell_idx = len(shells)
            shells.append(PrimShell(center=center, l=0, exponents=exps, coefficients=coeffs))
            p_shell_idx = len(shells)
            shells.append(PrimShell(center=center, l=1, exponents=exps, coefficients=p_coeffs))
            ao_to_shell.append(s_shell_idx)  # 1 S AO
            ao_to_shell.extend([p_shell_idx] * 3)  # 3 P AOs
        else:
            l = abs(int(s_type))
            shell_idx = len(shells)
            shells.append(PrimShell(center=center, l=l, exponents=exps, coefficients=coeffs))
            # Number of AOs depends on pure vs cartesian for l>=2.
            spherical = int(s_type) < 0
            n_ao = (2 * l + 1) if (spherical or l <= 1) else (l + 1) * (l + 2) // 2
            ao_to_shell.extend([shell_idx] * n_ao)

        prim_offset += np_s

    return shells, ao_to_shell


def parse_fchk(filepath: str | Path) -> GtoBasis:
    """Read a Gaussian fchk file into a :class:`GtoBasis`.

    Restricted calcs use only Alpha sections; unrestricted concatenates Alpha
    then Beta MOs (matching how :func:`moltui.gto.parse_molden` lays them out).
    """
    path = Path(filepath)
    sections = _read_sections(path)

    # --- Geometry ------------------------------------------------------------
    z = _require_array(sections, "Atomic numbers").astype(int)
    atom_coords = _require_array(sections, "Current cartesian coordinates").reshape(-1, 3)
    atom_symbols = [get_element_by_number(int(zi)).symbol for zi in z]

    # --- Basis ---------------------------------------------------------------
    shell_types = _require_array(sections, "Shell types").astype(int)
    nprim = _require_array(sections, "Number of primitives per shell").astype(int)
    shell_to_atom = _require_array(sections, "Shell to atom map").astype(int)
    exponents = _require_array(sections, "Primitive exponents")
    contractions = _require_array(sections, "Contraction coefficients")
    sp_sec = sections.get("P(S=P) Contraction coefficients")
    sp_contractions = sp_sec.array if sp_sec is not None else None

    shells, _ = _build_shells(
        shell_types,
        nprim,
        shell_to_atom,
        exponents,
        contractions,
        sp_contractions,
        atom_coords,
    )

    # Pure vs Cartesian per angular momentum. Gaussian uses 0 → pure (spherical),
    # nonzero → Cartesian. Default to pure when the field is absent (matches
    # Gaussian's default for d/f when not stated).
    pure_d = int(_require_scalar(sections, "Pure/Cartesian d shells")) == 0
    pure_f = int(_require_scalar(sections, "Pure/Cartesian f shells")) == 0
    spherical = {2: pure_d, 3: pure_f, 4: pure_f}

    # --- MOs -----------------------------------------------------------------
    nbasis = int(_require_scalar(sections, "Number of basis functions"))
    alpha_e = _require_array(sections, "Alpha Orbital Energies")
    alpha_c = _require_array(sections, "Alpha MO coefficients")
    n_alpha = alpha_e.shape[0]
    if alpha_c.shape[0] != n_alpha * nbasis:
        raise ValueError("Alpha MO coefficients length does not match nmo * nbasis")
    # fchk stores MOs row-major: MO 1 (nbasis values), MO 2, ... — so reshape
    # as (nmo, nbasis) and transpose to the (nao, nmo) layout used downstream.
    alpha_coef = alpha_c.reshape(n_alpha, nbasis).T

    beta_sec = sections.get("Beta MO coefficients")
    if beta_sec is not None and beta_sec.array is not None:
        beta_e = _require_array(sections, "Beta Orbital Energies")
        n_beta = beta_e.shape[0]
        beta_coef = beta_sec.array.reshape(n_beta, nbasis).T
        mo_energies = np.concatenate([alpha_e, beta_e])
        mo_coef = np.concatenate([alpha_coef, beta_coef], axis=1)
        mo_spins = ["Alpha"] * n_alpha + ["Beta"] * n_beta
        n_alpha_occ = int(_require_scalar(sections, "Number of alpha electrons"))
        n_beta_occ = int(_require_scalar(sections, "Number of beta electrons"))
        occ_alpha = np.array([1.0 if i < n_alpha_occ else 0.0 for i in range(n_alpha)])
        occ_beta = np.array([1.0 if i < n_beta_occ else 0.0 for i in range(n_beta)])
        mo_occupations = np.concatenate([occ_alpha, occ_beta])
    else:
        n_alpha_occ = int(_require_scalar(sections, "Number of alpha electrons"))
        n_beta_occ = int(_require_scalar(sections, "Number of beta electrons"))
        mo_energies = alpha_e
        mo_coef = alpha_coef
        mo_spins = ["Alpha"] * n_alpha
        # RHF: doubly-occupied up to n_alpha (== n_beta).
        occ = np.zeros(n_alpha)
        occ[:n_alpha_occ] = 2.0 if n_alpha_occ == n_beta_occ else 1.0
        mo_occupations = occ

    return GtoBasis(
        atom_symbols=atom_symbols,
        atom_coords_bohr=atom_coords,
        shells=shells,
        mo_energies=mo_energies,
        mo_occupations=mo_occupations,
        mo_coefficients=mo_coef,
        mo_symmetries=["A"] * mo_energies.shape[0],
        mo_spins=mo_spins,
        spherical=spherical,
    )


def load_fchk_data(filepath: str | Path) -> OrbitalData:
    """Load a fchk file as :class:`OrbitalData` for the orbital viewer."""
    basis = parse_fchk(filepath)
    atoms = [
        Atom(
            element=get_element(symbol),
            position=basis.atom_coords_bohr[i] * BOHR_TO_ANGSTROM,
        )
        for i, symbol in enumerate(basis.atom_symbols)
    ]
    molecule = Molecule(atoms=atoms, bonds=[])
    molecule.detect_bonds()
    return OrbitalData.from_gto_basis(basis, molecule)
