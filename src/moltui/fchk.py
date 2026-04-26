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

    # --- Normal modes -------------------------------------------------------
    # Prefer Gaussian's own analysis (Vib-Modes/Vib-E2) when present; otherwise
    # fall back to diagonalising the Hessian (Cartesian Force Constants) — the
    # default Freq job writes the Hessian even without SaveNormalModes.
    frequencies, normal_modes = _read_vib_blocks(sections, n_atoms=len(atom_symbols))
    if normal_modes is None and sections.get("Cartesian Force Constants") is not None:
        frequencies, normal_modes = _freqs_from_hessian_sections(sections)

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
        frequencies=frequencies,
        normal_modes=normal_modes,
    )


def _read_vib_blocks(
    sections: dict[str, _Section], n_atoms: int
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Extract frequencies and normal-mode displacements when present.

    Gaussian writes the Hessian (``Cartesian Force Constants``) by default,
    but ``Vib-Modes`` and ``Vib-E2`` only appear when the calculation used
    ``Freq=SaveNormalModes`` (or ``Freq=HPModes``). Hessian-only fchks return
    ``(None, None)`` here; diagonalising the Hessian is left to a follow-up.

    ``Vib-E2`` is a flat array of 14 scalars per mode (frequency, reduced mass,
    force constant, IR/Raman intensities, ...). Only the first scalar per mode
    is meaningful here.
    """
    modes_sec = sections.get("Vib-Modes")
    e2_sec = sections.get("Vib-E2")
    if modes_sec is None or e2_sec is None:
        return None, None

    n_modes_sec = sections.get("Number of Normal Modes")
    if n_modes_sec is None or n_modes_sec.scalar is None:
        return None, None
    n_modes = int(n_modes_sec.scalar)  # type: ignore[arg-type]

    e2 = e2_sec.array
    modes = modes_sec.array
    if e2 is None or modes is None:
        return None, None
    if e2.shape[0] < 14 * n_modes:
        raise ValueError("Vib-E2 block is shorter than 14 * n_modes")
    if modes.shape[0] != n_modes * n_atoms * 3:
        raise ValueError("Vib-Modes length does not match n_modes * n_atoms * 3")

    # Vib-E2 is laid out as 14 contiguous blocks of n_modes (frequencies first,
    # then reduced masses, force constants, IR/Raman intensities, ...). So the
    # first n_modes values are the frequencies — not strided every 14th.
    frequencies = e2[:n_modes].copy()
    normal_modes = modes.reshape(n_modes, n_atoms, 3).copy()
    return frequencies, normal_modes


# 1 sqrt(Hartree / (Bohr^2 * amu)) expressed in cm^-1 (CODATA 2018 derived).
_HARTREE_BOHR2_AMU_TO_CM1 = 5140.4843


def _unpack_lower_triangle(packed: np.ndarray, n: int) -> np.ndarray:
    """Expand a row-by-row lower-triangle vector to a full symmetric (n, n).

    Gaussian writes ``H[0,0], H[1,0], H[1,1], H[2,0], H[2,1], H[2,2], ...`` —
    same packing convention as LAPACK's ``L`` packed format.
    """
    expected = n * (n + 1) // 2
    if packed.shape[0] != expected:
        raise ValueError(f"packed length {packed.shape[0]} != n(n+1)/2 = {expected}")
    full = np.zeros((n, n), dtype=np.float64)
    idx = 0
    for i in range(n):
        full[i, : i + 1] = packed[idx : idx + i + 1]
        idx += i + 1
    return full + full.T - np.diag(np.diag(full))


def _trans_rot_basis(coords_bohr: np.ndarray, masses_amu: np.ndarray) -> np.ndarray:
    """Orthonormal basis (3N, k) for trans+rot in mass-weighted Cartesians.

    ``k`` is 6 for non-linear molecules and 5 for linear ones; the rank is
    determined by QR (dependent rotation columns get tiny diagonal entries).
    """
    n = coords_bohr.shape[0]
    sqrt_m = np.sqrt(masses_amu)
    com = (masses_amu[:, None] * coords_bohr).sum(axis=0) / masses_amu.sum()
    r = coords_bohr - com

    basis = np.zeros((3 * n, 6), dtype=np.float64)
    for axis in range(3):
        v = np.zeros((n, 3))
        v[:, axis] = sqrt_m
        basis[:, axis] = v.reshape(-1)
    for axis in range(3):
        e = np.zeros(3)
        e[axis] = 1.0
        v = np.cross(r, e) * sqrt_m[:, None]
        basis[:, 3 + axis] = v.reshape(-1)

    q, r_qr = np.linalg.qr(basis)
    keep = np.abs(np.diag(r_qr)) > 1e-8
    return q[:, keep]


def compute_freqs_from_hessian(filepath: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Compute vibrational frequencies + Cartesian mode displacements from the
    ``Cartesian Force Constants`` block alone.

    Returns ``(frequencies_cm1, modes)`` with shapes ``(n_modes,)`` and
    ``(n_modes, n_atoms, 3)``. Negative frequencies indicate imaginary modes.
    Independent of any precomputed ``Vib-Modes``/``Vib-E2`` data — exposed so
    tests can cross-validate against those blocks when both are present.
    """
    return _freqs_from_hessian_sections(_read_sections(Path(filepath)))


def _freqs_from_hessian_sections(
    sections: dict[str, _Section],
) -> tuple[np.ndarray, np.ndarray]:
    n_atoms = int(_require_scalar(sections, "Number of atoms"))
    coords = _require_array(sections, "Current cartesian coordinates").reshape(n_atoms, 3)
    masses = _require_array(sections, "Real atomic weights")
    hess_packed = _require_array(sections, "Cartesian Force Constants")

    n = 3 * n_atoms
    H = _unpack_lower_triangle(hess_packed, n)

    sqrt_m_per_dof = np.repeat(np.sqrt(masses), 3)
    H_mw = H / np.outer(sqrt_m_per_dof, sqrt_m_per_dof)

    tr_basis = _trans_rot_basis(coords, masses)
    P = np.eye(n) - tr_basis @ tr_basis.T
    H_proj = P @ H_mw @ P

    eigvals, eigvecs = np.linalg.eigh(H_proj)

    # Drop the 5 or 6 trans/rot eigenvalues — keep the 3N-k modes with the
    # largest |λ|, then re-sort by signed eigenvalue (negative → imaginary).
    n_keep = n - tr_basis.shape[1]
    keep_idx = np.argsort(np.abs(eigvals))[-n_keep:]
    keep_idx = keep_idx[np.argsort(eigvals[keep_idx])]
    eigvals = eigvals[keep_idx]
    eigvecs = eigvecs[:, keep_idx]

    frequencies = np.sign(eigvals) * np.sqrt(np.abs(eigvals)) * _HARTREE_BOHR2_AMU_TO_CM1

    # Un-mass-weight to Cartesian displacements, then renormalise per mode so
    # that Σ|L|² = 1 (Gaussian's convention for the Vib-Modes block).
    L_cart = (eigvecs / sqrt_m_per_dof[:, None]).reshape(n_atoms, 3, -1)
    norms = np.sqrt((L_cart**2).sum(axis=(0, 1)))
    L_cart = L_cart / norms[None, None, :]
    modes = np.transpose(L_cart, (2, 0, 1))
    return frequencies, modes


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

    normal_modes = None
    mode_frequencies = None
    if basis.normal_modes is not None:
        if basis.normal_modes.shape[1] != len(atoms):
            raise ValueError("fchk normal modes do not match atom count")
        normal_modes = basis.normal_modes * BOHR_TO_ANGSTROM
        if basis.frequencies is not None:
            mode_frequencies = basis.frequencies[: normal_modes.shape[0]]

    return OrbitalData.from_gto_basis(
        basis,
        molecule,
        mode_frequencies=mode_frequencies,
        normal_modes=normal_modes,
    )
