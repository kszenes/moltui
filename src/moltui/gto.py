"""Pure-NumPy Gaussian-type orbital evaluation and Molden file parsing.

Replaces PySCF for molden file support — no external chemistry dependencies needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from pathlib import Path

import numpy as np

BOHR_TO_ANGSTROM = 0.529177249

SHELL_LABEL_TO_L = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4}


def _parse_float(token: str) -> float:
    """Parse a float token, including Fortran D exponents."""
    return float(token.replace("D", "E").replace("d", "e"))


def _section_tag(line: str) -> str | None:
    """Return normalized section tag from a line like '[Atoms] AU'."""
    stripped = line.strip()
    if not stripped.startswith("["):
        return None
    end = stripped.find("]")
    if end < 0:
        return None
    return stripped[1:end].strip().lower()


# ---------------------------------------------------------------------------
# Data structures ------------------------------------------------------------
# ---------------------------------------------------------------------------


@dataclass
class PrimShell:
    """One contracted shell on a single atom."""

    center: np.ndarray  # (3,) in Bohr
    l: int  # angular momentum
    exponents: np.ndarray  # (nprim,)
    coefficients: np.ndarray  # (nprim,)


@dataclass
class GtoBasis:
    """Contracted GTO shells, nuclear coordinates, and MO coefficients.

    Field layout and AO/MO ordering follow the Molden file format (as from
    :func:`parse_molden` and typical PySCF Molden exports). Other sources (e.g. TREXIO)
    are mapped into this representation for :func:`eval_gto` and MO evaluation.
    """

    atom_symbols: list[str]
    atom_coords_bohr: np.ndarray  # (natom, 3)
    shells: list[PrimShell]
    mo_energies: np.ndarray
    mo_occupations: np.ndarray
    mo_coefficients: np.ndarray  # (nao, nmo)
    mo_symmetries: list[str]
    mo_spins: list[str]
    spherical: dict[int, bool]  # l -> True if spherical
    frequencies: np.ndarray | None = None  # (n_modes,)
    normal_modes: np.ndarray | None = None  # (n_modes, n_atoms, 3)


# ---------------------------------------------------------------------------
# Molden parser --------------------------------------------------------------
# ---------------------------------------------------------------------------


def _n_components(l: int, spherical: bool) -> int:
    return 2 * l + 1 if spherical else (l + 1) * (l + 2) // 2


def parse_molden(filepath: str | Path) -> GtoBasis:
    """Parse a Molden file into a :class:`GtoBasis`."""
    filepath = Path(filepath)
    lines = filepath.read_text().splitlines()

    atom_symbols: list[str] = []
    atom_coords: list[list[float]] = []
    shells: list[PrimShell] = []
    mo_energies: list[float] = []
    mo_occupations: list[float] = []
    mo_symmetries: list[str] = []
    mo_spins: list[str] = []
    mo_coeffs_list: list[list[float]] = []
    frequencies: list[float] = []
    normal_modes: list[np.ndarray] = []
    # Default: cartesian for d and above
    spherical: dict[int, bool] = {2: False, 3: False, 4: False}

    i = 0
    is_angstrom = False
    while i < len(lines):
        line = lines[i]
        tag = _section_tag(line)

        # --- [Atoms] section ---
        if tag == "atoms":
            is_angstrom = "angs" in line.lower()
            i += 1
            while i < len(lines) and _section_tag(lines[i]) is None:
                parts = lines[i].split()
                if len(parts) >= 6:
                    atom_symbols.append(parts[0])
                    x, y, z = _parse_float(parts[3]), _parse_float(parts[4]), _parse_float(parts[5])
                    if is_angstrom:
                        x /= BOHR_TO_ANGSTROM
                        y /= BOHR_TO_ANGSTROM
                        z /= BOHR_TO_ANGSTROM
                    atom_coords.append([x, y, z])
                i += 1
            continue

        # --- [GTO] section ---
        if tag == "gto":
            i += 1
            while i < len(lines) and _section_tag(lines[i]) is None:
                parts = lines[i].split()

                def starts_atom(parts: list[str]) -> bool:
                    # Standard Molden format: <atom_idx> 0
                    # Molcas format:          <atom_idx>
                    is_molcas = len(parts) == 1 and parts[0].isdigit()
                    is_standard = len(parts) == 2 and parts[0].isdigit() and parts[1] == "0"
                    return is_molcas or is_standard

                if starts_atom(parts):
                    atom_idx = int(parts[0]) - 1  # 1-based to 0-based
                    i += 1
                    # Read shells for this atom
                    while i < len(lines):
                        sline = lines[i].strip()
                        if not sline or _section_tag(sline) is not None:
                            break
                        sparts = sline.split()
                        if starts_atom(sparts):
                            break
                        shell_type = sparts[0].lower()
                        nprim = int(sparts[1])
                        exps = []
                        coeffs = []
                        coeffs_p = []
                        for _ in range(nprim):
                            i += 1
                            pparts = lines[i].split()
                            exps.append(_parse_float(pparts[0]))
                            coeffs.append(_parse_float(pparts[1]))
                            if shell_type == "sp":
                                if len(pparts) < 3:
                                    raise ValueError(
                                        "Invalid [GTO] sp primitive line in Molden file"
                                    )
                                coeffs_p.append(_parse_float(pparts[2]))
                        i += 1
                        if shell_type == "sp":
                            shells.append(
                                PrimShell(
                                    center=np.array(atom_coords[atom_idx]),
                                    l=0,
                                    exponents=np.array(exps),
                                    coefficients=np.array(coeffs),
                                )
                            )
                            shells.append(
                                PrimShell(
                                    center=np.array(atom_coords[atom_idx]),
                                    l=1,
                                    exponents=np.array(exps),
                                    coefficients=np.array(coeffs_p),
                                )
                            )
                        else:
                            if shell_type not in SHELL_LABEL_TO_L:
                                raise ValueError(
                                    f"Unsupported shell type '{shell_type}' in [GTO] section"
                                )
                            l = SHELL_LABEL_TO_L[shell_type]
                            shells.append(
                                PrimShell(
                                    center=np.array(atom_coords[atom_idx]),
                                    l=l,
                                    exponents=np.array(exps),
                                    coefficients=np.array(coeffs),
                                )
                            )
                else:
                    i += 1
            continue

        # --- Spherical/Cartesian flags ---
        if tag == "5d":
            spherical[2] = True
            spherical[3] = True
        if tag == "7f":
            spherical[3] = True
        if tag == "5d10f":
            spherical[2] = True
            spherical[3] = False
        if tag == "5d7f":
            spherical[2] = True
            spherical[3] = True
        if tag == "9g":
            spherical[4] = True

        # --- [MO] section ---
        if tag == "mo":
            i += 1
            current_coeffs: list[float] = []
            while i < len(lines):
                mline = lines[i].strip()
                if _section_tag(mline) is not None:
                    break
                if mline.startswith("Sym="):
                    if current_coeffs:
                        mo_coeffs_list.append(current_coeffs)
                        current_coeffs = []
                    sym = mline.split("=")[1].strip()
                    # Remove leading digits
                    sym = sym.lstrip("0123456789")
                    # Capitalize (except trailing g/u)
                    if sym and sym[-1].lower() in ("g", "u"):
                        sym_core = sym[:-1].capitalize()
                        sym = sym_core + sym[-1].lower()
                    else:
                        sym = sym.capitalize()
                    mo_symmetries.append(sym)
                elif mline.startswith("Ene="):
                    mo_energies.append(_parse_float(mline.split("=")[1].strip()))
                elif mline.startswith("Spin="):
                    mo_spins.append(mline.split("=")[1].strip())
                elif mline.startswith("Occup="):
                    mo_occupations.append(_parse_float(mline.split("=")[1].strip()))
                elif mline and mline[0].isdigit():
                    parts = mline.split()
                    current_coeffs.append(_parse_float(parts[1]))
                i += 1
            if current_coeffs:
                mo_coeffs_list.append(current_coeffs)
            continue

        # --- [FR-COORD] section ---
        if tag == "fr-coord":
            i += 1
            fr_symbols: list[str] = []
            fr_coords: list[list[float]] = []
            while i < len(lines) and _section_tag(lines[i]) is None:
                parts = lines[i].split()
                if len(parts) >= 4:
                    fr_symbols.append(parts[0])
                    fr_coords.append(
                        [_parse_float(parts[1]), _parse_float(parts[2]), _parse_float(parts[3])]
                    )
                i += 1
            if not atom_symbols and fr_symbols:
                atom_symbols = fr_symbols
                atom_coords = fr_coords
            continue

        # --- [FREQ] section ---
        if tag == "freq":
            i += 1
            while i < len(lines) and _section_tag(lines[i]) is None:
                for token in lines[i].split():
                    frequencies.append(_parse_float(token))
                i += 1
            continue

        # --- [FR-NORM-COORD] section ---
        if tag == "fr-norm-coord":
            i += 1
            mode_vectors: list[list[float]] | None = None
            while i < len(lines) and _section_tag(lines[i]) is None:
                mline = lines[i].strip()
                if not mline:
                    i += 1
                    continue
                if mline.lower().startswith("vibration"):
                    if mode_vectors:
                        normal_modes.append(np.array(mode_vectors, dtype=np.float64))
                    mode_vectors = []
                    i += 1
                    continue
                parts = mline.split()
                floats = []
                for token in parts:
                    try:
                        floats.append(_parse_float(token))
                    except ValueError:
                        pass
                if len(floats) >= 3:
                    if mode_vectors is None:
                        mode_vectors = []
                    mode_vectors.append(floats[-3:])
                i += 1
            if mode_vectors:
                normal_modes.append(np.array(mode_vectors, dtype=np.float64))
            continue

        i += 1

    coords_arr = np.array(atom_coords)
    if mo_coeffs_list:
        mo_coeff_arr = np.array(mo_coeffs_list, dtype=np.float64).T  # (nao, nmo)
    else:
        mo_coeff_arr = np.zeros((0, 0), dtype=np.float64)

    freqs_arr = np.array(frequencies, dtype=np.float64) if frequencies else None
    if normal_modes:
        mode_lengths = {mode.shape[0] for mode in normal_modes}
        if len(mode_lengths) != 1:
            raise ValueError("Inconsistent mode vector lengths in [FR-NORM-COORD]")
        normal_modes_arr: np.ndarray | None = np.stack(normal_modes, axis=0)
    else:
        normal_modes_arr = None

    return GtoBasis(
        atom_symbols=atom_symbols,
        atom_coords_bohr=coords_arr,
        shells=shells,
        mo_energies=np.array(mo_energies),
        mo_occupations=np.array(mo_occupations),
        mo_coefficients=mo_coeff_arr,
        mo_symmetries=mo_symmetries,
        mo_spins=mo_spins,
        spherical=spherical,
        frequencies=freqs_arr,
        normal_modes=normal_modes_arr,
    )


# ---------------------------------------------------------------------------
# Real solid harmonics -------------------------------------------------------
# ---------------------------------------------------------------------------


def real_solid_harmonics(
    l: int, dx: np.ndarray, dy: np.ndarray, dz: np.ndarray
) -> list[np.ndarray]:
    """Return 2l+1 real solid harmonics evaluated at (dx, dy, dz).

    Ordering follows the Molden convention:
      l=0: 1
      l=1: x, y, z
      l=2 [5d]: d0, d+1, d-1, d+2, d-2
      l=3 [7f]: f0, f+1, f-1, f+2, f-2, f+3, f-3
      l=4 [9g]: g0, g+1, g-1, g+2, g-2, g+3, g-3, g+4, g-4
    """
    r2 = dx * dx + dy * dy + dz * dz

    if l == 0:
        return [np.ones_like(dx)]

    if l == 1:
        return [dx, dy, dz]

    if l == 2:
        sqrt3 = np.sqrt(3.0)
        return [
            (3.0 * dz * dz - r2) / 2.0,
            sqrt3 * dx * dz,
            sqrt3 * dy * dz,
            sqrt3 / 2.0 * (dx * dx - dy * dy),
            sqrt3 * dx * dy,
        ]

    if l == 3:
        sqrt6 = np.sqrt(6.0)
        sqrt15 = np.sqrt(15.0)
        sqrt2_5 = np.sqrt(2.5)
        return [
            dz * (5.0 * dz * dz - 3.0 * r2) / 2.0,
            sqrt6 / 4.0 * dx * (5.0 * dz * dz - r2),
            sqrt6 / 4.0 * dy * (5.0 * dz * dz - r2),
            sqrt15 / 2.0 * dz * (dx * dx - dy * dy),
            sqrt15 * dx * dy * dz,
            sqrt2_5 / 2.0 * dx * (dx * dx - 3.0 * dy * dy),
            sqrt2_5 / 2.0 * dy * (3.0 * dx * dx - dy * dy),
        ]

    if l == 4:
        xx = dx * dx
        yy = dy * dy
        zz = dz * dz
        r4 = r2 * r2
        sqrt10 = np.sqrt(10.0)
        sqrt5 = np.sqrt(5.0)
        sqrt70 = np.sqrt(70.0)
        sqrt35 = np.sqrt(35.0)
        return [
            (35.0 * zz * zz - 30.0 * zz * r2 + 3.0 * r4) / 8.0,
            sqrt10 / 4.0 * dx * dz * (7.0 * zz - 3.0 * r2),
            sqrt10 / 4.0 * dy * dz * (7.0 * zz - 3.0 * r2),
            sqrt5 / 4.0 * (xx - yy) * (7.0 * zz - r2),
            sqrt5 / 2.0 * dx * dy * (7.0 * zz - r2),
            sqrt70 / 4.0 * dx * dz * (xx - 3.0 * yy),
            sqrt70 / 4.0 * dy * dz * (3.0 * xx - yy),
            sqrt35 / 8.0 * (xx * xx - 6.0 * xx * yy + yy * yy),
            sqrt35 / 2.0 * dx * dy * (xx - yy),
        ]

    raise NotImplementedError(f"Angular momentum l={l} not implemented")


def cartesian_harmonics(l: int, dx: np.ndarray, dy: np.ndarray, dz: np.ndarray) -> list[np.ndarray]:
    """Return cartesian harmonic monomials in Molden's documented order."""
    if l == 0:
        return [np.ones_like(dx)]

    if l == 1:
        return [dx, dy, dz]

    if l == 2:
        return [dx * dx, dy * dy, dz * dz, dx * dy, dx * dz, dy * dz]

    if l == 3:
        xx = dx * dx
        yy = dy * dy
        zz = dz * dz
        return [
            xx * dx,  # xxx
            yy * dy,  # yyy
            zz * dz,  # zzz
            dx * yy,  # xyy
            xx * dy,  # xxy
            xx * dz,  # xxz
            dx * zz,  # xzz
            dy * zz,  # yzz
            yy * dz,  # yyz
            dx * dy * dz,  # xyz
        ]

    if l == 4:
        xx = dx * dx
        yy = dy * dy
        zz = dz * dz
        return [
            xx * xx,  # xxxx
            yy * yy,  # yyyy
            zz * zz,  # zzzz
            xx * dx * dy,  # xxxy
            xx * dx * dz,  # xxxz
            yy * dy * dx,  # yyyx
            yy * dy * dz,  # yyyz
            zz * dz * dx,  # zzzx
            zz * dz * dy,  # zzzy
            xx * yy,  # xxyy
            xx * zz,  # xxzz
            yy * zz,  # yyzz
            xx * dy * dz,  # xxyz
            yy * dx * dz,  # yyxz
            zz * dx * dy,  # zzxy
        ]

    raise NotImplementedError(f"Cartesian harmonics for l={l} not implemented")


# ---------------------------------------------------------------------------
# GTO normalization ----------------------------------------------------------
# ---------------------------------------------------------------------------


def _dfact(n: int) -> int:
    """Double factorial (2l-1)!!."""
    if n <= 0:
        return 1
    result = 1
    for k in range(n, 0, -2):
        result *= k
    return result


def _prim_norm(l: int, alpha: float) -> float:
    """Normalization factor for a primitive GTO with angular momentum l."""
    return (
        (2.0 * alpha / pi) ** 0.75 * (4.0 * alpha) ** (l / 2.0) / np.sqrt(float(_dfact(2 * l - 1)))
    )


def _contraction_norm(l: int, exponents: np.ndarray, coefficients: np.ndarray) -> float:
    """Normalization for a contracted shell so that <phi|phi>=1."""
    df = float(_dfact(2 * l - 1))
    s = 0.0
    for i in range(len(exponents)):
        ni = _prim_norm(l, exponents[i])
        for j in range(len(exponents)):
            nj = _prim_norm(l, exponents[j])
            aij = exponents[i] + exponents[j]
            s += (
                coefficients[i]
                * coefficients[j]
                * ni
                * nj
                * (pi / aij) ** 1.5
                * df
                / (2.0 * aij) ** l
            )
    return 1.0 / np.sqrt(s)


# ---------------------------------------------------------------------------
# GTO evaluation (optimized) ------------------------------------------------
# ---------------------------------------------------------------------------


@dataclass
class _PreparedShell:
    center_idx: int
    l: int
    alphas: np.ndarray
    weighted_coeffs: np.ndarray  # c_k * N_k * cnorm
    ncomp: int
    spherical: bool


def _prepare_shells(
    shells: list[PrimShell], spherical: dict[int, bool]
) -> tuple[list[_PreparedShell], np.ndarray]:
    """Precompute normalization and deduplicate centers."""
    center_map: dict[tuple, int] = {}
    centers_list: list[np.ndarray] = []
    prepared: list[_PreparedShell] = []

    for shell in shells:
        key = tuple(shell.center)
        if key not in center_map:
            center_map[key] = len(centers_list)
            centers_list.append(shell.center)
        cidx = center_map[key]

        l = shell.l
        is_sph = spherical.get(l, l <= 1)
        ncomp = _n_components(l, is_sph)

        cnorm = _contraction_norm(l, shell.exponents, shell.coefficients)
        wc = np.array(
            [
                shell.coefficients[k] * _prim_norm(l, shell.exponents[k]) * cnorm
                for k in range(len(shell.exponents))
            ]
        )

        prepared.append(
            _PreparedShell(
                center_idx=cidx,
                l=l,
                alphas=shell.exponents,
                weighted_coeffs=wc,
                ncomp=ncomp,
                spherical=is_sph,
            )
        )

    centers = np.array(centers_list)
    return prepared, centers


# Screening threshold: exp(-x) < 1e-15 when x > ~34.5
_SCREEN_THRESH = 34.5


def eval_gto(
    shells: list[PrimShell],
    grid_points: np.ndarray,
    spherical: dict[int, bool],
) -> np.ndarray:
    """Evaluate all AO basis functions on grid_points. Returns (npoints, nao).

    Uses precomputed norms, batched exp, shared centers, and screening.
    """
    npts = grid_points.shape[0]

    prepared, centers = _prepare_shells(shells, spherical)

    # Displacements and squared distances for each unique center
    dr_all = grid_points[np.newaxis, :, :] - centers[:, np.newaxis, :]  # (nc, npts, 3)
    r2_all = np.sum(dr_all * dr_all, axis=2)  # (nc, npts)

    total_ao = sum(s.ncomp for s in prepared)
    result = np.zeros((npts, total_ao))

    col = 0
    for shell in prepared:
        l = shell.l
        cidx = shell.center_idx
        ncomp = shell.ncomp
        r2 = r2_all[cidx]

        # Screen using the most diffuse exponent
        alpha_min = shell.alphas.min()
        mask = alpha_min * r2 < _SCREEN_THRESH
        n_active = mask.sum()

        if n_active == 0:
            col += ncomp
            continue

        if n_active < npts:
            r2_active = r2[mask]
            dr_active = dr_all[cidx, mask, :]
        else:
            r2_active = r2
            dr_active = dr_all[cidx]
            mask = None

        # Vectorized contraction over primitives
        exps = np.exp(-shell.alphas[:, np.newaxis] * r2_active[np.newaxis, :])
        radial = shell.weighted_coeffs @ exps

        if shell.spherical:
            harmonics = real_solid_harmonics(l, dr_active[:, 0], dr_active[:, 1], dr_active[:, 2])
        else:
            harmonics = cartesian_harmonics(l, dr_active[:, 0], dr_active[:, 1], dr_active[:, 2])
        if len(harmonics) != ncomp:
            raise ValueError(
                f"Harmonic component mismatch for l={l}: expected {ncomp}, got {len(harmonics)} "
                f"(spherical={shell.spherical})"
            )

        if mask is None:
            for m in range(ncomp):
                result[:, col + m] = radial * harmonics[m]
        else:
            for m in range(ncomp):
                result[mask, col + m] = radial * harmonics[m]

        col += ncomp

    return result
