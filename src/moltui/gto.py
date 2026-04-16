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
class MoldenBasis:
    """Parsed molden basis set and MO data (no PySCF objects)."""

    atom_symbols: list[str]
    atom_coords_bohr: np.ndarray  # (natom, 3)
    shells: list[PrimShell]
    mo_energies: np.ndarray
    mo_occupations: np.ndarray
    mo_coefficients: np.ndarray  # (nao, nmo)
    mo_symmetries: list[str]
    mo_spins: list[str]
    spherical: dict[int, bool]  # l -> True if spherical


# ---------------------------------------------------------------------------
# Molden parser --------------------------------------------------------------
# ---------------------------------------------------------------------------


def _n_components(l: int, spherical: bool) -> int:
    return 2 * l + 1 if spherical else (l + 1) * (l + 2) // 2


def parse_molden(filepath: str | Path) -> MoldenBasis:
    """Parse a Molden file into atoms, basis, and MO data."""
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
    # Default: cartesian for d and above
    spherical: dict[int, bool] = {2: False, 3: False, 4: False}

    i = 0
    is_angstrom = False
    while i < len(lines):
        line = lines[i]

        # --- [Atoms] section ---
        if "[Atoms]" in line:
            is_angstrom = "Angs" in line
            i += 1
            while i < len(lines) and not lines[i].startswith("["):
                parts = lines[i].split()
                if len(parts) >= 6:
                    atom_symbols.append(parts[0])
                    x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
                    if is_angstrom:
                        x /= BOHR_TO_ANGSTROM
                        y /= BOHR_TO_ANGSTROM
                        z /= BOHR_TO_ANGSTROM
                    atom_coords.append([x, y, z])
                i += 1
            continue

        # --- [GTO] section ---
        if "[GTO]" in line:
            i += 1
            while i < len(lines) and not lines[i].startswith("["):
                parts = lines[i].split()
                if len(parts) >= 2 and parts[0].isdigit():
                    atom_idx = int(parts[0]) - 1  # 1-based to 0-based
                    i += 1
                    # Read shells for this atom
                    while i < len(lines):
                        sline = lines[i].strip()
                        if not sline or sline.startswith("["):
                            break
                        sparts = sline.split()
                        if len(sparts) >= 2 and sparts[0].isdigit():
                            break
                        shell_type = sparts[0].lower()
                        nprim = int(sparts[1])
                        exps = []
                        coeffs = []
                        for _ in range(nprim):
                            i += 1
                            pparts = lines[i].split()
                            exps.append(float(pparts[0]))
                            coeffs.append(float(pparts[1]))
                        i += 1
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
        if "[5d]" in line or "[5D]" in line:
            spherical[2] = True
        if "[7f]" in line or "[7F]" in line:
            spherical[3] = True
        if "[9g]" in line or "[9G]" in line:
            spherical[4] = True

        # --- [MO] section ---
        if "[MO]" in line:
            i += 1
            current_coeffs: list[float] = []
            while i < len(lines):
                mline = lines[i].strip()
                if mline.startswith("["):
                    break
                if mline.startswith("Sym="):
                    if current_coeffs:
                        mo_coeffs_list.append(current_coeffs)
                        current_coeffs = []
                    mo_symmetries.append(mline.split("=")[1].strip())
                elif mline.startswith("Ene="):
                    mo_energies.append(float(mline.split("=")[1].strip()))
                elif mline.startswith("Spin="):
                    mo_spins.append(mline.split("=")[1].strip())
                elif mline.startswith("Occup="):
                    mo_occupations.append(float(mline.split("=")[1].strip()))
                elif mline and mline[0].isdigit():
                    parts = mline.split()
                    current_coeffs.append(float(parts[1]))
                i += 1
            if current_coeffs:
                mo_coeffs_list.append(current_coeffs)
            continue

        i += 1

    coords_arr = np.array(atom_coords)
    mo_coeff_arr = np.array(mo_coeffs_list).T  # (nao, nmo)

    return MoldenBasis(
        atom_symbols=atom_symbols,
        atom_coords_bohr=coords_arr,
        shells=shells,
        mo_energies=np.array(mo_energies),
        mo_occupations=np.array(mo_occupations),
        mo_coefficients=mo_coeff_arr,
        mo_symmetries=mo_symmetries,
        mo_spins=mo_spins,
        spherical=spherical,
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

    raise NotImplementedError(f"Angular momentum l={l} not implemented")


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
                center_idx=cidx, l=l, alphas=shell.exponents, weighted_coeffs=wc, ncomp=ncomp
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

        harmonics = real_solid_harmonics(l, dr_active[:, 0], dr_active[:, 1], dr_active[:, 2])

        if mask is None:
            for m in range(ncomp):
                result[:, col + m] = radial * harmonics[m]
        else:
            for m in range(ncomp):
                result[mask, col + m] = radial * harmonics[m]

        col += ncomp

    return result
