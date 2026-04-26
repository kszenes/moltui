"""Equivalence tests for the Gaussian formatted-checkpoint parser.

Each fchk in ``data/gaussian/`` has a sibling ``.molden`` produced by
``iodata.dump_one``. The parsers should agree on geometry, MO energies, and the
value of each MO at arbitrary points in space.

Comparing on a grid (rather than coefficient-by-coefficient) is the right
abstraction because the two formats use different AO orderings and the
generalized SP shells in STO-3G fchk files are split into segmented S+P shells
when iodata writes the molden. Only the *physical* MO ought to match.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from moltui.gto import eval_gto, prepare_gto_cache

DATA = Path(__file__).resolve().parent.parent / "data" / "gaussian"

PAIRS = [
    "h2o_sto3g",
    "ch3_hf_sto3g",
    "water_ccpvdz_pure_hf_g03",
    "o2_cc_pvtz_cart",
]


def _eval_basis_on_points(basis, points_bohr: np.ndarray) -> np.ndarray:
    """Return MO values shape (npts, nmo) for ``basis`` at ``points_bohr``."""
    cache = prepare_gto_cache(basis.shells, basis.spherical)
    ao = eval_gto(basis.shells, points_bohr, basis.spherical, prepared_cache=cache)
    return ao.astype(np.float64) @ basis.mo_coefficients


def _sample_points(coords_bohr: np.ndarray, n: int = 24, seed: int = 0) -> np.ndarray:
    """Random points in the molecule's bbox plus a small padding (Bohr units)."""
    rng = np.random.default_rng(seed)
    pad = 2.0  # Bohr
    lo = coords_bohr.min(axis=0) - pad
    hi = coords_bohr.max(axis=0) + pad
    return rng.uniform(lo, hi, size=(n, 3))


@pytest.fixture(scope="module")
def fchk_module():
    return pytest.importorskip("moltui.fchk")


@pytest.mark.parametrize("stem", PAIRS)
def test_geometry_matches_molden(stem: str, fchk_module) -> None:
    from moltui.molden import load_molden_data

    fchk_data = fchk_module.load_fchk_data(DATA / f"{stem}.fchk")
    molden_data = load_molden_data(DATA / f"{stem}.molden")

    assert len(fchk_data.molecule.atoms) == len(molden_data.molecule.atoms)
    for a, b in zip(fchk_data.molecule.atoms, molden_data.molecule.atoms):
        assert a.element.symbol == b.element.symbol
        assert np.allclose(a.position, b.position, atol=1e-6)


@pytest.mark.parametrize("stem", PAIRS)
def test_mo_energies_match_molden(stem: str, fchk_module) -> None:
    from moltui.molden import load_molden_data

    fchk_data = fchk_module.load_fchk_data(DATA / f"{stem}.fchk")
    molden_data = load_molden_data(DATA / f"{stem}.molden")

    # iodata writes all MOs; both must agree element-wise.
    assert fchk_data.mo_energies.shape == molden_data.mo_energies.shape
    np.testing.assert_allclose(fchk_data.mo_energies, molden_data.mo_energies, atol=1e-6)


@pytest.mark.parametrize("stem", PAIRS)
def test_mo_values_match_molden_on_grid(stem: str, fchk_module) -> None:
    """The strongest test: parsed basis + MOs must produce the same wavefunction."""
    from moltui.molden import load_molden_data

    fchk_data = fchk_module.load_fchk_data(DATA / f"{stem}.fchk")
    molden_data = load_molden_data(DATA / f"{stem}.molden")

    coords_bohr = fchk_data._basis.atom_coords_bohr
    points = _sample_points(coords_bohr)

    fchk_vals = _eval_basis_on_points(fchk_data._basis, points)
    molden_vals = _eval_basis_on_points(molden_data._basis, points)

    assert fchk_vals.shape == molden_vals.shape
    # MOs from iodata round-trip should agree with the fchk source up to float
    # precision — no per-MO sign flip, no permutation, since iodata copies the
    # coefficients without re-diagonalizing. Use a generous atol because GTO
    # values have wide dynamic range near nuclei.
    np.testing.assert_allclose(fchk_vals, molden_vals, atol=1e-5, rtol=1e-4)


def test_normal_modes_match_molden(fchk_module) -> None:
    """Vib-Modes / Vib-E2 must agree with the cclib-converted molden.

    iodata's molden writer drops ``[FREQ]``/``[FR-NORM-COORD]``; the paired
    ``peroxide_tsopt.molden`` is therefore produced via cclib instead. Mode
    eigenvectors carry an arbitrary sign per mode, so we compare absolute
    values element-wise.
    """
    from moltui.molden import load_molden_data

    fchk_data = fchk_module.load_fchk_data(DATA / "peroxide_tsopt.fchk")
    molden_data = load_molden_data(DATA / "peroxide_tsopt.molden")

    assert fchk_data.normal_modes is not None
    assert molden_data.normal_modes is not None
    assert fchk_data.mode_frequencies is not None
    assert molden_data.mode_frequencies is not None

    np.testing.assert_allclose(fchk_data.mode_frequencies, molden_data.mode_frequencies, atol=1e-3)

    assert fchk_data.normal_modes.shape == molden_data.normal_modes.shape
    np.testing.assert_allclose(
        np.abs(fchk_data.normal_modes), np.abs(molden_data.normal_modes), atol=1e-6
    )
