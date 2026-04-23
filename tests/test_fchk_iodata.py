"""Cross-check Gaussian fchk parsing against IODATA's reference parser."""

from __future__ import annotations

import importlib.resources
from pathlib import Path

import numpy as np
import pytest

from moltui.fchk import (
    _fchk_to_moltui_ao_permutation,
    _read_sections,
    load_fchk_data,
    parse_fchk,
    parse_fchk_atoms,
    parse_fchk_trajectory,
)
from moltui.gto import BOHR_TO_ANGSTROM

iodata = pytest.importorskip("iodata")
DATA = Path(str(importlib.resources.files("iodata") / "test" / "data"))
FCHK_FILES = sorted(DATA.glob("*.fchk"))


def _molecule_atomic_numbers(molecule) -> np.ndarray:
    return np.array([atom.element.atomic_number for atom in molecule.atoms], dtype=int)


def _molecule_coords(molecule) -> np.ndarray:
    return np.array([atom.position for atom in molecule.atoms], dtype=np.float64)


@pytest.mark.parametrize("path", FCHK_FILES, ids=lambda p: p.name)
def test_fchk_current_geometry_matches_iodata(path: Path) -> None:
    try:
        ref = iodata.load_one(str(path))
    except Exception:
        # IODATA load_one expects basis data and fails on trajectory-only fchks;
        # those are covered by the trajectory test below.
        pytest.skip("IODATA load_one cannot read this fchk variant")

    molecule = parse_fchk_atoms(path)

    np.testing.assert_array_equal(_molecule_atomic_numbers(molecule), ref.atnums.astype(int))
    np.testing.assert_allclose(
        _molecule_coords(molecule), ref.atcoords * BOHR_TO_ANGSTROM, atol=2e-6, rtol=1e-8
    )


@pytest.mark.parametrize("path", FCHK_FILES, ids=lambda p: p.name)
def test_fchk_orbital_data_matches_iodata(path: Path) -> None:
    try:
        ref = iodata.load_one(str(path))
    except Exception:
        pytest.skip("IODATA load_one cannot read this fchk variant")
    if ref.mo is None:
        pytest.skip("No IODATA MO data")

    basis = parse_fchk(path)
    data = load_fchk_data(path)
    ao_perm = _fchk_to_moltui_ao_permutation(_read_sections(path)["Shell types"].array.astype(int))

    assert data.n_mos == ref.mo.norb
    assert basis.mo_coefficients.shape == ref.mo.coeffs.shape
    np.testing.assert_allclose(basis.mo_energies, ref.mo.energies, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(basis.mo_occupations, ref.mo.occs, atol=0.0, rtol=0.0)
    # IODATA stores coefficients in its fchk AO convention. MolTUI reorders rows
    # into its evaluator/Molden convention, so compare after applying the same
    # fchk->MolTUI AO permutation to IODATA's rows.
    np.testing.assert_allclose(basis.mo_coefficients, ref.mo.coeffs[ao_perm, :], atol=0.0, rtol=0.0)


@pytest.mark.parametrize("path", FCHK_FILES, ids=lambda p: p.name)
def test_fchk_trajectories_match_iodata_load_many(path: Path) -> None:
    try:
        ref_frames = list(iodata.load_many(str(path)))
    except Exception:
        pytest.skip("IODATA load_many cannot read a trajectory from this fchk")
    if not ref_frames:
        pytest.skip("No IODATA trajectory frames")

    trajectory = parse_fchk_trajectory(path)
    ref_coords = np.stack([frame.atcoords for frame in ref_frames], axis=0) * BOHR_TO_ANGSTROM

    assert trajectory.frames.shape == ref_coords.shape
    np.testing.assert_allclose(trajectory.frames, ref_coords, atol=2e-6, rtol=1e-8)
    np.testing.assert_array_equal(
        _molecule_atomic_numbers(trajectory.molecule), ref_frames[0].atnums.astype(int)
    )
    np.testing.assert_allclose(
        _molecule_coords(trajectory.molecule), ref_coords[0], atol=2e-6, rtol=1e-8
    )
