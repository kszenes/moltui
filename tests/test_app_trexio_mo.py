"""CLI / app wiring for TREXIO files with molecular orbitals."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("trexio")

import numpy as np

from moltui.app import _detect_filetype, _prepare_trexio_cli_session
from moltui.trexio_molden import load_trexio_orbital_data

TREXIO_DATA = Path(__file__).resolve().parent.parent / "data" / "trexio"


def _require(path: Path) -> Path:
    if not path.exists():
        pytest.skip(f"data file not found: {path}")
    return path


def _mf_h5() -> Path:
    return _require(TREXIO_DATA / "n2_sp" / "n2.h5")


def test_detect_filetype_h5_is_trexio() -> None:
    assert _detect_filetype(str(_mf_h5())) == "trexio"


def test_prepare_trexio_cli_session_attaches_orbital_data_and_isosurfaces() -> None:
    """Regression: opening mf.h5 must populate MO data so the orbital UI can mount."""
    mol, orbital_data, isosurfaces, current_mo, _toast = _prepare_trexio_cli_session(_mf_h5())
    assert orbital_data is not None
    assert orbital_data.n_mos > 0
    assert current_mo == orbital_data.homo_idx
    assert len(isosurfaces) > 0
    assert len(mol.atoms) == len(orbital_data.molecule.atoms)


def test_missing_mo_energies_flags_and_toast() -> None:
    """TREXIO file without mo_energy sets has_mo_energies=False and produces a toast."""
    path = _require(TREXIO_DATA / "missing_moenergies" / "C2H2.hdf5")

    orbital_data = load_trexio_orbital_data(path)
    assert orbital_data is not None
    assert orbital_data.has_mo_energies is False
    assert orbital_data.has_mo_occupations is True
    # Energies should be zero-filled
    assert np.all(orbital_data.mo_energies == 0.0)
    # Occupations should be real values
    assert orbital_data.mo_occupations.sum() > 0

    # CLI session should produce a warning toast
    _, _, _, _, toast = _prepare_trexio_cli_session(path)
    assert toast is not None
    assert "energies" in toast
