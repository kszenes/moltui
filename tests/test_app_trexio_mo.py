"""CLI / app wiring for TREXIO files with molecular orbitals."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("trexio")

from moltui.app import _detect_filetype, _prepare_trexio_cli_session


def _mf_h5() -> Path:
    p = Path(__file__).resolve().parent / "trexio" / "mf.h5"
    if not p.is_file():
        pytest.skip("tests/trexio/mf.h5 fixture not present")
    return p


def test_detect_filetype_h5_is_trexio() -> None:
    assert _detect_filetype(str(_mf_h5())) == "trexio"


def test_prepare_trexio_cli_session_attaches_orbital_data_and_isosurfaces() -> None:
    """Regression: opening mf.h5 must populate MO data so the orbital UI can mount."""
    mol, orbital_data, isosurfaces, current_mo = _prepare_trexio_cli_session(_mf_h5())
    assert orbital_data is not None
    assert orbital_data.n_mos > 0
    assert current_mo == orbital_data.homo_idx
    assert len(isosurfaces) > 0
    assert len(mol.atoms) == len(orbital_data.molecule.atoms)
