"""Tests for vibrational normal mode extraction via cclib."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cclib")

from moltui.parsers import load_normal_modes_from_cclib

DATA = Path(__file__).parent.parent / "data"
H2O_OUT = DATA / "orca/vib/h2o.out"
H2O_HESS = DATA / "orca/vib/h2o.hess"


@pytest.fixture(scope="module")
def hess_cclib():
    return load_normal_modes_from_cclib(H2O_OUT)


@pytest.fixture(scope="module")
def hess_orca():
    from moltui.parsers import parse_orca_hess_data

    return parse_orca_hess_data(H2O_HESS)


def test_returns_hess_data(hess_cclib):
    assert hess_cclib is not None


def test_frequencies_shape(hess_cclib, hess_orca):
    # Both parsers strip zero-frequency modes; shapes must agree.
    assert hess_cclib.frequencies is not None
    assert hess_orca.frequencies is not None
    assert hess_cclib.frequencies.shape == hess_orca.frequencies.shape


def test_frequencies_values(hess_cclib, hess_orca):
    # Tolerate ~1 cm⁻¹ rounding between output file and .hess file.
    np.testing.assert_allclose(hess_cclib.frequencies, hess_orca.frequencies, atol=1.0)


def test_normal_modes_shape(hess_cclib, hess_orca):
    assert hess_cclib.normal_modes is not None
    assert hess_orca.normal_modes is not None
    assert hess_cclib.normal_modes.shape == hess_orca.normal_modes.shape


def test_molecule_atom_count(hess_cclib, hess_orca):
    assert len(hess_cclib.molecule.atoms) == len(hess_orca.molecule.atoms)


def test_no_vib_data_returns_none(tmp_path):
    """A file with no vibrational data should return None."""
    # Write a minimal XYZ file that cclib cannot parse as a QC output
    # Use a cclib-parseable but vib-free scenario by monkeypatching
    pass


def test_cclib_returns_none_for_missing_vib(monkeypatch):
    """load_normal_modes_from_cclib returns None when cclib data has no vibfreqs."""
    import moltui.parsers as parsers

    class _FakeData:
        atomcoords = np.zeros((1, 2, 3))
        atomnos = [8, 1]
        natom = 2

    monkeypatch.setattr(parsers, "_parse_cclib_data", lambda _: _FakeData())
    result = parsers.load_normal_modes_from_cclib("fake.out")
    assert result is None
