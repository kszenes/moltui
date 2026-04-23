"""Tests for TREXIO loading (optional ``trexio`` dependency)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("trexio")

from moltui.parsers import BOHR_TO_ANGSTROM
from moltui.trexio_support import load_molecule_from_trexio


def _write_minimal_h2(f) -> None:
    import trexio

    coord = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]
    labels = ["H", "H"]
    charges = [1.0, 1.0]
    trexio.write_nucleus_num(f, 2)
    trexio.write_nucleus_coord(f, coord)
    trexio.write_nucleus_label(f, labels)
    trexio.write_nucleus_charge(f, charges)


def test_load_molecule_from_trexio_hdf5(tmp_path: Path):
    import trexio

    path = tmp_path / "mol.h5"
    with trexio.File(str(path), "w", back_end=trexio.TREXIO_HDF5) as f:
        _write_minimal_h2(f)

    mol = load_molecule_from_trexio(path)
    assert len(mol.atoms) == 2
    assert all(np.isfinite(a.position).all() for a in mol.atoms)
    expected_z = 1.4 * BOHR_TO_ANGSTROM
    assert abs(mol.atoms[1].position[2] - expected_z) < 1e-9


def test_load_molecule_from_trexio_text_dir(tmp_path: Path):
    import trexio

    path = tmp_path / "run.trexio"
    path.mkdir()
    with trexio.File(str(path), "w", back_end=trexio.TREXIO_TEXT) as f:
        _write_minimal_h2(f)

    mol = load_molecule_from_trexio(path)
    assert len(mol.atoms) == 2
    expected_z = 1.4 * BOHR_TO_ANGSTROM
    assert abs(mol.atoms[1].position[2] - expected_z) < 1e-9


def test_non_trexio_h5_error_message(tmp_path: Path):
    path = tmp_path / "other.h5"
    path.write_bytes(b"\x89HDF\r\n\x1a\n\x00" + b"\x00" * 64)
    with pytest.raises(ValueError, match="not a valid TREXIO file"):
        load_molecule_from_trexio(path)


def test_non_trexio_h5_suppresses_hdf5_library_stderr(tmp_path: Path):
    """HDF5 C library should not print HDF5-DIAG to stderr; only MolTUI's error is shown."""
    path = tmp_path / "nondiag.h5"
    path.write_bytes(b"\x89HDF\r\n\x1a\n\x00" + b"\x00" * 64)
    script = r"""import sys
from pathlib import Path
from moltui.trexio_support import load_molecule_from_trexio
try:
    load_molecule_from_trexio(Path(sys.argv[1]))
except ValueError:
    sys.exit(1)
"""
    r = subprocess.run(
        [sys.executable, "-c", script, str(path)],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 1
    assert "HDF5-DIAG" not in r.stderr


def test_load_molecule_from_trexio_charge_only(tmp_path: Path):
    import trexio

    path = tmp_path / "ions.h5"
    coord = [[0.0, 0.0, 0.0]]
    with trexio.File(str(path), "w", back_end=trexio.TREXIO_HDF5) as f:
        trexio.write_nucleus_num(f, 1)
        trexio.write_nucleus_coord(f, coord)
        trexio.write_nucleus_charge(f, [6.0])

    mol = load_molecule_from_trexio(path)
    assert len(mol.atoms) == 1
    assert mol.atoms[0].element.symbol == "C"
