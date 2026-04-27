#!/usr/bin/env python3
"""Tests for _detect_filetype in app.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from moltui.app import _detect_filetype


class TestDetectFiletype:
    def test_xyz_detected(self, tmp_path: Path):
        path = tmp_path / "water.xyz"
        path.write_text("3\nwater\nO 0 0 0\nH 0.758 0 0.504\nH -0.758 0 0.504\n")
        assert _detect_filetype(str(path)) == "xyz"

    def test_molden_detected(self, tmp_path: Path):
        path = tmp_path / "sample.molden"
        path.write_text("[MOLDEN FORMAT]\n[Atoms] AU\nH 1 1 0.0 0.0 0.0\n")
        assert _detect_filetype(str(path)) == "molden"

    def test_cube_detected(self, tmp_path: Path):
        path = tmp_path / "sample.cube"
        path.write_text("comment 1\ncomment 2\n2 0.0 0.0 0.0\n")
        assert _detect_filetype(str(path)) == "cube"

    def test_com_gjf_detected(self, tmp_path: Path) -> None:
        com = tmp_path / "x.com"
        com.write_text("%mem=60GB\n#p b3lyp/6-31g\n\ntest\n\n0 1\nH 0 0 0\n\n")
        assert _detect_filetype(str(com)) == "com"

    def test_gbw_by_extension(self):
        # .gbw detection is by extension, doesn't read content
        path = Path("fake.gbw")
        assert _detect_filetype(str(path)) == "gbw"

    def test_xyz_content_detection(self, tmp_path: Path):
        """XYZ is detected by first non-empty line being an integer."""
        path = tmp_path / "molecule.txt"
        path.write_text("3\ncomment\nH 0 0 0\nH 1 0 0\nH 0 1 0\n")
        assert _detect_filetype(str(path)) == "xyz"

    def test_molden_content_detection(self, tmp_path: Path):
        """Molden is detected by [Molden Format] marker."""
        path = tmp_path / "molecule.txt"
        path.write_text("[Molden Format]\n[Atoms] AU\n")
        assert _detect_filetype(str(path)) == "molden"

    def test_h5_detected_as_trexio_without_utf8_sniff(self, tmp_path: Path):
        path = tmp_path / "empty.h5"
        path.write_bytes(b"\x89HDF\r\n\x1a\n")  # HDF5 signature prefix
        assert _detect_filetype(str(path)) == "trexio"

    def test_trexio_file_extension_detected(self, tmp_path: Path):
        path = tmp_path / "state.trexio"
        path.write_bytes(b"")
        assert _detect_filetype(str(path)) == "trexio"

    def test_trexio_directory_detected(self, tmp_path: Path):
        d = tmp_path / "case.trexio"
        d.mkdir()
        assert _detect_filetype(str(d)) == "trexio"

    def test_plain_directory_not_trexio_raises(self, tmp_path: Path):
        d = tmp_path / "emptydir"
        d.mkdir()
        with pytest.raises(ValueError, match="is a directory"):
            _detect_filetype(str(d))

    def test_regression_directory_mf_sniff_does_not_call_open(
        self,
        tmp_path: Path,
    ) -> None:
        """Regression: `moltui mf` used to crash with IsADirectoryError.

        _detect_filetype must handle directories before content sniffing; an empty
        folder named ``mf`` is not TREXIO text and must raise ValueError, not
        ``OSError``/``IsADirectoryError`` from ``open()``.
        """
        mf = tmp_path / "mf"
        mf.mkdir()
        # If open() runs on this directory, Python raises IsADirectoryError instead.
        with pytest.raises(ValueError, match="is a directory"):
            _detect_filetype(str(mf))

    def test_regression_subprocess_mf_directory_value_error_not_is_a_directory_error(
        self,
        tmp_path: Path,
    ) -> None:
        """Fresh process: must get ValueError, not IsADirectoryError from open(dir)."""
        mf = tmp_path / "mf"
        mf.mkdir()
        script = r"""import sys
from pathlib import Path
from moltui.app import _detect_filetype
try:
    _detect_filetype(str(Path(sys.argv[1])))
except ValueError:
    sys.exit(0)
except IsADirectoryError:
    sys.exit(1)
"""
        r = subprocess.run(
            [sys.executable, "-c", script, str(mf)],
            capture_output=True,
        )
        assert r.returncode == 0, (
            "expected ValueError path; IsADirectoryError means open() was used on a directory"
        )

    def test_trexio_text_directory_without_trexio_suffix(self, tmp_path: Path):
        trexio = pytest.importorskip("trexio")

        d = tmp_path / "mf"
        d.mkdir()
        with trexio.File(str(d), "w", back_end=trexio.TREXIO_TEXT) as f:
            trexio.write_nucleus_num(f, 1)
            trexio.write_nucleus_coord(f, [[0.0, 0.0, 0.0]])
            trexio.write_nucleus_charge(f, [1.0])

        assert _detect_filetype(str(d)) == "trexio"
