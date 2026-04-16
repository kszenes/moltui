#!/usr/bin/env python3
"""Tests for _detect_filetype in app.py."""

from __future__ import annotations

import tempfile
from pathlib import Path

from moltui.app import _detect_filetype

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


class TestDetectFiletype:
    def test_xyz_detected(self):
        assert _detect_filetype(str(EXAMPLES_DIR / "water.xyz")) == "xyz"

    def test_molden_detected(self):
        molden_files = list(EXAMPLES_DIR.glob("*.molden"))
        if not molden_files:
            return
        assert _detect_filetype(str(molden_files[0])) == "molden"

    def test_cube_detected(self):
        cube_files = list(EXAMPLES_DIR.glob("*.cube"))
        if not cube_files:
            return
        assert _detect_filetype(str(cube_files[0])) == "cube"

    def test_gbw_by_extension(self):
        # .gbw detection is by extension, doesn't read content
        with tempfile.NamedTemporaryFile(suffix=".gbw") as f:
            assert _detect_filetype(f.name) == "gbw"

    def test_xyz_content_detection(self):
        """XYZ is detected by first non-empty line being an integer."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("3\ncomment\nH 0 0 0\nH 1 0 0\nH 0 1 0\n")
            f.flush()
            assert _detect_filetype(f.name) == "xyz"
            Path(f.name).unlink()

    def test_molden_content_detection(self):
        """Molden is detected by [Molden Format] marker."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("[Molden Format]\n[Atoms] AU\n")
            f.flush()
            assert _detect_filetype(f.name) == "molden"
            Path(f.name).unlink()
