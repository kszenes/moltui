#!/usr/bin/env python3
"""Tests for parsers.py: XYZ, cube, and load_molecule dispatch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from moltui.parsers import load_molecule, parse_cube_data, parse_xyz

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


# --- XYZ parsing ---


class TestParseXYZ:
    def test_water(self):
        mol = parse_xyz(EXAMPLES_DIR / "water.xyz")
        assert len(mol.atoms) == 3
        symbols = [a.element.symbol for a in mol.atoms]
        assert "O" in symbols
        assert symbols.count("H") == 2

    def test_aspirin_atom_count(self):
        mol = parse_xyz(EXAMPLES_DIR / "aspirin.xyz")
        assert len(mol.atoms) == 21

    def test_bonds_detected(self):
        mol = parse_xyz(EXAMPLES_DIR / "water.xyz")
        assert len(mol.bonds) > 0

    def test_ethanol(self):
        mol = parse_xyz(EXAMPLES_DIR / "ethanol.xyz")
        symbols = [a.element.symbol for a in mol.atoms]
        assert "C" in symbols
        assert "O" in symbols

    def test_positions_are_float_arrays(self):
        mol = parse_xyz(EXAMPLES_DIR / "water.xyz")
        for atom in mol.atoms:
            assert atom.position.shape == (3,)
            assert atom.position.dtype == np.float64


# --- Cube parsing ---


class TestParseCubeData:
    @pytest.fixture
    def cube_data(self):
        cube_files = list(EXAMPLES_DIR.glob("*.cube"))
        if not cube_files:
            pytest.skip("No .cube files in examples/")
        return parse_cube_data(cube_files[0])

    def test_molecule_has_atoms(self, cube_data):
        assert len(cube_data.molecule.atoms) > 0

    def test_volumetric_data_shape(self, cube_data):
        assert cube_data.data.ndim == 3
        assert cube_data.data.shape == cube_data.n_points

    def test_origin_and_axes(self, cube_data):
        assert cube_data.origin.shape == (3,)
        assert cube_data.axes.shape == (3, 3)

    def test_bonds_detected(self, cube_data):
        assert len(cube_data.molecule.bonds) > 0


# --- load_molecule dispatch ---


class TestLoadMolecule:
    def test_xyz_dispatch(self):
        mol = load_molecule(EXAMPLES_DIR / "water.xyz")
        assert len(mol.atoms) == 3

    def test_cube_dispatch(self):
        cube_files = list(EXAMPLES_DIR.glob("*.cube"))
        if not cube_files:
            pytest.skip("No .cube files in examples/")
        mol = load_molecule(cube_files[0])
        assert len(mol.atoms) > 0

    def test_unsupported_format_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            load_molecule(Path("/fake/file.pdb"))

    def test_gbw_raises(self):
        with pytest.raises(ValueError, match=".gbw"):
            load_molecule(Path("/fake/file.gbw"))


# --- Smoke test: load all example XYZ files ---


XYZ_FILES = sorted(EXAMPLES_DIR.glob("*.xyz"))


@pytest.mark.parametrize("xyz_file", XYZ_FILES, ids=lambda p: p.name)
def test_load_xyz_smoke(xyz_file: Path):
    mol = parse_xyz(xyz_file)
    assert len(mol.atoms) > 0
    assert all(a.position.shape == (3,) for a in mol.atoms)
