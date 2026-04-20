#!/usr/bin/env python3
"""Tests for parsers.py: XYZ, cube, and load_molecule dispatch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from moltui.parsers import load_molecule, parse_cube_data, parse_xyz, parse_xyz_trajectory


def _write_xyz(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(body)
    return path


def _write_cube(tmp_path: Path, name: str = "sample.cube") -> Path:
    path = tmp_path / name
    path.write_text(
        "\n".join(
            [
                "Cube file",
                "Generated for tests",
                "2 0.0 0.0 0.0",
                "2 1.0 0.0 0.0",
                "2 0.0 1.0 0.0",
                "2 0.0 0.0 1.0",
                "8 0.0 0.0 0.0 0.0",
                "1 0.0 0.0 0.0 1.8",
                "0.1 0.2 0.3 0.4 0.5 0.6",
                "0.7 0.8",
                "",
            ]
        )
    )
    return path
EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
REPO_ROOT = Path(__file__).resolve().parent.parent


WATER_XYZ = """3
water molecule
O 0.00000 0.00000 0.00000
H 0.75700 0.58600 0.00000
H -0.75700 0.58600 0.00000
"""

TWO_FRAME_XYZ = """3
frame 1
O 0.00000 0.00000 0.00000
H 0.75700 0.58600 0.00000
H -0.75700 0.58600 0.00000
3
frame 2
O 0.10000 0.00000 0.00000
H 0.85700 0.58600 0.00000
H -0.65700 0.58600 0.00000
"""


# --- XYZ parsing ---


class TestParseXYZ:
    def test_water(self, tmp_path: Path):
        xyz = _write_xyz(
            tmp_path,
            "water.xyz",
            "3\nwater\nO 0.0000 0.0000 0.0000\nH 0.7586 0.0000 0.5043\nH -0.7586 0.0000 0.5043\n",
        )
        mol = parse_xyz(xyz)
        assert len(mol.atoms) == 3
        symbols = [a.element.symbol for a in mol.atoms]
        assert "O" in symbols
        assert symbols.count("H") == 2

    def test_aspirin_atom_count(self, tmp_path: Path):
        atom_lines = "\n".join(f"H {i * 0.7:.3f} 0.0 0.0" for i in range(21))
        xyz = _write_xyz(tmp_path, "aspirin_like.xyz", f"21\nmock\n{atom_lines}\n")
        mol = parse_xyz(xyz)
        assert len(mol.atoms) == 21

    def test_bonds_detected(self, tmp_path: Path):
        xyz = _write_xyz(
            tmp_path,
            "water.xyz",
            "3\nwater\nO 0.0000 0.0000 0.0000\nH 0.7586 0.0000 0.5043\nH -0.7586 0.0000 0.5043\n",
        )
        mol = parse_xyz(xyz)
        assert len(mol.bonds) > 0

    def test_ethanol(self, tmp_path: Path):
        xyz = _write_xyz(
            tmp_path,
            "ethanol.xyz",
            (
                "9\nethanol\n"
                "C 0.000 0.000 0.000\n"
                "C 1.520 0.000 0.000\n"
                "O 2.020 1.200 0.000\n"
                "H -0.540 0.900 0.000\n"
                "H -0.540 -0.900 0.000\n"
                "H 1.980 -0.900 0.000\n"
                "H 1.980 0.500 0.900\n"
                "H 1.980 0.500 -0.900\n"
                "H 2.980 1.100 0.000\n"
            ),
        )
        mol = parse_xyz(xyz)
        symbols = [a.element.symbol for a in mol.atoms]
        assert "C" in symbols
        assert "O" in symbols

    def test_positions_are_float_arrays(self, tmp_path: Path):
        xyz = _write_xyz(
            tmp_path,
            "water.xyz",
            "3\nwater\nO 0.0000 0.0000 0.0000\nH 0.7586 0.0000 0.5043\nH -0.7586 0.0000 0.5043\n",
        )
        mol = parse_xyz(xyz)
        for atom in mol.atoms:
            assert atom.position.shape == (3,)
            assert atom.position.dtype == np.float64


# --- Cube parsing ---


class TestParseCubeData:
    @pytest.fixture
    def cube_data(self, tmp_path: Path):
        cube_file = _write_cube(tmp_path)
        return parse_cube_data(cube_file)

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
    def test_xyz_dispatch(self, tmp_path: Path):
        xyz = _write_xyz(
            tmp_path,
            "water.xyz",
            "3\nwater\nO 0.0000 0.0000 0.0000\nH 0.7586 0.0000 0.5043\nH -0.7586 0.0000 0.5043\n",
        )
        mol = load_molecule(xyz)
        assert len(mol.atoms) == 3

    def test_cube_dispatch(self, tmp_path: Path):
        cube_file = _write_cube(tmp_path)
        mol = load_molecule(cube_file)
        assert len(mol.atoms) > 0

    def test_unsupported_format_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            load_molecule(Path("/fake/file.pdb"))

    def test_gbw_raises(self):
        with pytest.raises(ValueError, match=".gbw"):
            load_molecule(Path("/fake/file.gbw"))


# --- Smoke test: load all example XYZ files ---


def test_load_xyz_smoke(tmp_path: Path):
    xyz_files = [
        _write_xyz(
            tmp_path,
            "water.xyz",
            "3\nwater\nO 0.0 0.0 0.0\nH 0.7586 0.0 0.5043\nH -0.7586 0.0 0.5043\n",
        ),
        _write_xyz(tmp_path, "co2.xyz", "3\nco2\nO -1.16 0.0 0.0\nC 0.0 0.0 0.0\nO 1.16 0.0 0.0\n"),
    ]
    for xyz_file in xyz_files:
        mol = parse_xyz(xyz_file)
        assert len(mol.atoms) > 0
        assert all(a.position.shape == (3,) for a in mol.atoms)
XYZ_FILES = sorted(EXAMPLES_DIR.glob("*.xyz"))


@pytest.mark.parametrize("xyz_file", XYZ_FILES, ids=lambda p: p.name)
def test_load_xyz_smoke(xyz_file: Path):
    mol = parse_xyz(xyz_file)
    assert len(mol.atoms) > 0
    assert all(a.position.shape == (3,) for a in mol.atoms)


# --- Multi-frame XYZ (trajectory) ---


class TestParseXYZTrajectory:
    def test_single_frame_returns_list_of_one(self, tmp_path: Path):
        f = tmp_path / "water.xyz"
        f.write_text(WATER_XYZ)
        frames = parse_xyz_trajectory(f)
        assert len(frames) == 1
        assert len(frames[0].atoms) == 3

    def test_two_frames(self, tmp_path: Path):
        f = tmp_path / "traj.xyz"
        f.write_text(TWO_FRAME_XYZ)
        frames = parse_xyz_trajectory(f)
        assert len(frames) == 2
        assert len(frames[0].atoms) == 3
        assert len(frames[1].atoms) == 3
        # Atoms moved between frames
        assert not np.allclose(frames[0].atoms[0].position, frames[1].atoms[0].position)

    def test_parse_xyz_returns_first_frame(self, tmp_path: Path):
        f = tmp_path / "traj.xyz"
        f.write_text(TWO_FRAME_XYZ)
        mol = parse_xyz(f)
        frames = parse_xyz_trajectory(f)
        assert len(mol.atoms) == len(frames[0].atoms)
        for a, b in zip(mol.atoms, frames[0].atoms):
            assert np.allclose(a.position, b.position)

    def test_trailing_blank_lines_tolerated(self, tmp_path: Path):
        f = tmp_path / "traj.xyz"
        f.write_text(TWO_FRAME_XYZ + "\n\n")
        frames = parse_xyz_trajectory(f)
        assert len(frames) == 2

    def test_bonds_detected_per_frame(self, tmp_path: Path):
        f = tmp_path / "traj.xyz"
        f.write_text(TWO_FRAME_XYZ)
        frames = parse_xyz_trajectory(f)
        for mol in frames:
            assert len(mol.bonds) > 0

    def test_vfile_09_if_present(self):
        vfile = REPO_ROOT / "vfile_09.xyz"
        if not vfile.exists():
            pytest.skip("vfile_09.xyz not in repo root")
        frames = parse_xyz_trajectory(vfile)
        assert len(frames) == 19
        for mol in frames:
            assert len(mol.atoms) == 16
