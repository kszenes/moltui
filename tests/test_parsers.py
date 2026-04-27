#!/usr/bin/env python3
"""Tests for parsers.py: XYZ, cube, and load_molecule dispatch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from moltui.parsers import (
    load_molecule,
    parse_com_file,
    parse_cube_data,
    parse_orca_hess_data,
    parse_xyz,
    parse_xyz_trajectory,
)


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


def _write_orca_hess(tmp_path: Path, name: str = "sample.hess") -> Path:
    path = tmp_path / name
    normal_mode_rows = []
    for i in range(9):
        row = ["1.0" if i == j else "0.0" for j in range(9)]
        normal_mode_rows.append(f"{i} " + " ".join(row))
    path.write_text(
        "\n".join(
            [
                "$atoms",
                "3",
                "O 15.999 0.000000 0.000000 0.000000",
                "H 1.008 0.000000 1.430000 1.100000",
                "H 1.008 0.000000 -1.430000 1.100000",
                "$vibrational_frequencies",
                "9",
                "0 0.0",
                "1 0.0",
                "2 0.0",
                "3 0.0",
                "4 0.0",
                "5 0.0",
                "6 2041.3",
                "7 4493.7",
                "8 4796.3",
                "$normal_modes",
                "9 9",
                "0 1 2 3 4 5 6 7 8",
                *normal_mode_rows,
                "",
            ]
        )
    )
    return path

def _write_com(tmp_path: Path, name: str = "x.com") -> Path:
    path = tmp_path / name
    path.write_text(
        "%mem=60GB\n#p b3lyp/6-31g\n\ntest\n\n0 1\nH 0 0 0\n\n"
    )
    return path


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


class TestParseXYZTrajectory:
    def test_parses_multiple_frames(self, tmp_path: Path):
        xyz = _write_xyz(
            tmp_path,
            "traj.xyz",
            (
                "3\nframe1\n"
                "O 0.0000 0.0000 0.0000\n"
                "H 0.7586 0.0000 0.5043\n"
                "H -0.7586 0.0000 0.5043\n"
                "3\nframe2\n"
                "O 0.0100 0.0000 0.0000\n"
                "H 0.7686 0.0000 0.5043\n"
                "H -0.7486 0.0000 0.5043\n"
            ),
        )
        traj = parse_xyz_trajectory(xyz)
        assert traj.frames.shape == (2, 3, 3)
        assert len(traj.molecule.atoms) == 3
        assert np.isclose(traj.frames[1, 0, 0], 0.01)

    def test_rejects_inconsistent_symbols(self, tmp_path: Path):
        xyz = _write_xyz(
            tmp_path,
            "bad_traj.xyz",
            ("2\nframe1\nH 0.0 0.0 0.0\nO 0.0 0.0 1.0\n2\nframe2\nO 0.0 0.0 0.0\nH 0.0 0.0 1.0\n"),
        )
        with pytest.raises(ValueError, match="preserve atom ordering and symbols"):
            parse_xyz_trajectory(xyz)


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

    def test_hess_dispatch(self, tmp_path: Path):
        hess_file = _write_orca_hess(tmp_path)
        mol = load_molecule(hess_file)
        assert len(mol.atoms) == 3

    def test_unsupported_format_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            load_molecule(Path("/fake/file.pdb"))

    def test_gbw_raises(self):
        with pytest.raises(ValueError, match=".gbw"):
            load_molecule(Path("/fake/file.gbw"))

    def test_com_dispatch(self, tmp_path: Path) -> None:
        com = _write_com(tmp_path)
        mol = load_molecule(com)
        assert len(mol.atoms) == 1


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


def test_parse_orca_hess_data_includes_normal_modes(tmp_path: Path) -> None:
    hess_file = _write_orca_hess(tmp_path)
    hess_data = parse_orca_hess_data(hess_file)

    assert len(hess_data.molecule.atoms) == 3
    assert hess_data.frequencies is not None
    assert hess_data.normal_modes is not None
    assert hess_data.frequencies.shape == (9,)
    assert hess_data.normal_modes.shape == (9, 3, 3)
