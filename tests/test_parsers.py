#!/usr/bin/env python3
"""Tests for parsers.py: XYZ, cube, and load_molecule dispatch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from moltui.parsers import (
    _parse_xyz_comment_metadata,
    load_molecule,
    parse_cif,
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

    def test_parse_xyz_uses_first_frame_when_frame_sizes_differ(self, tmp_path: Path):
        xyz = _write_xyz(
            tmp_path,
            "dataset.xyz",
            "3\nwater\nO 0 0 0\nH 0 0 1\nH 0 1 0\n1\nnitrogen\nN 2 2 2\n",
        )

        mol = parse_xyz(xyz)

        assert [atom.element.symbol for atom in mol.atoms] == ["O", "H", "H"]


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


# --- Extended XYZ ---


class TestExtXYZ:
    def test_comment_metadata_handles_quoted_and_bare_values(self):
        metadata = _parse_xyz_comment_metadata(
            'comment Lattice="4.0 0.0 0.0 0.0 5.0 0.0 0.0 0.0 6.0" energy=-15.5 note'
        )
        assert metadata == {
            "Lattice": "4.0 0.0 0.0 0.0 5.0 0.0 0.0 0.0 6.0",
            "energy": "-15.5",
        }

    def test_lattice_attached(self, tmp_path: Path):
        body = (
            "3\n"
            'Lattice="4.0 0.0 0.0 0.0 5.0 0.0 0.0 0.0 6.0" '
            'Properties="species:S:1:pos:R:3"\n'
            "O 0.0 0.0 0.0\n"
            "H 0.7586 0.0 0.5043\n"
            "H -0.7586 0.0 0.5043\n"
        )
        xyz = _write_xyz(tmp_path, "water_cell.extxyz", body)
        traj = parse_xyz_trajectory(xyz)
        assert traj.lattice is not None
        assert traj.lattice.shape == (3, 3)
        np.testing.assert_array_equal(np.diag(traj.lattice), [4.0, 5.0, 6.0])
        assert traj.molecule.lattice is not None
        np.testing.assert_array_equal(traj.molecule.lattice, traj.lattice)

    def test_lattice_only_no_properties(self, tmp_path: Path):
        body = '1\nLattice="3.0 0.0 0.0 0.0 3.0 0.0 0.0 0.0 3.0"\nH 0.0 0.0 0.0\n'
        xyz = _write_xyz(tmp_path, "h_only.extxyz", body)
        mol = parse_xyz(xyz)
        assert mol.lattice is not None
        assert mol.lattice[0, 0] == 3.0

    def test_properties_only_no_lattice(self, tmp_path: Path):
        body = '1\nProperties="species:S:1:pos:R:3"\nH 0.0 0.0 0.0\n'
        xyz = _write_xyz(tmp_path, "no_cell.xyz", body)
        mol = parse_xyz(xyz)
        assert mol.lattice is None

    def test_plain_xyz_no_lattice(self, tmp_path: Path):
        xyz = _write_xyz(
            tmp_path,
            "water.xyz",
            "3\nwater\nO 0.0 0.0 0.0\nH 0.7586 0.0 0.5043\nH -0.7586 0.0 0.5043\n",
        )
        mol = parse_xyz(xyz)
        assert mol.lattice is None

    def test_extra_columns_ignored(self, tmp_path: Path):
        body = '1\nProperties="species:S:1:pos:R:3:forces:R:3"\nC 0.1 0.2 0.3 -0.5 0.5 1.5\n'
        xyz = _write_xyz(tmp_path, "with_forces.extxyz", body)
        mol = parse_xyz(xyz)
        assert mol.atoms[0].element.symbol == "C"
        np.testing.assert_allclose(mol.atoms[0].position, [0.1, 0.2, 0.3])

    def test_reordered_properties(self, tmp_path: Path):
        body = '1\nProperties="forces:R:3:species:S:1:pos:R:3"\n-0.1 -0.2 -0.3 N 1.0 2.0 3.0\n'
        xyz = _write_xyz(tmp_path, "reordered.extxyz", body)
        mol = parse_xyz(xyz)
        assert mol.atoms[0].element.symbol == "N"
        np.testing.assert_allclose(mol.atoms[0].position, [1.0, 2.0, 3.0])

    def test_lattice_with_quoted_spaces(self, tmp_path: Path):
        body = (
            "2\n"
            'Lattice="2.0 0.0 0.0   0.0 2.0 0.0   0.0 0.0 2.0" '
            'Properties="species:S:1:pos:R:3" energy=-1.5\n'
            "H 0.0 0.0 0.0\n"
            "H 1.0 0.0 0.0\n"
        )
        xyz = _write_xyz(tmp_path, "spaced_lattice.extxyz", body)
        mol = parse_xyz(xyz)
        assert mol.lattice is not None
        np.testing.assert_array_equal(np.diag(mol.lattice), [2.0, 2.0, 2.0])

    def test_atomic_number_species(self, tmp_path: Path):
        body = (
            "3\n"
            'Lattice="5.0 0.0 0.0 0.0 5.0 0.0 0.0 0.0 5.0" '
            'Properties="species:I:1:pos:R:3"\n'
            "8 0.0 0.0 0.0\n"
            "1 0.7586 0.0 0.5043\n"
            "1 -0.7586 0.0 0.5043\n"
        )
        xyz = _write_xyz(tmp_path, "z_species.extxyz", body)
        mol = parse_xyz(xyz)
        symbols = [a.element.symbol for a in mol.atoms]
        assert symbols == ["O", "H", "H"]

    def test_mixed_symbol_and_z(self, tmp_path: Path):
        body = "2\nplain\n6 0.0 0.0 0.0\nH 1.1 0.0 0.0\n"
        xyz = _write_xyz(tmp_path, "mixed.xyz", body)
        mol = parse_xyz(xyz)
        assert [a.element.symbol for a in mol.atoms] == ["C", "H"]

    def test_multiframe_lattice_from_first(self, tmp_path: Path):
        body = (
            "1\n"
            'Lattice="4.0 0.0 0.0 0.0 4.0 0.0 0.0 0.0 4.0" '
            'Properties="species:S:1:pos:R:3"\n'
            "H 0.0 0.0 0.0\n"
            "1\n"
            "frame2\n"
            "H 0.1 0.0 0.0\n"
        )
        xyz = _write_xyz(tmp_path, "multi.extxyz", body)
        traj = parse_xyz_trajectory(xyz)
        assert traj.frames.shape == (2, 1, 3)
        assert traj.lattice is not None
        assert traj.lattice[0, 0] == 4.0


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


GRAPHITE_CIF = """data_graphite
_cell_length_a       2.46000
_cell_length_b       2.46000
_cell_length_c       6.71000
_cell_angle_alpha   90.00000
_cell_angle_beta    90.00000
_cell_angle_gamma  120.00000
_symmetry_space_group_name_H-M   'P 1'

loop_
_symmetry_equiv_pos_as_xyz
'x, y, z'

loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C1 C 0.00000 0.00000 0.00000
C2 C 0.33333 0.66667 0.00000
C3 C 0.00000 0.00000 0.50000
C4 C 0.66667 0.33333 0.50000
"""


class TestParseCIF:
    def test_graphite_fractional(self, tmp_path: Path):
        path = tmp_path / "graphite.cif"
        path.write_text(GRAPHITE_CIF)
        mol = parse_cif(path)

        assert len(mol.atoms) == 4
        assert all(a.element.symbol == "C" for a in mol.atoms)

        positions = np.array([a.position for a in mol.atoms])
        assert np.allclose(positions[0], [0.0, 0.0, 0.0], atol=1e-3)
        assert np.allclose(positions[1], [0.0, 1.420282, 0.0], atol=1e-3)
        assert np.allclose(positions[2], [0.0, 0.0, 3.355], atol=1e-3)
        assert np.allclose(positions[3], [1.23, 0.710141, 3.355], atol=1e-3)

    def test_graphite_periodic_bonds_end_to_end(self, tmp_path: Path):
        path = tmp_path / "graphite.cif"
        path.write_text(GRAPHITE_CIF)
        mol = parse_cif(path)

        assert mol.bond_shifts is not None
        assert len(mol.bonds) == 6
        assert len(mol.bond_shifts) == 6
        assert (0, 0, 0) in mol.bond_shifts
        lengths = [dist for _, _, dist in mol.get_bond_lengths()]
        assert lengths == pytest.approx([1.4203] * 6, abs=1e-4)

    def test_cartesian_no_cell(self, tmp_path: Path):
        path = tmp_path / "h2.cif"
        path.write_text(
            "data_h2\n"
            "loop_\n"
            "_atom_site_label\n"
            "_atom_site_type_symbol\n"
            "_atom_site_Cartn_x\n"
            "_atom_site_Cartn_y\n"
            "_atom_site_Cartn_z\n"
            "H1 H 0.0 0.0 0.0\n"
            "H2 H 0.74 0.0 0.0\n"
        )
        mol = parse_cif(path)
        assert len(mol.atoms) == 2
        assert np.allclose(mol.atoms[1].position, [0.74, 0.0, 0.0])

    def test_label_with_charge_suffix(self, tmp_path: Path):
        path = tmp_path / "nacl.cif"
        path.write_text(
            "data_nacl\n"
            "_cell_length_a 5.6\n"
            "_cell_length_b 5.6\n"
            "_cell_length_c 5.6\n"
            "_cell_angle_alpha 90\n"
            "_cell_angle_beta 90\n"
            "_cell_angle_gamma 90\n"
            "loop_\n"
            "_atom_site_label\n"
            "_atom_site_type_symbol\n"
            "_atom_site_fract_x\n"
            "_atom_site_fract_y\n"
            "_atom_site_fract_z\n"
            "Na1 Na+ 0.0 0.0 0.0\n"
            "Cl1 Cl- 0.5 0.5 0.5\n"
        )
        mol = parse_cif(path)
        symbols = sorted(a.element.symbol for a in mol.atoms)
        assert symbols == ["Cl", "Na"]

    def test_load_molecule_dispatches_cif(self, tmp_path: Path):
        path = tmp_path / "graphite.cif"
        path.write_text(GRAPHITE_CIF)
        mol = load_molecule(path)
        assert len(mol.atoms) == 4

    def test_semicolon_multiline_text_field(self, tmp_path: Path):
        """Semicolon-delimited text blocks (publication CIF) must not corrupt parsing."""
        path = tmp_path / "multiline.cif"
        path.write_text(
            "data_x\n"
            "_publ_section_title\n"
            ";\n"
            "  A multi-line title that\n"
            "  spans several lines.\n"
            ";\n"
            "_cell_length_a 5.6\n"
            "_cell_length_b 5.6\n"
            "_cell_length_c 5.6\n"
            "_cell_angle_alpha 90\n"
            "_cell_angle_beta 90\n"
            "_cell_angle_gamma 90\n"
            "loop_\n"
            "_atom_site_label\n"
            "_atom_site_type_symbol\n"
            "_atom_site_fract_x\n"
            "_atom_site_fract_y\n"
            "_atom_site_fract_z\n"
            "C1 C 0.0 0.0 0.0\n"
        )
        mol = parse_cif(path)
        assert len(mol.atoms) == 1
        assert mol.atoms[0].element.symbol == "C"

    def test_lowercase_atom_label_two_letter_element(self, tmp_path: Path):
        """A lowercase label like 'ca1' (no _atom_site_type_symbol) must resolve to Ca."""
        path = tmp_path / "ca.cif"
        path.write_text(
            "data_ca\n"
            "_cell_length_a 5.0\n"
            "_cell_length_b 5.0\n"
            "_cell_length_c 5.0\n"
            "_cell_angle_alpha 90\n"
            "_cell_angle_beta 90\n"
            "_cell_angle_gamma 90\n"
            "loop_\n"
            "_atom_site_label\n"
            "_atom_site_fract_x\n"
            "_atom_site_fract_y\n"
            "_atom_site_fract_z\n"
            "ca1 0.0 0.0 0.0\n"
            "c1  0.5 0.5 0.5\n"
        )
        mol = parse_cif(path)
        symbols = [a.element.symbol for a in mol.atoms]
        assert symbols == ["Ca", "C"]

    def test_fractional_occupancy_warns(self, tmp_path: Path):
        """Partial CIF occupancies are shown as full atoms but warn the user."""
        import warnings as _warnings

        from moltui.parsers import CIFParseWarning

        path = tmp_path / "partial.cif"
        path.write_text(
            "data_x\n"
            "_cell_length_a 5.0\n"
            "_cell_length_b 5.0\n"
            "_cell_length_c 5.0\n"
            "_cell_angle_alpha 90\n"
            "_cell_angle_beta 90\n"
            "_cell_angle_gamma 90\n"
            "loop_\n"
            "_atom_site_label\n"
            "_atom_site_type_symbol\n"
            "_atom_site_fract_x\n"
            "_atom_site_fract_y\n"
            "_atom_site_fract_z\n"
            "_atom_site_occupancy\n"
            "C1 C 0.0 0.0 0.0 0.5\n"
            "O1 O 0.5 0.5 0.5 1.0\n"
        )
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always", CIFParseWarning)
            mol = parse_cif(path)
        assert [atom.element.symbol for atom in mol.atoms] == ["C", "O"]
        cif_warns = [w for w in caught if issubclass(w.category, CIFParseWarning)]
        assert len(cif_warns) == 1
        assert "fractional occupancies" in str(cif_warns[0].message)

    def test_hm_name_without_symops_warns(self, tmp_path: Path):
        """Non-P1 H-M space group with no symop loop must emit CIFParseWarning."""
        import warnings as _warnings

        from moltui.parsers import CIFParseWarning

        path = tmp_path / "hm.cif"
        path.write_text(
            "data_x\n"
            "_cell_length_a 5.0\n"
            "_cell_length_b 5.0\n"
            "_cell_length_c 5.0\n"
            "_cell_angle_alpha 90\n"
            "_cell_angle_beta 90\n"
            "_cell_angle_gamma 90\n"
            "_symmetry_space_group_name_H-M 'P 21/c'\n"
            "loop_\n"
            "_atom_site_label\n"
            "_atom_site_type_symbol\n"
            "_atom_site_fract_x\n"
            "_atom_site_fract_y\n"
            "_atom_site_fract_z\n"
            "C1 C 0.0 0.0 0.0\n"
        )
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always", CIFParseWarning)
            mol = parse_cif(path)
        assert len(mol.atoms) == 1
        cif_warns = [w for w in caught if issubclass(w.category, CIFParseWarning)]
        assert len(cif_warns) == 1
        assert "P 21/c" in str(cif_warns[0].message)

    def test_p1_hm_name_no_warning(self, tmp_path: Path):
        """A P1/P-1 H-M name with no symop loop must NOT warn."""
        import warnings as _warnings

        from moltui.parsers import CIFParseWarning

        path = tmp_path / "p1.cif"
        path.write_text(
            "data_x\n"
            "_cell_length_a 5.0\n"
            "_cell_length_b 5.0\n"
            "_cell_length_c 5.0\n"
            "_cell_angle_alpha 90\n"
            "_cell_angle_beta 90\n"
            "_cell_angle_gamma 90\n"
            "_symmetry_space_group_name_H-M 'P 1'\n"
            "loop_\n"
            "_atom_site_label\n"
            "_atom_site_type_symbol\n"
            "_atom_site_fract_x\n"
            "_atom_site_fract_y\n"
            "_atom_site_fract_z\n"
            "C1 C 0.0 0.0 0.0\n"
        )
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always", CIFParseWarning)
            parse_cif(path)
        assert not [w for w in caught if issubclass(w.category, CIFParseWarning)]

    def test_hm_name_with_symops_no_warning(self, tmp_path: Path):
        """An H-M name accompanied by a symop loop must NOT warn."""
        import warnings as _warnings

        from moltui.parsers import CIFParseWarning

        path = tmp_path / "ok.cif"
        path.write_text(
            "data_x\n"
            "_cell_length_a 5.0\n"
            "_cell_length_b 5.0\n"
            "_cell_length_c 5.0\n"
            "_cell_angle_alpha 90\n"
            "_cell_angle_beta 90\n"
            "_cell_angle_gamma 90\n"
            "_symmetry_space_group_name_H-M 'P 21/c'\n"
            "loop_\n"
            "_symmetry_equiv_pos_as_xyz\n"
            "'x, y, z'\n"
            "'-x, y+1/2, -z+1/2'\n"
            "loop_\n"
            "_atom_site_label\n"
            "_atom_site_type_symbol\n"
            "_atom_site_fract_x\n"
            "_atom_site_fract_y\n"
            "_atom_site_fract_z\n"
            "C1 C 0.1 0.2 0.3\n"
        )
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always", CIFParseWarning)
            parse_cif(path)
        assert not [w for w in caught if issubclass(w.category, CIFParseWarning)]

    def test_uncertainty_parens_stripped(self, tmp_path: Path):
        path = tmp_path / "u.cif"
        path.write_text(
            "data_u\n"
            "_cell_length_a 2.46(1)\n"
            "_cell_length_b 2.46(1)\n"
            "_cell_length_c 6.71(2)\n"
            "_cell_angle_alpha 90.0\n"
            "_cell_angle_beta 90.0\n"
            "_cell_angle_gamma 120.0\n"
            "loop_\n"
            "_atom_site_label\n"
            "_atom_site_type_symbol\n"
            "_atom_site_fract_x\n"
            "_atom_site_fract_y\n"
            "_atom_site_fract_z\n"
            "C1 C 0.0(1) 0.0 0.0\n"
        )
        mol = parse_cif(path)
        assert len(mol.atoms) == 1


class TestParseSymop:
    """Unit tests for the symop string → (rotation, translation) parser."""

    @staticmethod
    def _check(op: str, rot_expected: list[list[float]], trans_expected: list[float]) -> None:
        from moltui.parsers import _parse_symop

        rot, trans = _parse_symop(op)
        np.testing.assert_allclose(rot, np.array(rot_expected, dtype=np.float64), atol=1e-12)
        np.testing.assert_allclose(trans, np.array(trans_expected, dtype=np.float64), atol=1e-12)

    def test_identity(self):
        self._check("x,y,z", [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [0, 0, 0])

    def test_inversion(self):
        self._check("-x,-y,-z", [[-1, 0, 0], [0, -1, 0], [0, 0, -1]], [0, 0, 0])

    def test_screw_21a(self):
        self._check("1/2-x,1/2+y,-z", [[-1, 0, 0], [0, 1, 0], [0, 0, -1]], [0.5, 0.5, 0.0])

    def test_glide(self):
        self._check("1/2+x,1/2-y,z", [[1, 0, 0], [0, -1, 0], [0, 0, 1]], [0.5, 0.5, 0.0])

    def test_permutation(self):
        self._check("y,x,-z+1/2", [[0, 1, 0], [1, 0, 0], [0, 0, -1]], [0, 0, 0.5])

    def test_whitespace_tolerated(self):
        self._check(
            " 1/2 - x , 1/2 + y , - z ", [[-1, 0, 0], [0, 1, 0], [0, 0, -1]], [0.5, 0.5, 0.0]
        )

    def test_multiplied_coefficient(self):
        self._check("2*x,2y,z/2", [[2, 0, 0], [0, 2, 0], [0, 0, 0.5]], [0, 0, 0])

    def test_thirds(self):
        self._check("x+1/3,y+2/3,z", [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [1 / 3, 2 / 3, 0.0])

    def test_leading_plus(self):
        self._check("+x,+y,+z", [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [0, 0, 0])

    def test_quoted_input(self):
        # Quotes can leak through if upstream forgot to strip them.
        self._check("'x,y,z'", [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [0, 0, 0])

    def test_compound_translation(self):
        self._check("x+1/2-1/2,y,z", [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [0, 0, 0])

    def test_negative_var_after_translation(self):
        self._check("1/4-y,1/4+x,1/4+z", [[0, -1, 0], [1, 0, 0], [0, 0, 1]], [0.25, 0.25, 0.25])

    def test_invalid_component_count_raises(self):
        from moltui.parsers import _parse_symop

        with pytest.raises(ValueError, match="3 components"):
            _parse_symop("x,y")

    def test_empty_component_raises(self):
        from moltui.parsers import _parse_symop

        with pytest.raises(ValueError, match="empty component"):
            _parse_symop("x,,z")


class TestApplySymops:
    @staticmethod
    def _ops(specs: list[str]):
        from moltui.parsers import _parse_symop

        return [_parse_symop(s) for s in specs]

    def test_identity_no_duplication(self):
        from moltui.parsers import _apply_symops

        fracs = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
        symbols = ["A", "B"]
        out_fracs, out_syms = _apply_symops(fracs, symbols, self._ops(["x,y,z"]))
        assert out_syms == symbols
        np.testing.assert_allclose(out_fracs, fracs)

    def test_inversion_doubles_general_position(self):
        from moltui.parsers import _apply_symops

        fracs = np.array([[0.1, 0.2, 0.3]])
        out_fracs, out_syms = _apply_symops(fracs, ["X"], self._ops(["x,y,z", "-x,-y,-z"]))
        assert len(out_syms) == 2
        np.testing.assert_allclose(np.sort(out_fracs[:, 0]), [0.1, 0.9])
        np.testing.assert_allclose(np.sort(out_fracs[:, 1]), [0.2, 0.8])

    def test_inversion_keeps_origin_atom_once(self):
        # Atom on the inversion center maps to itself; should not duplicate.
        from moltui.parsers import _apply_symops

        fracs = np.array([[0.0, 0.0, 0.0]])
        out_fracs, out_syms = _apply_symops(fracs, ["A"], self._ops(["x,y,z", "-x,-y,-z"]))
        assert len(out_syms) == 1
        np.testing.assert_allclose(out_fracs[0], [0.0, 0.0, 0.0])

    def test_p21a_general_position_multiplicity(self):
        # P 21/a: 4 ops, generic site → 4 atoms.
        from moltui.parsers import _apply_symops

        fracs = np.array([[0.241, 0.222, -0.099]])
        ops = self._ops(
            [
                "x,y,z",
                "1/2-x,1/2+y,-z",
                "-x,-y,-z",
                "1/2+x,1/2-y,z",
            ]
        )
        out_fracs, out_syms = _apply_symops(fracs, ["C"], ops)
        assert len(out_syms) == 4
        # All within [0, 1) after wrapping.
        assert np.all((out_fracs >= -1e-9) & (out_fracs < 1.0))


class TestCifSymmetryExpansion:
    def test_aspirin_expands_21_to_84(self):
        path = Path(__file__).parent.parent / "data" / "crystal" / "aspirin.cif"
        if not path.exists():
            pytest.skip("aspirin.cif not present")
        mol = parse_cif(path)
        # 21 atoms × 4 ops = 84 atoms with no special-position collapses.
        assert len(mol.atoms) == 84
        from collections import Counter

        counts = Counter(a.element.symbol for a in mol.atoms)
        # Asymmetric unit: 9C, 4O, 8H → ×4
        assert counts["C"] == 36
        assert counts["O"] == 16
        assert counts["H"] == 32

    def test_graphite_identity_only_keeps_4_atoms(self, tmp_path: Path):
        path = tmp_path / "graphite.cif"
        path.write_text(GRAPHITE_CIF)
        mol = parse_cif(path)
        assert len(mol.atoms) == 4

    def test_inversion_only_no_duplication_at_origin(self, tmp_path: Path):
        # Single atom at the inversion center should not duplicate.
        body = (
            "data_inv\n"
            "_cell_length_a 5.0\n"
            "_cell_length_b 5.0\n"
            "_cell_length_c 5.0\n"
            "_cell_angle_alpha 90\n"
            "_cell_angle_beta 90\n"
            "_cell_angle_gamma 90\n"
            "loop_\n_symmetry_equiv_pos_as_xyz\n"
            "'x,y,z'\n'-x,-y,-z'\n"
            "loop_\n"
            "_atom_site_label\n_atom_site_type_symbol\n"
            "_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n"
            "C1 C 0.0 0.0 0.0\n"
        )
        path = tmp_path / "inv.cif"
        path.write_text(body)
        mol = parse_cif(path)
        assert len(mol.atoms) == 1

    def test_inversion_doubles_general_site(self, tmp_path: Path):
        body = (
            "data_inv\n"
            "_cell_length_a 5.0\n"
            "_cell_length_b 5.0\n"
            "_cell_length_c 5.0\n"
            "_cell_angle_alpha 90\n"
            "_cell_angle_beta 90\n"
            "_cell_angle_gamma 90\n"
            "loop_\n_symmetry_equiv_pos_as_xyz\n"
            "'x,y,z'\n'-x,-y,-z'\n"
            "loop_\n"
            "_atom_site_label\n_atom_site_type_symbol\n"
            "_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n"
            "C1 C 0.1 0.2 0.3\n"
        )
        path = tmp_path / "g.cif"
        path.write_text(body)
        mol = parse_cif(path)
        assert len(mol.atoms) == 2

    def test_alternative_symop_header(self, tmp_path: Path):
        # Newer CIF dictionary uses _space_group_symop_operation_xyz.
        body = (
            "data_alt\n"
            "_cell_length_a 5.0\n_cell_length_b 5.0\n_cell_length_c 5.0\n"
            "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
            "loop_\n_space_group_symop_operation_xyz\n"
            "'x,y,z'\n'-x,-y,-z'\n"
            "loop_\n"
            "_atom_site_label\n_atom_site_type_symbol\n"
            "_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n"
            "C1 C 0.1 0.2 0.3\n"
        )
        path = tmp_path / "alt.cif"
        path.write_text(body)
        mol = parse_cif(path)
        assert len(mol.atoms) == 2

    def test_symop_loop_with_id_column(self, tmp_path: Path):
        # id column followed by op column (common in CCDC-exported CIFs).
        body = (
            "data_with_id\n"
            "_cell_length_a 5.0\n_cell_length_b 5.0\n_cell_length_c 5.0\n"
            "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
            "loop_\n_symmetry_equiv_pos_site_id\n_symmetry_equiv_pos_as_xyz\n"
            "1 'x,y,z'\n2 '-x,-y,-z'\n"
            "loop_\n"
            "_atom_site_label\n_atom_site_type_symbol\n"
            "_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n"
            "C1 C 0.1 0.2 0.3\n"
        )
        path = tmp_path / "id.cif"
        path.write_text(body)
        mol = parse_cif(path)
        assert len(mol.atoms) == 2


def test_parse_orca_hess_data_includes_normal_modes(tmp_path: Path) -> None:
    hess_file = _write_orca_hess(tmp_path)
    hess_data = parse_orca_hess_data(hess_file)

    assert len(hess_data.molecule.atoms) == 3
    assert hess_data.frequencies is not None
    assert hess_data.normal_modes is not None
    assert hess_data.frequencies.shape == (9,)
    assert hess_data.normal_modes.shape == (9, 3, 3)
