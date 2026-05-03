from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from moltui.app import MoltuiApp, _default_volume_isovalue, _detect_filetype
from moltui.elements import Molecule
from moltui.isosurface import extract_isosurfaces
from moltui.parsers import (
    VolumetricData,
    parse_cube_data,
    parse_vasp_volumetric_data,
    parse_xsf_volumetric_data,
)


def _write_tiny_cube(tmp_path: Path) -> Path:
    path = tmp_path / "tiny.cube"
    path.write_text(
        "\n".join(
            [
                "Cube file",
                "periodic test",
                "2 0.0 0.0 0.0",
                "2 1.0 0.0 0.0",
                "2 0.0 1.0 0.0",
                "2 0.0 0.0 1.0",
                "1 0.0 0.0 0.0 0.1",
                "1 0.0 0.0 0.0 1.9",
                "0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8",
                "",
            ]
        )
    )
    return path


def test_cube_periodic_option_assigns_lattice_and_periodic_bonds(tmp_path: Path):
    path = _write_tiny_cube(tmp_path)

    default = parse_cube_data(path)
    periodic = parse_cube_data(path, periodic=True)

    assert default.molecule.lattice is None
    assert periodic.periodic is True
    assert periodic.molecule.lattice is not None
    assert periodic.molecule.pbc == (True, True, True)
    np.testing.assert_allclose(periodic.molecule.lattice, np.eye(3) * 2.0 * 0.529177249)
    assert len(default.molecule.bonds) == 0
    assert len(periodic.molecule.bonds) == 1


def _write_zero_axis_cube(tmp_path: Path) -> Path:
    path = tmp_path / "slice.cube"
    path.write_text(
        "\n".join(
            [
                "Cube file",
                "2D slice",
                "2 0.0 0.0 0.0",
                "2 1.0 0.0 0.0",
                "2 0.0 1.0 0.0",
                "2 0.0 0.0 0.0",
                "1 0.0 0.0 0.0 0.1",
                "1 0.0 0.0 0.0 1.9",
                "0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8",
                "",
            ]
        )
    )
    return path


def test_cube_periodic_option_infers_nonperiodic_axes_from_zero_grid_vectors(tmp_path: Path):
    periodic = parse_cube_data(_write_zero_axis_cube(tmp_path), periodic=True)

    assert periodic.molecule.pbc == (True, True, False)
    np.testing.assert_allclose(
        periodic.molecule.lattice,
        np.diag([2.0 * 0.529177249, 2.0 * 0.529177249, 0.0]),
    )
    assert len(periodic.molecule.bonds) == 0


def test_zero_axis_periodic_cube_can_extract_isosurface(tmp_path: Path):
    periodic = parse_cube_data(_write_zero_axis_cube(tmp_path), periodic=True)

    meshes = extract_isosurfaces(periodic, isovalue=0.45)

    assert len(meshes) == 1
    assert all(np.isfinite(mesh.vertices).all() for mesh in meshes)
    assert all(np.isfinite(mesh.normals).all() for mesh in meshes)


def test_xsf_datagrid_parses_first_grid(tmp_path: Path):
    path = tmp_path / "grid.xsf"
    path.write_text(
        "\n".join(
            [
                "CRYSTAL",
                "PRIMVEC",
                "2 0 0",
                "0 2 0",
                "0 0 2",
                "PRIMCOORD",
                "1 1",
                "H 0 0 0",
                "BEGIN_BLOCK_DATAGRID_3D",
                "density",
                "BEGIN_DATAGRID_3D_density",
                "2 2 2",
                "0 0 0",
                "2 0 0",
                "0 2 0",
                "0 0 2",
                "0 1 2 3 4 5 6 7",
                "END_DATAGRID_3D",
                "END_BLOCK_DATAGRID_3D",
                "",
            ]
        )
    )

    volume = parse_xsf_volumetric_data(path)

    assert volume.periodic is True
    assert volume.n_points == (2, 2, 2)
    np.testing.assert_allclose(volume.origin, [0, 0, 0])
    np.testing.assert_allclose(volume.axes, np.eye(3) * 2.0)
    np.testing.assert_allclose(volume.data.ravel(), np.arange(8))


def test_vasp_volumetric_parser_reads_structure_and_grid(tmp_path: Path):
    path = tmp_path / "CHGCAR"
    path.write_text(
        "\n".join(
            [
                "H",
                "1.0",
                "2 0 0",
                "0 2 0",
                "0 0 2",
                "H",
                "1",
                "Direct",
                "0 0 0",
                "",
                "2 2 2",
                "0 1 2 3 4 5 6 7",
                "",
            ]
        )
    )

    assert _detect_filetype(str(path)) == "vasp-volumetric"
    volume = parse_vasp_volumetric_data(path)

    assert volume.periodic is True
    assert volume.n_points == (2, 2, 2)
    np.testing.assert_allclose(volume.molecule.lattice, np.eye(3) * 2.0)
    np.testing.assert_allclose(volume.axes, np.eye(3))
    np.testing.assert_allclose(volume.data.ravel(), np.arange(8))


def test_positive_volumetric_isovalue_defaults_and_slider_range_follow_data():
    volume = VolumetricData(
        molecule=Molecule(atoms=[], bonds=[]),
        origin=np.zeros(3),
        axes=np.eye(3),
        n_points=(2, 2, 2),
        data=np.linspace(0.4, 1.2, 8).reshape((2, 2, 2)),
    )
    app = MoltuiApp(molecule=volume.molecule, isovalue=_default_volume_isovalue(volume))
    app._volumetric_data = volume

    assert app.isovalue == 0.8
    assert app._isovalue_range() == (0.4, 1.2, pytest.approx(0.008))


def test_isosurface_uses_full_grid_axes_transform():
    data = np.zeros((2, 2, 2), dtype=np.float64)
    data[1, :, :] = 1.0
    volume = VolumetricData(
        molecule=Molecule(atoms=[], bonds=[]),
        origin=np.array([1.0, 2.0, 3.0]),
        axes=np.array([[2.0, 1.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]]),
        n_points=(2, 2, 2),
        data=data,
    )

    meshes = extract_isosurfaces(volume, isovalue=0.5)

    assert len(meshes) == 1
    np.testing.assert_allclose(meshes[0].vertices[:, 0], 2.0)
    np.testing.assert_allclose(np.sort(np.unique(meshes[0].vertices[:, 1])), [2.5, 5.5])
