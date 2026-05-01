from types import SimpleNamespace

import numpy as np

from moltui.app import _build_view_render_scene, _export_render_kwargs
from moltui.elements import Atom, Molecule, get_element


def test_export_render_kwargs_use_view_lighting_and_geometry_settings():
    view = SimpleNamespace(
        pan_x=1.25,
        pan_y=-0.75,
        licorice=True,
        vdw=False,
        ambient=0.46,
        diffuse=0.58,
        specular=0.21,
        shininess=44.0,
        atom_scale=0.42,
        bond_radius=0.11,
    )

    kwargs = _export_render_kwargs(view)

    assert kwargs["ssaa"] == 2
    assert kwargs["pan"] == (1.25, -0.75)
    assert kwargs["licorice"] is True
    assert kwargs["vdw"] is False
    assert kwargs["ambient"] == 0.46
    assert kwargs["diffuse"] == 0.58
    assert kwargs["specular"] == 0.21
    assert kwargs["shininess"] == 44.0
    assert kwargs["atom_scale"] == 0.42
    assert kwargs["bond_radius"] == 0.11
    assert kwargs["cell_dash"] == (5, 3)
    assert kwargs["cell_line_width"] == 2


def test_build_view_render_scene_includes_periodic_ghosts_for_export_path():
    C = get_element("C")
    lattice = np.array(
        [
            [2.46, 0.0, 0.0],
            [-1.23, 2.13042249, 0.0],
            [0.0, 0.0, 6.71],
        ]
    )
    molecule = Molecule(
        atoms=[
            Atom(C, np.array([0.0, 0.0, 0.0])),
            Atom(C, np.array([0.0, 1.42028166, 0.0])),
            Atom(C, np.array([0.0, 0.0, 3.355])),
            Atom(C, np.array([1.23, 0.71014083, 3.355])),
        ],
        bonds=[],
        lattice=lattice,
    )
    molecule.detect_bonds_periodic()

    view = SimpleNamespace(
        molecule=molecule,
        isosurfaces=[],
        show_orbitals=False,
        show_replication=True,
        show_cell=True,
        show_bonds=True,
        supercell_dims=(1, 1, 1),
    )

    render_mol, isos, cell_dims = _build_view_render_scene(view)

    assert len(render_mol.bonds) == 36
    assert isos is None
    assert cell_dims == (1, 1, 1)
