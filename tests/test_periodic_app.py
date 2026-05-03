#!/usr/bin/env python3
"""App-level regression tests for periodic structures and sidebar state."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from tests.test_parsers import GRAPHITE_CIF


def _install_skimage_stub() -> None:
    if "skimage.measure" in sys.modules:
        return
    skimage = types.ModuleType("skimage")
    measure = types.ModuleType("skimage.measure")

    def _marching_cubes(*_args, **_kwargs):  # pragma: no cover
        raise RuntimeError("marching_cubes stub called unexpectedly")

    measure.marching_cubes = _marching_cubes
    skimage.measure = measure
    sys.modules["skimage"] = skimage
    sys.modules["skimage.measure"] = measure


def _write_graphite_cif(tmp_path: Path) -> Path:
    path = tmp_path / "graphite.cif"
    path.write_text(GRAPHITE_CIF)
    return path


def _write_graphite_extxyz(tmp_path: Path) -> Path:
    path = tmp_path / "graphite.extxyz"
    path.write_text(
        "\n".join(
            [
                "4",
                (
                    'Lattice="2.46 0.0 0.0 -1.23 2.13042249 0.0 0.0 0.0 6.71" '
                    'Properties="species:S:1:pos:R:3"'
                ),
                "C 0.0 0.0 0.0",
                "C 0.0 1.42028166 0.0",
                "C 0.0 0.0 3.355",
                "C 1.23 0.71014083 3.355",
                "",
            ]
        )
    )
    return path


def _plain_molecule():
    from moltui.elements import Atom, Molecule, get_element

    atoms = [
        Atom(get_element("O"), np.array([0.0, 0.0, 0.0])),
        Atom(get_element("H"), np.array([0.9, 0.0, 0.0])),
        Atom(get_element("H"), np.array([-0.3, 0.8, 0.0])),
    ]
    molecule = Molecule(atoms=atoms, bonds=[])
    molecule.detect_bonds()
    return molecule


def _periodic_h2():
    from moltui.elements import Atom, Molecule, get_element

    H = get_element("H")
    mol = Molecule(
        atoms=[
            Atom(H, np.array([0.0, 0.0, 0.0])),
            Atom(H, np.array([1.9, 0.0, 0.0])),
        ],
        bonds=[],
        lattice=np.diag([2.0, 2.0, 2.0]),
    )
    mol.detect_bonds_periodic()
    return mol


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("writer", "parser_name"),
    [
        (_write_graphite_cif, "parse_cif"),
        (_write_graphite_extxyz, "parse_xyz"),
    ],
)
async def test_b_toggle_refreshes_periodic_geometry_panel(
    tmp_path: Path,
    writer,
    parser_name: str,
) -> None:
    _install_skimage_stub()

    from moltui.app import MoltuiApp
    from moltui.parsers import parse_cif, parse_xyz

    parser = {"parse_cif": parse_cif, "parse_xyz": parse_xyz}[parser_name]
    path = writer(tmp_path)
    app = MoltuiApp(molecule=parser(path), filepath=str(path))

    async with app.run_test() as pilot:
        await pilot.pause()
        assert app._display_geometry is not None
        assert len(app._display_geometry.molecule.bonds) == 6

        await pilot.press("b")
        await pilot.pause()
        assert app._display_geometry is not None
        assert len(app._display_geometry.molecule.bonds) == 2

        await pilot.press("b")
        await pilot.pause()
        assert app._display_geometry is not None
        assert len(app._display_geometry.molecule.bonds) == 6


@pytest.mark.asyncio
async def test_m_from_visual_mode_restores_geometry_panel() -> None:
    _install_skimage_stub()

    from moltui.app import MoltuiApp
    from moltui.geometry_panel import GeometryPanel
    from moltui.visual_panel import VisualPanel

    app = MoltuiApp(molecule=_plain_molecule(), filepath="sample.xyz")

    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("V")
        await pilot.pause()

        visual = app.query_one(VisualPanel)
        geometry = app.query_one(GeometryPanel)
        assert visual.has_class("visible")
        assert not geometry.has_class("visible")

        await pilot.press("m")
        await pilot.pause()

        assert not visual.has_class("visible")
        assert geometry.has_class("visible")


@pytest.mark.asyncio
async def test_sidebar_visual_restore_v_s_s_keeps_visual_panel() -> None:
    _install_skimage_stub()

    from moltui.app import MoltuiApp
    from moltui.geometry_panel import GeometryPanel
    from moltui.visual_panel import VisualPanel

    app = MoltuiApp(molecule=_plain_molecule(), filepath="sample.xyz")

    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("V")
        await pilot.pause()

        visual = app.query_one(VisualPanel)
        geometry = app.query_one(GeometryPanel)
        assert visual.has_class("visible")

        await pilot.press("S")
        await pilot.pause()
        assert not visual.has_class("visible")
        assert not geometry.has_class("visible")

        await pilot.press("S")
        await pilot.pause()
        assert visual.has_class("visible")
        assert not geometry.has_class("visible")


def test_pan_clamp_uses_periodic_cell_radius_for_single_atom() -> None:
    _install_skimage_stub()

    from moltui.app import MoleculeView
    from moltui.elements import Atom, Molecule, get_element

    H = get_element("H")
    mol = Molecule(
        atoms=[Atom(H, np.array([0.2, 0.4, 0.6]))],
        bonds=[],
        lattice=np.array([[2.0, 0.0, 0.0], [4.0, 6.0, 0.0], [-6.0, 2.0, 10.0]]),
    )
    view = MoleculeView()
    view.set_molecule(mol)
    view.pan_x = 1.0
    view.pan_y = -1.0

    view._clamp_pan()

    assert view.pan_x != 0.0
    assert view.pan_y != 0.0
    assert view.camera_distance > 4.0


@pytest.mark.asyncio
async def test_periodic_bond_highlight_resolves_to_adjacent_display_atoms() -> None:
    _install_skimage_stub()

    from textual.widgets import DataTable

    from moltui.app import MoleculeView, MoltuiApp, _position_key
    from moltui.geometry_panel import GeometryPanel

    app = MoltuiApp(molecule=_periodic_h2(), filepath="sample.extxyz")

    async with app.run_test() as pilot:
        await pilot.pause()
        panel = app.query_one(GeometryPanel)
        table = panel.query_one("#bonds-table", DataTable)
        view = app.query_one(MoleculeView)

        table.move_cursor(row=0)
        panel._emit_current_highlight(table)
        await pilot.pause()

        assert app._display_geometry is not None
        positions = [
            atom.position
            for atom in app._display_geometry.molecule.atoms
            if _position_key(atom.position) in view.highlighted_display_positions
        ]
        assert len(positions) == 2
        assert np.linalg.norm(positions[0] - positions[1]) == pytest.approx(0.1, abs=1e-6)
        assert np.linalg.norm(
            app.molecule.atoms[0].position - app.molecule.atoms[1].position
        ) == pytest.approx(1.9, abs=1e-6)
