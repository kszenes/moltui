#!/usr/bin/env python3
"""Tests for trajectory autoplay and visual panel focus with animation data."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest


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


def _make_molecule():
    from moltui.elements import Atom, Molecule, get_element

    atoms = [
        Atom(get_element("O"), np.array([0.0, 0.0, 0.0])),
        Atom(get_element("H"), np.array([0.9, 0.0, 0.0])),
        Atom(get_element("H"), np.array([-0.3, 0.8, 0.0])),
    ]
    molecule = Molecule(atoms=atoms, bonds=[])
    molecule.detect_bonds()
    return molecule


def _make_trajectory_data(n_frames: int = 3):
    from moltui.app import TrajectoryData

    frames = np.zeros((n_frames, 3, 3), dtype=np.float64)
    return TrajectoryData(frames=frames)


def _make_normal_mode_data():
    from moltui.app import NormalModeData

    eq_coords = np.zeros((3, 3), dtype=np.float64)
    mode_vectors = np.zeros((3, 3, 3), dtype=np.float64)
    mode_vectors[:, :, 0] = 0.01
    return NormalModeData(
        equilibrium_coords=eq_coords,
        mode_vectors=mode_vectors,
        frequencies=np.array([1000.0, 2000.0, 3000.0]),
    )


def _make_periodic_trajectory_case():
    from moltui.app import TrajectoryData
    from moltui.elements import Atom, Molecule, get_element

    H = get_element("H")
    lattice = np.diag([2.0, 2.0, 2.0])
    molecule = Molecule(
        atoms=[
            Atom(H, np.array([0.0, 0.0, 0.0])),
            Atom(H, np.array([1.9, 0.0, 0.0])),
        ],
        bonds=[],
        lattice=lattice,
    )
    molecule.detect_bonds_periodic()
    frames = np.array(
        [
            [[0.0, 0.0, 0.0], [1.9, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.9, 0.1, 0.0]],
        ],
        dtype=np.float64,
    )
    return molecule, TrajectoryData(frames=frames)


def _make_periodic_selection_case():
    from moltui.elements import Atom, Molecule, get_element

    H = get_element("H")
    lattice = np.diag([2.0, 2.0, 2.0])
    molecule = Molecule(
        atoms=[
            Atom(H, np.array([0.0, 0.0, 0.0])),
            Atom(H, np.array([0.7, 0.0, 0.0])),
            Atom(H, np.array([1.9, 0.0, 0.0])),
        ],
        bonds=[],
        lattice=lattice,
    )
    molecule.detect_bonds_periodic()
    return molecule


@pytest.mark.asyncio
async def test_trajectory_autoplays_on_mount() -> None:
    _install_skimage_stub()

    from moltui.app import MoltuiApp

    molecule = _make_molecule()
    trajectory_data = _make_trajectory_data(n_frames=3)
    app = MoltuiApp(molecule=molecule, filepath="sample.xyz", trajectory_data=trajectory_data)

    async with app.run_test() as pilot:
        await pilot.pause()
        assert app._is_playing


@pytest.mark.asyncio
async def test_single_frame_trajectory_does_not_autoplay() -> None:
    _install_skimage_stub()

    from moltui.app import MoltuiApp

    molecule = _make_molecule()
    trajectory_data = _make_trajectory_data(n_frames=1)
    app = MoltuiApp(molecule=molecule, filepath="sample.xyz", trajectory_data=trajectory_data)

    async with app.run_test() as pilot:
        await pilot.pause()
        assert not app._is_playing


@pytest.mark.asyncio
async def test_opening_visual_panel_focuses_trajectory_slider() -> None:
    _install_skimage_stub()

    from moltui.app import MoltuiApp
    from moltui.visual_panel import Slider, VisualPanel

    molecule = _make_molecule()
    trajectory_data = _make_trajectory_data(n_frames=3)
    app = MoltuiApp(molecule=molecule, filepath="sample.xyz", trajectory_data=trajectory_data)

    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("V")
        await pilot.pause()

        visual_panel = app.query_one(VisualPanel)
        slider = visual_panel.query_one("#slider-trajectory-speed", Slider)
        assert visual_panel.has_class("visible")
        assert slider.has_focus


@pytest.mark.asyncio
async def test_opening_visual_panel_focuses_vibrational_slider() -> None:
    _install_skimage_stub()

    from moltui.app import MoltuiApp
    from moltui.visual_panel import Slider, VisualPanel

    molecule = _make_molecule()
    normal_mode_data = _make_normal_mode_data()
    app = MoltuiApp(molecule=molecule, filepath="sample.xyz", normal_mode_data=normal_mode_data)

    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("V")
        await pilot.pause()

        visual_panel = app.query_one(VisualPanel)
        slider = visual_panel.query_one("#slider-vibrational-speed", Slider)
        assert visual_panel.has_class("visible")
        assert slider.has_focus


@pytest.mark.asyncio
async def test_opening_visual_panel_focuses_atom_scale_for_plain_molecule() -> None:
    _install_skimage_stub()

    from moltui.app import MoltuiApp
    from moltui.visual_panel import Slider, VisualPanel

    molecule = _make_molecule()
    app = MoltuiApp(molecule=molecule, filepath="sample.xyz")

    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("V")
        await pilot.pause()

        visual_panel = app.query_one(VisualPanel)
        assert visual_panel.has_class("visible")
        assert visual_panel.query_one("#slider-atom-scale", Slider).has_focus


@pytest.mark.asyncio
async def test_periodic_trajectory_refreshes_bonds_and_display_geometry() -> None:
    _install_skimage_stub()

    from moltui.app import MoltuiApp
    from moltui.geometry_panel import GeometryPanel

    molecule, trajectory_data = _make_periodic_trajectory_case()
    app = MoltuiApp(molecule=molecule, filepath="sample.extxyz", trajectory_data=trajectory_data)

    async with app.run_test() as pilot:
        await pilot.pause()
        app._stop_playback()
        assert app.trajectory_data is not None
        app.trajectory_data.frame_index = 1
        app._apply_active_animation_geometry()
        await pilot.pause()

        assert app.molecule.bonds == [(0, 1)]
        assert app.molecule.bond_shifts == [(-1, 0, 0)]
        assert app._display_geometry is not None
        assert any(
            np.isclose(atom.position[1], 0.1) for atom in app._display_geometry.molecule.atoms
        )

        panel = app.query_one(GeometryPanel)
        assert panel._molecule is app._display_geometry.molecule


@pytest.mark.asyncio
async def test_periodic_replication_toggle_preserves_selection_mapping() -> None:
    _install_skimage_stub()

    from textual.widgets import DataTable

    from moltui.app import MoleculeView, MoltuiApp, TrajectoryData
    from moltui.geometry_panel import GeometryPanel

    molecule = _make_periodic_selection_case()
    trajectory_data = TrajectoryData(
        frames=np.array([[a.position for a in molecule.atoms]], dtype=np.float64)
    )
    app = MoltuiApp(molecule=molecule, filepath="sample.extxyz", trajectory_data=trajectory_data)

    async with app.run_test() as pilot:
        await pilot.pause()
        app._stop_playback()
        panel = app.query_one(GeometryPanel)
        table = panel.query_one("#bonds-table", DataTable)
        view = app.query_one(MoleculeView)
        assert app._display_geometry is not None
        assert table.row_count >= 1

        for row in range(table.row_count):
            table.move_cursor(row=row)
            panel._emit_current_highlight(table)
            await pilot.pause()
            if len(view.highlighted_display_positions) == 2:
                break
        assert len(view.highlighted_display_positions) == 2

        await pilot.press("b")
        await pilot.pause()
        assert app._display_geometry is not None
        assert table.cursor_row == 0
        assert panel._molecule is app._display_geometry.molecule
        assert len(view.highlighted_display_positions) == 2

        await pilot.press("b")
        await pilot.pause()
        assert app._display_geometry is not None
        assert table.cursor_row == 0
        assert panel._molecule is app._display_geometry.molecule
        assert len(view.highlighted_display_positions) == 2
