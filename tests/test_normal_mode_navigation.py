#!/usr/bin/env python3
"""Tests for normal-mode panel navigation behavior."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest


def _install_skimage_stub() -> None:
    """Provide a lightweight skimage stub so app imports in test envs."""
    if "skimage.measure" in sys.modules:
        return
    skimage = types.ModuleType("skimage")
    measure = types.ModuleType("skimage.measure")

    def _marching_cubes(*_args, **kwargs):  # pragma: no cover - never used in this test
        raise RuntimeError("marching_cubes stub called unexpectedly")

    measure.marching_cubes = _marching_cubes
    skimage.measure = measure
    sys.modules["skimage"] = skimage
    sys.modules["skimage.measure"] = measure


@pytest.mark.asyncio
async def test_panel_next_prev_actions_select_normal_mode() -> None:
    _install_skimage_stub()

    from moltui.app import MoltuiApp, NormalModeData
    from moltui.elements import Atom, Molecule, get_element
    from moltui.normal_mode_panel import NormalModePanel

    atoms = [
        Atom(get_element("O"), np.array([0.0, 0.0, 0.0])),
        Atom(get_element("H"), np.array([0.9, 0.0, 0.0])),
        Atom(get_element("H"), np.array([-0.3, 0.8, 0.0])),
    ]
    molecule = Molecule(atoms=atoms, bonds=[])
    molecule.detect_bonds()

    # NormalModeData passed to the app is expected to be already pre-filtered
    # of rigid-body modes (filtering happens at file-load).
    n_modes = 3
    mode_vectors = np.zeros((n_modes, len(atoms), 3), dtype=np.float64)
    for i in range(n_modes):
        mode_vectors[i, :, 0] = 0.01 * (i + 1)

    normal_mode_data = NormalModeData(
        equilibrium_coords=np.array([a.position.copy() for a in atoms]),
        mode_vectors=mode_vectors,
        frequencies=np.array([2041.3, 4493.7, 4796.3]),
    )

    app = MoltuiApp(
        molecule=molecule,
        filepath="sample.molden",
        normal_mode_data=normal_mode_data,
    )

    async with app.run_test() as pilot:
        app.action_toggle_normal_mode_panel()
        await pilot.pause()
        assert app.query_one(NormalModePanel).has_class("visible")
        assert app.normal_mode_data is not None
        assert app.normal_mode_data.mode_index == 0
        assert app._is_playing

        app.action_panel_next()
        await pilot.pause()
        assert app.normal_mode_data.mode_index == 1

        app.action_panel_prev()
        await pilot.pause()
        assert app.normal_mode_data.mode_index == 0


@pytest.mark.asyncio
async def test_m_and_capital_m_cycle_view_modes() -> None:
    _install_skimage_stub()

    from moltui.app import MoleculeView, MoltuiApp, NormalModeData
    from moltui.elements import Atom, Molecule, get_element
    from moltui.mo_panel import MOPanel
    from moltui.normal_mode_panel import NormalModePanel

    atoms = [
        Atom(get_element("O"), np.array([0.0, 0.0, 0.0])),
        Atom(get_element("H"), np.array([0.9, 0.0, 0.0])),
        Atom(get_element("H"), np.array([-0.3, 0.8, 0.0])),
    ]
    molecule = Molecule(atoms=atoms, bonds=[])
    molecule.detect_bonds()

    mode_vectors = np.zeros((3, len(atoms), 3), dtype=np.float64)
    mode_vectors[:, :, 0] = 0.01
    normal_mode_data = NormalModeData(
        equilibrium_coords=np.array([a.position.copy() for a in atoms]),
        mode_vectors=mode_vectors,
        frequencies=np.array([2000.0, 4000.0, 4700.0]),
    )

    orbital_data = SimpleNamespace(
        mo_energies=np.array([-0.5]),
        mo_occupations=np.array([2.0]),
        mo_symmetries=["A1"],
        mo_spins=["Alpha"],
        n_mos=1,
        homo_idx=0,
        has_mo_energies=True,
        has_mo_occupations=True,
    )

    app = MoltuiApp(
        molecule=molecule,
        filepath="sample.molden",
        orbital_data=orbital_data,
        current_mo=0,
        normal_mode_data=normal_mode_data,
    )

    async with app.run_test() as pilot:
        assert app.query_one(MOPanel).has_class("visible")
        assert not app.query_one(NormalModePanel).has_class("visible")
        assert app.query_one(MoleculeView).show_orbitals

        await pilot.press("m")
        await pilot.pause()
        assert app.query_one(NormalModePanel).has_class("visible")
        assert not app.query_one(MoleculeView).show_orbitals
        assert app._is_playing

        await pilot.press("m")
        await pilot.pause()
        assert not app.query_one(MOPanel).has_class("visible")
        assert not app.query_one(NormalModePanel).has_class("visible")
        assert not app._is_playing

        await pilot.press("M")
        await pilot.pause()
        assert app.query_one(NormalModePanel).has_class("visible")
        assert app._is_playing

        await pilot.press("M")
        await pilot.pause()
        assert app.query_one(MOPanel).has_class("visible")
        assert app.query_one(MoleculeView).show_orbitals
        assert not app._is_playing


@pytest.mark.asyncio
async def test_panel_next_prev_with_trimmed_vibrational_modes() -> None:
    _install_skimage_stub()

    from moltui.app import MoltuiApp, NormalModeData
    from moltui.elements import Atom, Molecule, get_element
    from moltui.normal_mode_panel import NormalModePanel

    atoms = [
        Atom(get_element("O"), np.array([0.0, 0.0, 0.0])),
        Atom(get_element("H"), np.array([0.9, 0.0, 0.0])),
        Atom(get_element("H"), np.array([-0.3, 0.8, 0.0])),
    ]
    molecule = Molecule(atoms=atoms, bonds=[])
    molecule.detect_bonds()

    # Typical PySCF-style trimmed output: only vibrational modes (no 6 zero modes).
    n_modes = 3
    mode_vectors = np.zeros((n_modes, len(atoms), 3), dtype=np.float64)
    for i in range(n_modes):
        mode_vectors[i, :, 0] = 0.01 * (i + 1)

    normal_mode_data = NormalModeData(
        equilibrium_coords=np.array([a.position.copy() for a in atoms]),
        mode_vectors=mode_vectors,
        frequencies=np.array([2041.3, 4493.7, 4796.3]),
    )

    app = MoltuiApp(
        molecule=molecule,
        filepath="sample_trimmed.molden",
        normal_mode_data=normal_mode_data,
    )

    async with app.run_test() as pilot:
        app.action_toggle_normal_mode_panel()
        await pilot.pause()
        assert app.query_one(NormalModePanel).has_class("visible")
        assert app.normal_mode_data is not None
        assert app.normal_mode_data.mode_index == 0

        app.action_panel_next()
        await pilot.pause()
        assert app.normal_mode_data.mode_index == 1

        app.action_panel_prev()
        await pilot.pause()
        assert app.normal_mode_data.mode_index == 0


@pytest.mark.asyncio
async def test_mo_and_normal_mode_views_are_mutually_exclusive() -> None:
    _install_skimage_stub()

    from moltui.app import MoleculeView, MoltuiApp, NormalModeData
    from moltui.elements import Atom, Molecule, get_element

    atoms = [
        Atom(get_element("O"), np.array([0.0, 0.0, 0.0])),
        Atom(get_element("H"), np.array([0.9, 0.0, 0.0])),
        Atom(get_element("H"), np.array([-0.3, 0.8, 0.0])),
    ]
    molecule = Molecule(atoms=atoms, bonds=[])
    molecule.detect_bonds()

    mode_vectors = np.zeros((9, len(atoms), 3), dtype=np.float64)
    mode_vectors[:, :, 0] = 0.01
    normal_mode_data = NormalModeData(
        equilibrium_coords=np.array([a.position.copy() for a in atoms]),
        mode_vectors=mode_vectors,
        frequencies=np.arange(9.0),
    )

    orbital_data = SimpleNamespace(
        mo_energies=np.array([-0.5]),
        mo_occupations=np.array([2.0]),
        mo_symmetries=["A1"],
        mo_spins=["Alpha"],
        n_mos=1,
        homo_idx=0,
        has_mo_energies=True,
        has_mo_occupations=True,
    )

    app = MoltuiApp(
        molecule=molecule,
        filepath="sample.molden",
        orbital_data=orbital_data,
        current_mo=0,
        normal_mode_data=normal_mode_data,
    )

    async with app.run_test() as pilot:
        app.action_toggle_normal_mode_panel()
        await pilot.pause()
        view = app.query_one(MoleculeView)
        assert app._is_playing
        assert not view.show_orbitals

        app.action_toggle_mo_panel()
        await pilot.pause()
        view = app.query_one(MoleculeView)
        assert not app._is_playing
        assert view.show_orbitals


@pytest.mark.asyncio
async def test_header_shows_only_active_view_info() -> None:
    _install_skimage_stub()

    from moltui.app import MoltuiApp, NormalModeData
    from moltui.elements import Atom, Molecule, get_element

    atoms = [
        Atom(get_element("O"), np.array([0.0, 0.0, 0.0])),
        Atom(get_element("H"), np.array([0.9, 0.0, 0.0])),
        Atom(get_element("H"), np.array([-0.3, 0.8, 0.0])),
    ]
    molecule = Molecule(atoms=atoms, bonds=[])
    molecule.detect_bonds()

    normal_mode_data = NormalModeData(
        equilibrium_coords=np.array([a.position.copy() for a in atoms]),
        mode_vectors=np.zeros((3, len(atoms), 3), dtype=np.float64),
        frequencies=np.array([2000.0, 4000.0, 4700.0]),
    )
    orbital_data = SimpleNamespace(
        mo_energies=np.array([-0.5]),
        mo_occupations=np.array([2.0]),
        mo_symmetries=["A1"],
        mo_spins=["Alpha"],
        n_mos=1,
        homo_idx=0,
        has_mo_energies=True,
        has_mo_occupations=True,
    )

    app = MoltuiApp(
        molecule=molecule,
        filepath="sample.molden",
        orbital_data=orbital_data,
        current_mo=0,
        normal_mode_data=normal_mode_data,
    )

    async with app.run_test() as pilot:
        assert "MO " in app.title
        assert "Mode " not in app.title

        await pilot.press("m")
        await pilot.pause()
        assert "Mode " in app.title
        assert "MO " not in app.title


@pytest.mark.asyncio
async def test_mode_specific_actions_disabled_outside_active_mode() -> None:
    _install_skimage_stub()

    from moltui.app import MoltuiApp, NormalModeData
    from moltui.elements import Atom, Molecule, get_element

    atoms = [
        Atom(get_element("O"), np.array([0.0, 0.0, 0.0])),
        Atom(get_element("H"), np.array([0.9, 0.0, 0.0])),
        Atom(get_element("H"), np.array([-0.3, 0.8, 0.0])),
    ]
    molecule = Molecule(atoms=atoms, bonds=[])
    molecule.detect_bonds()
    orbital_data = SimpleNamespace(
        mo_energies=np.array([-0.5]),
        mo_occupations=np.array([2.0]),
        mo_symmetries=["A1"],
        mo_spins=["Alpha"],
        n_mos=1,
        homo_idx=0,
        has_mo_energies=True,
        has_mo_occupations=True,
    )
    normal_mode_data = NormalModeData(
        equilibrium_coords=np.array([a.position.copy() for a in atoms]),
        mode_vectors=np.zeros((3, len(atoms), 3), dtype=np.float64),
        frequencies=np.array([2000.0, 4000.0, 4700.0]),
    )
    app = MoltuiApp(
        molecule=molecule,
        filepath="sample.molden",
        orbital_data=orbital_data,
        current_mo=0,
        normal_mode_data=normal_mode_data,
    )

    async with app.run_test() as pilot:
        # Starts in MO mode
        assert app.check_action("toggle_playback", tuple()) is False

        await pilot.press("m")  # move to normal mode
        await pilot.pause()
        assert app.check_action("next_mo", tuple()) is False


@pytest.mark.asyncio
async def test_animation_tick_safe_after_app_unmount() -> None:
    _install_skimage_stub()

    from moltui.app import MoltuiApp, NormalModeData
    from moltui.elements import Atom, Molecule, get_element

    atoms = [
        Atom(get_element("O"), np.array([0.0, 0.0, 0.0])),
        Atom(get_element("H"), np.array([0.9, 0.0, 0.0])),
        Atom(get_element("H"), np.array([-0.3, 0.8, 0.0])),
    ]
    molecule = Molecule(atoms=atoms, bonds=[])
    molecule.detect_bonds()
    normal_mode_data = NormalModeData(
        equilibrium_coords=np.array([a.position.copy() for a in atoms]),
        mode_vectors=np.zeros((3, len(atoms), 3), dtype=np.float64),
        frequencies=np.array([2000.0, 4000.0, 4700.0]),
    )
    app = MoltuiApp(
        molecule=molecule,
        filepath="sample.molden",
        normal_mode_data=normal_mode_data,
    )

    async with app.run_test() as pilot:
        app.action_toggle_normal_mode_panel()
        await pilot.pause()
        assert app._is_playing

    # Simulate a late timer callback that fires after widgets are unmounted.
    app._animation_tick()


@pytest.mark.asyncio
async def test_brackets_and_space_control_multixyz_frames_in_geometry_mode() -> None:
    _install_skimage_stub()

    from textual.coordinate import Coordinate
    from textual.widgets import DataTable

    from moltui.app import MoltuiApp, TrajectoryData
    from moltui.elements import Atom, Molecule, get_element
    from moltui.geometry_panel import GeometryPanel

    atoms = [
        Atom(get_element("H"), np.array([0.0, 0.0, 0.0])),
        Atom(get_element("H"), np.array([0.7, 0.0, 0.0])),
    ]
    molecule = Molecule(atoms=atoms, bonds=[])
    molecule.detect_bonds()

    frames = np.array(
        [
            [[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]],
            [[0.1, 0.0, 0.0], [0.9, 0.0, 0.0]],
            [[0.2, 0.0, 0.0], [1.1, 0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    trajectory_data = TrajectoryData(frames=frames)

    app = MoltuiApp(
        molecule=molecule,
        filepath="traj.xyz",
        trajectory_data=trajectory_data,
    )

    async with app.run_test() as pilot:
        await pilot.pause()
        app._stop_playback()
        assert app.trajectory_data is not None
        app.trajectory_data.frame_index = 0
        app._apply_active_animation_geometry()
        await pilot.pause()

        panel = app.query_one(GeometryPanel)
        bond_table = panel.query_one("#bonds-table", DataTable)
        initial_length = bond_table.get_cell_at(Coordinate(0, 2))

        assert app.check_action("next_animation_step", tuple()) is True
        assert app.check_action("prev_animation_step", tuple()) is True
        assert app.check_action("toggle_playback", tuple()) is True

        await pilot.press("]")
        await pilot.pause()
        assert app.trajectory_data is not None
        assert app.trajectory_data.frame_index == 1
        updated_length = bond_table.get_cell_at(Coordinate(0, 2))
        assert updated_length != initial_length
        assert updated_length == "0.8000"

        await pilot.press("[")
        await pilot.pause()
        assert app.trajectory_data.frame_index == 0

        await pilot.press("space")
        await pilot.pause()
        assert app._is_playing

        await pilot.press("space")
        await pilot.pause()
        assert not app._is_playing


@pytest.mark.asyncio
async def test_multixyz_recomputes_bonds_and_breaks_on_dissociation() -> None:
    _install_skimage_stub()

    from textual.widgets import DataTable

    from moltui.app import MoltuiApp, TrajectoryData
    from moltui.elements import Atom, Molecule, get_element
    from moltui.geometry_panel import GeometryPanel

    atoms = [
        Atom(get_element("H"), np.array([0.0, 0.0, 0.0])),
        Atom(get_element("H"), np.array([0.74, 0.0, 0.0])),
    ]
    molecule = Molecule(atoms=atoms, bonds=[])
    molecule.detect_bonds()
    assert len(molecule.bonds) == 1

    # H-H bond threshold with current settings is ~0.806 A (0.31+0.31)*1.3.
    frames = np.array(
        [
            [[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.20, 0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    trajectory_data = TrajectoryData(frames=frames)

    app = MoltuiApp(
        molecule=molecule,
        filepath="dissociation.xyz",
        trajectory_data=trajectory_data,
    )

    async with app.run_test() as pilot:
        await pilot.pause()
        app._stop_playback()
        assert app.trajectory_data is not None
        app.trajectory_data.frame_index = 0
        app._apply_active_animation_geometry()
        await pilot.pause()

        panel = app.query_one(GeometryPanel)
        bond_table = panel.query_one("#bonds-table", DataTable)
        assert bond_table.row_count == 1

        await pilot.press("]")
        await pilot.pause()

        assert app.trajectory_data.frame_index == 1
        assert len(app.molecule.bonds) == 0
        assert bond_table.row_count == 0


@pytest.mark.asyncio
async def test_sidebar_table_has_initial_focus_on_open() -> None:
    _install_skimage_stub()

    from moltui.app import MoleculeView, MoltuiApp
    from moltui.elements import Atom, Molecule, get_element

    atoms = [
        Atom(get_element("O"), np.array([0.0, 0.0, 0.0])),
        Atom(get_element("H"), np.array([0.9, 0.0, 0.0])),
        Atom(get_element("H"), np.array([-0.3, 0.8, 0.0])),
    ]
    molecule = Molecule(atoms=atoms, bonds=[])
    molecule.detect_bonds()
    app = MoltuiApp(molecule=molecule, filepath="sample.xyz")

    async with app.run_test() as pilot:
        await pilot.pause()
        table = app._active_panel_table()
        assert table is not None
        assert table.has_focus
        assert not app.query_one(MoleculeView).has_focus
