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

    # 3 atoms, non-linear -> first vibrational mode index starts at 6.
    n_modes = 9
    mode_vectors = np.zeros((n_modes, len(atoms), 3), dtype=np.float64)
    for i in range(n_modes):
        mode_vectors[i, :, 0] = 0.01 * (i + 1)

    normal_mode_data = NormalModeData(
        equilibrium_coords=np.array([a.position.copy() for a in atoms]),
        mode_vectors=mode_vectors,
        frequencies=np.arange(float(n_modes)),
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
        assert app.normal_mode_data.mode_index == 6
        assert app._is_playing

        app.action_panel_next()
        await pilot.pause()
        assert app.normal_mode_data.mode_index == 7

        app.action_panel_prev()
        await pilot.pause()
        assert app.normal_mode_data.mode_index == 6


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

    molden_data = SimpleNamespace(
        mo_energies=np.array([-0.5]),
        mo_occupations=np.array([2.0]),
        mo_symmetries=["A1"],
        mo_spins=["Alpha"],
        n_mos=1,
        homo_idx=0,
    )

    app = MoltuiApp(
        molecule=molecule,
        filepath="sample.molden",
        molden_data=molden_data,
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

    molden_data = SimpleNamespace(
        mo_energies=np.array([-0.5]),
        mo_occupations=np.array([2.0]),
        mo_symmetries=["A1"],
        mo_spins=["Alpha"],
        n_mos=1,
        homo_idx=0,
    )

    app = MoltuiApp(
        molecule=molecule,
        filepath="sample.molden",
        molden_data=molden_data,
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
    molden_data = SimpleNamespace(
        mo_energies=np.array([-0.5]),
        mo_occupations=np.array([2.0]),
        mo_symmetries=["A1"],
        mo_spins=["Alpha"],
        n_mos=1,
        homo_idx=0,
    )

    app = MoltuiApp(
        molecule=molecule,
        filepath="sample.molden",
        molden_data=molden_data,
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
    molden_data = SimpleNamespace(
        mo_energies=np.array([-0.5]),
        mo_occupations=np.array([2.0]),
        mo_symmetries=["A1"],
        mo_spins=["Alpha"],
        n_mos=1,
        homo_idx=0,
    )
    normal_mode_data = NormalModeData(
        equilibrium_coords=np.array([a.position.copy() for a in atoms]),
        mode_vectors=np.zeros((3, len(atoms), 3), dtype=np.float64),
        frequencies=np.array([2000.0, 4000.0, 4700.0]),
    )
    app = MoltuiApp(
        molecule=molecule,
        filepath="sample.molden",
        molden_data=molden_data,
        current_mo=0,
        normal_mode_data=normal_mode_data,
    )

    async with app.run_test() as pilot:
        # Starts in MO mode
        assert app.check_action("toggle_playback", tuple()) is False
        assert app.check_action("next_mode", tuple()) is False

        await pilot.press("m")  # move to normal mode
        await pilot.pause()
        assert app.check_action("next_mo", tuple()) is False
