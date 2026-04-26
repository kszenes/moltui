#!/usr/bin/env python3
"""Regression tests for half-page and boundary DataTable navigation keys."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest


def _install_skimage_stub() -> None:
    """Provide a lightweight skimage stub so app imports in test envs."""
    if "skimage.measure" in sys.modules:
        return
    skimage = types.ModuleType("skimage")
    measure = types.ModuleType("skimage.measure")

    def _marching_cubes(*_args, **_kwargs):  # pragma: no cover - never used in this test
        raise RuntimeError("marching_cubes stub called unexpectedly")

    measure.marching_cubes = _marching_cubes
    skimage.measure = measure
    sys.modules["skimage"] = skimage
    sys.modules["skimage.measure"] = measure


def _half_page_rows(row_height: int) -> int:
    viewport_rows = max(1, row_height - 1)
    return max(1, viewport_rows // 2)


def _chain_molecule(length: int):
    from moltui.elements import Atom, Molecule, get_element

    atoms = [
        Atom(get_element("C"), np.array([float(i) * 1.3, 0.0, 0.0], dtype=np.float64))
        for i in range(length)
    ]
    molecule = Molecule(atoms=atoms, bonds=[])
    molecule.detect_bonds()
    return molecule


def _row_key_atoms(table, row: int) -> set[int]:
    row_key = list(table.rows.keys())[row]
    assert row_key.value is not None
    return {int(idx) for idx in row_key.value.split("-")}


@pytest.mark.asyncio
async def test_geometry_table_half_page_and_boundary_navigation() -> None:
    _install_skimage_stub()

    from textual.widgets import DataTable

    from moltui.app import MoleculeView, MoltuiApp
    from moltui.geometry_panel import GeometryPanel

    app = MoltuiApp(molecule=_chain_molecule(32), filepath="sample.xyz")

    async with app.run_test() as pilot:
        await pilot.pause()
        panel = app.query_one(GeometryPanel)
        table = panel.query_one("#bonds-table", DataTable)
        view = app.query_one(MoleculeView)

        step = _half_page_rows(table.size.height)
        start = table.cursor_row

        await pilot.press("d")
        await pilot.pause()
        expected = min(table.row_count - 1, start + step)
        assert table.cursor_row == expected
        assert view.highlighted_atoms == _row_key_atoms(table, expected)

        await pilot.press("ctrl+u")
        await pilot.pause()
        expected = max(0, expected - step)
        assert table.cursor_row == expected
        assert view.highlighted_atoms == _row_key_atoms(table, expected)

        await pilot.press("G")
        await pilot.pause()
        assert table.cursor_row == table.row_count - 1
        assert view.highlighted_atoms == _row_key_atoms(table, table.row_count - 1)

        await pilot.press("g")
        await pilot.pause()
        assert table.cursor_row == 0
        assert view.highlighted_atoms == _row_key_atoms(table, 0)


@pytest.mark.asyncio
async def test_mo_table_half_page_and_boundary_navigation() -> None:
    _install_skimage_stub()

    from textual.widgets import DataTable

    from moltui.app import MoltuiApp
    from moltui.mo_panel import MOPanel

    molecule = _chain_molecule(3)
    n_mos = 40
    orbital_data = types.SimpleNamespace(
        molecule=molecule,
        mo_energies=np.linspace(-1.0, 1.0, n_mos, dtype=np.float64),
        mo_occupations=np.array(([2.0] * 20) + ([0.0] * 20), dtype=np.float64),
        mo_symmetries=["A1"] * n_mos,
        mo_spins=["Alpha"] * n_mos,
        n_mos=n_mos,
        homo_idx=19,
        has_mo_energies=True,
        has_mo_occupations=True,
    )
    app = MoltuiApp(
        molecule=molecule,
        filepath="sample.molden",
        orbital_data=orbital_data,
        current_mo=orbital_data.homo_idx,
    )
    app._debounced_switch_mo = lambda: None

    async with app.run_test() as pilot:
        await pilot.pause()
        app.action_toggle_mo_panel()
        await pilot.pause()

        panel = app.query_one(MOPanel)
        table = panel.query_one(DataTable)
        step = _half_page_rows(table.size.height)
        start = table.cursor_row

        await pilot.press("ctrl+d")
        await pilot.pause()
        expected = min(table.row_count - 1, start + step)
        assert table.cursor_row == expected
        assert app.current_mo == int(list(table.rows.keys())[expected].value)

        await pilot.press("u")
        await pilot.pause()
        expected = max(0, expected - step)
        assert table.cursor_row == expected
        assert app.current_mo == int(list(table.rows.keys())[expected].value)

        await pilot.press("G")
        await pilot.pause()
        assert table.cursor_row == table.row_count - 1
        assert app.current_mo == int(list(table.rows.keys())[table.row_count - 1].value)

        await pilot.press("g")
        await pilot.pause()
        assert table.cursor_row == 0
        assert app.current_mo == int(list(table.rows.keys())[0].value)


@pytest.mark.asyncio
async def test_normal_mode_table_half_page_and_boundary_navigation() -> None:
    _install_skimage_stub()

    from textual.widgets import DataTable

    from moltui.app import MoltuiApp, NormalModeData
    from moltui.normal_mode_panel import NormalModePanel

    molecule = _chain_molecule(4)
    n_modes = 36
    normal_mode_data = NormalModeData(
        equilibrium_coords=np.array(
            [atom.position.copy() for atom in molecule.atoms],
            dtype=np.float64,
        ),
        mode_vectors=np.zeros((n_modes, len(molecule.atoms), 3), dtype=np.float64),
        frequencies=np.linspace(100.0, 2400.0, n_modes, dtype=np.float64),
    )
    app = MoltuiApp(
        molecule=molecule,
        filepath="sample.molden",
        normal_mode_data=normal_mode_data,
    )

    async with app.run_test() as pilot:
        await pilot.pause()
        app.action_toggle_normal_mode_panel()
        await pilot.pause()

        panel = app.query_one(NormalModePanel)
        table = panel.query_one(DataTable)
        step = _half_page_rows(table.size.height)
        start = table.cursor_row

        await pilot.press("d")
        await pilot.pause()
        expected = min(table.row_count - 1, start + step)
        assert table.cursor_row == expected
        assert app.normal_mode_data is not None
        assert app.normal_mode_data.mode_index == int(list(table.rows.keys())[expected].value)

        await pilot.press("ctrl+u")
        await pilot.pause()
        expected = max(0, expected - step)
        assert table.cursor_row == expected
        assert app.normal_mode_data.mode_index == int(list(table.rows.keys())[expected].value)

        await pilot.press("G")
        await pilot.pause()
        assert table.cursor_row == table.row_count - 1
        assert app.normal_mode_data.mode_index == int(
            list(table.rows.keys())[table.row_count - 1].value
        )

        await pilot.press("g")
        await pilot.pause()
        assert table.cursor_row == 0
        assert app.normal_mode_data.mode_index == int(list(table.rows.keys())[0].value)
