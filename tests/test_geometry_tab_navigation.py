#!/usr/bin/env python3
"""Regression tests for GeometryPanel tab navigation through app-level Tab handling."""

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


@pytest.mark.asyncio
async def test_tab_and_shift_tab_switch_geometry_tabs() -> None:
    _install_skimage_stub()

    from textual.widgets import TabbedContent

    from moltui.app import MoleculeView, MoltuiApp
    from moltui.elements import Atom, Molecule, get_element
    from moltui.geometry_panel import GeometryPanel

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
        panel = app.query_one(GeometryPanel)
        tabs = panel.query_one(TabbedContent)
        assert tabs.active == "tab-atoms"
        atoms = panel.query_one("#atoms-table")
        assert atoms.row_count == 3
        assert [key.value for key in atoms.rows] == ["0", "1", "2"]
        assert atoms.get_row_at(1) == ["2:H", " 0.9000", " 0.0000", " 0.0000"]
        assert atoms.get_row_at(2) == ["3:H", "-0.3000", " 0.8000", " 0.0000"]
        assert app.query_one(MoleculeView).highlighted_atoms == {0}

        await pilot.press("n")
        await pilot.pause()
        assert app.query_one(MoleculeView).highlighted_atoms == {1}

        await pilot.press("tab")
        await pilot.pause()
        assert tabs.active == "tab-bonds"

        await pilot.press("shift+tab")
        await pilot.pause()
        assert tabs.active == "tab-atoms"
