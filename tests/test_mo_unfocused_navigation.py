#!/usr/bin/env python3
"""Regression test for first MO navigation with unfocused table."""

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
async def test_first_n_changes_mo_once_when_table_unfocused() -> None:
    _install_skimage_stub()

    from moltui.app import MoleculeView, MoltuiApp
    from moltui.elements import Atom, Molecule, get_element
    from moltui.mo_panel import MOPanel

    atoms = [
        Atom(get_element("O"), np.array([0.0, 0.0, 0.0])),
        Atom(get_element("H"), np.array([0.9, 0.0, 0.0])),
        Atom(get_element("H"), np.array([-0.3, 0.8, 0.0])),
    ]
    molecule = Molecule(atoms=atoms, bonds=[])
    molecule.detect_bonds()
    orbital_data = types.SimpleNamespace(
        molecule=molecule,
        mo_energies=np.array([-0.5, 0.2]),
        mo_occupations=np.array([2.0, 0.0]),
        mo_symmetries=["A1", "B2"],
        mo_spins=["Alpha", "Alpha"],
        n_mos=2,
        homo_idx=0,
        has_mo_energies=True,
        has_mo_occupations=True,
    )
    app = MoltuiApp(
        molecule=orbital_data.molecule,
        filepath="sample.molden",
        orbital_data=orbital_data,
        current_mo=orbital_data.homo_idx,
    )

    # Keep the test focused on event-routing behavior.
    app._debounced_switch_mo = lambda: None
    original_set_current_mo = app._set_current_mo
    calls: list[int] = []

    def wrapped_set_current_mo(mo_idx: int) -> None:
        calls.append(mo_idx)
        original_set_current_mo(mo_idx)

    app._set_current_mo = wrapped_set_current_mo

    async with app.run_test() as pilot:
        await pilot.pause()
        view = app.query_one(MoleculeView)
        mo_panel = app.query_one(MOPanel)
        view.focus()
        await pilot.pause()
        assert view.has_focus
        assert not mo_panel.has_focus

        start = app.current_mo
        expected = mo_panel.adjacent_mo(start, 1)
        assert expected is not None

        calls.clear()
        await pilot.press("n")
        await pilot.pause()
        await pilot.pause()

        assert app.current_mo == expected
        assert calls == [expected]
