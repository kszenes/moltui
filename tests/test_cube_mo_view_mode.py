#!/usr/bin/env python3
"""Regression tests for cube-file MO/geometry view cycling."""

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

    def _marching_cubes(*_args, **_kwargs):  # pragma: no cover - never used in this test
        raise RuntimeError("marching_cubes stub called unexpectedly")

    measure.marching_cubes = _marching_cubes
    skimage.measure = measure
    sys.modules["skimage"] = skimage
    sys.modules["skimage.measure"] = measure


@pytest.mark.asyncio
async def test_cube_starts_in_mo_view_without_sidebar_then_cycles_to_geometry() -> None:
    _install_skimage_stub()

    from moltui.app import MoleculeView, MoltuiApp
    from moltui.elements import Atom, Molecule, get_element
    from moltui.geometry_panel import GeometryPanel
    from moltui.mo_panel import MOPanel

    atoms = [
        Atom(get_element("O"), np.array([0.0, 0.0, 0.0])),
        Atom(get_element("H"), np.array([0.9, 0.0, 0.0])),
        Atom(get_element("H"), np.array([-0.3, 0.8, 0.0])),
    ]
    molecule = Molecule(atoms=atoms, bonds=[])
    molecule.detect_bonds()

    app = MoltuiApp(molecule=molecule, filepath="sample.cube", isosurfaces=[])
    app._cube_data = SimpleNamespace()

    async with app.run_test() as pilot:
        await pilot.pause()
        view = app.query_one(MoleculeView)
        mo_panel = app.query_one(MOPanel)
        geom_panel = app.query_one(GeometryPanel)

        assert view.show_orbitals
        assert not mo_panel.has_class("visible")
        assert not geom_panel.has_class("visible")

        await pilot.press("m")
        await pilot.pause()

        assert not view.show_orbitals
        assert not mo_panel.has_class("visible")
        assert geom_panel.has_class("visible")
