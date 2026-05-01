#!/usr/bin/env python3
"""Regression test for initial focus when opening the visual panel."""

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
async def test_opening_visual_panel_focuses_atom_scale_when_no_contextual_sliders() -> None:
    _install_skimage_stub()

    from moltui.app import MoltuiApp
    from moltui.elements import Atom, Molecule, get_element
    from moltui.visual_panel import Slider, VisualPanel

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
        visual_panel = app.query_one(VisualPanel)
        assert not visual_panel.has_class("visible")

        await pilot.press("V")
        await pilot.pause()

        assert visual_panel.has_class("visible")
        assert visual_panel.query_one("#slider-atom-scale", Slider).has_focus


@pytest.mark.asyncio
async def test_opening_visual_panel_focuses_show_box_for_periodic_molecule() -> None:
    _install_skimage_stub()

    from moltui.app import MoltuiApp
    from moltui.elements import Atom, Molecule, get_element
    from moltui.visual_panel import Toggle, VisualPanel

    H = get_element("H")
    molecule = Molecule(
        atoms=[
            Atom(H, np.array([0.0, 0.0, 0.0])),
            Atom(H, np.array([1.9, 0.0, 0.0])),
        ],
        bonds=[],
        lattice=np.diag([2.0, 2.0, 2.0]),
    )
    molecule.detect_bonds_periodic()
    app = MoltuiApp(molecule=molecule, filepath="sample.extxyz")

    async with app.run_test() as pilot:
        await pilot.pause()
        visual_panel = app.query_one(VisualPanel)

        await pilot.press("V")
        await pilot.pause()

        show_box = visual_panel.query_one("#checkbox-show-cell", Toggle)
        assert visual_panel.has_class("visible")
        assert show_box.has_focus


@pytest.mark.asyncio
async def test_opening_visual_panel_focuses_isovalue_when_visible() -> None:
    _install_skimage_stub()

    from moltui.app import MoltuiApp
    from moltui.elements import Atom, Molecule, get_element
    from moltui.isosurface import IsosurfaceMesh
    from moltui.visual_panel import Slider, VisualPanel

    atoms = [
        Atom(get_element("O"), np.array([0.0, 0.0, 0.0])),
        Atom(get_element("H"), np.array([0.9, 0.0, 0.0])),
        Atom(get_element("H"), np.array([-0.3, 0.8, 0.0])),
    ]
    molecule = Molecule(atoms=atoms, bonds=[])
    molecule.detect_bonds()
    app = MoltuiApp(molecule=molecule, filepath="sample.xyz")
    app._isosurfaces = [
        IsosurfaceMesh(
            vertices=np.zeros((3, 3), dtype=np.float64),
            faces=np.zeros((1, 3), dtype=np.int32),
            normals=np.zeros((3, 3), dtype=np.float64),
            color=(255, 135, 0),
        )
    ]

    async with app.run_test() as pilot:
        await pilot.pause()
        visual_panel = app.query_one(VisualPanel)
        assert not visual_panel.has_class("visible")

        await pilot.press("V")
        await pilot.pause()

        isovalue = visual_panel.query_one("#slider-isovalue", Slider)
        assert visual_panel.has_class("visible")
        assert isovalue.has_focus


@pytest.mark.asyncio
async def test_tab_and_shift_tab_adjust_visual_slider_value() -> None:
    _install_skimage_stub()

    from moltui.app import MoltuiApp
    from moltui.elements import Atom, Molecule, get_element
    from moltui.isosurface import IsosurfaceMesh
    from moltui.visual_panel import Slider, VisualPanel

    atoms = [
        Atom(get_element("O"), np.array([0.0, 0.0, 0.0])),
        Atom(get_element("H"), np.array([0.9, 0.0, 0.0])),
        Atom(get_element("H"), np.array([-0.3, 0.8, 0.0])),
    ]
    molecule = Molecule(atoms=atoms, bonds=[])
    molecule.detect_bonds()
    app = MoltuiApp(molecule=molecule, filepath="sample.xyz")
    app._isosurfaces = [
        IsosurfaceMesh(
            vertices=np.zeros((3, 3), dtype=np.float64),
            faces=np.zeros((1, 3), dtype=np.int32),
            normals=np.zeros((3, 3), dtype=np.float64),
            color=(255, 135, 0),
        )
    ]

    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("V")
        await pilot.pause()

        visual_panel = app.query_one(VisualPanel)
        isovalue = visual_panel.query_one("#slider-isovalue", Slider)
        assert isovalue.has_focus

        start = isovalue.value
        await pilot.press("tab")
        await pilot.pause()
        assert isovalue.value > start

        await pilot.press("shift+tab")
        await pilot.pause()
        assert isovalue.value == pytest.approx(start)


@pytest.mark.asyncio
async def test_n_and_p_navigate_focus_within_visual_panel() -> None:
    _install_skimage_stub()

    from moltui.app import MoltuiApp
    from moltui.elements import Atom, Molecule, get_element
    from moltui.visual_panel import Slider, VisualPanel

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
        await pilot.press("V")
        await pilot.pause()

        visual_panel = app.query_one(VisualPanel)
        atom_scale = visual_panel.query_one("#slider-atom-scale", Slider)
        bond_radius = visual_panel.query_one("#slider-bond-radius", Slider)

        assert visual_panel.has_class("visible")
        assert atom_scale.has_focus

        await pilot.press("n")
        await pilot.pause()
        assert bond_radius.has_focus

        await pilot.press("p")
        await pilot.pause()
        assert atom_scale.has_focus
