#!/usr/bin/env python3
"""Tests for image_renderer.py: projection, render_scene, and export pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from moltui.elements import Atom, Molecule, get_element
from moltui.image_renderer import ImageRenderer, render_scene, rotation_matrix

# --- rotation_matrix ---


class TestRotationMatrix:
    def test_identity_at_zero(self):
        R = rotation_matrix(0, 0, 0)
        np.testing.assert_array_almost_equal(R, np.eye(3))

    def test_orthogonal(self):
        R = rotation_matrix(0.3, -0.5, 0.7)
        np.testing.assert_array_almost_equal(R @ R.T, np.eye(3), decimal=10)

    def test_determinant_is_one(self):
        R = rotation_matrix(1.2, -0.3, 2.1)
        assert np.linalg.det(R) == pytest.approx(1.0)


# --- ImageRenderer projection ---


class TestProjection:
    def test_center_projects_to_center(self):
        r = ImageRenderer(100, 100)
        sx, sy, sz = r._project(np.array([0.0, 0.0, 5.0]))
        assert sx == pytest.approx(50.0)
        assert sy == pytest.approx(50.0)

    def test_behind_camera_returns_nan(self):
        r = ImageRenderer(100, 100)
        sx, sy, sz = r._project(np.array([0.0, 0.0, -1.0]))
        assert np.isnan(sx)
        assert np.isnan(sy)

    def test_right_projects_right(self):
        r = ImageRenderer(100, 100)
        sx_center, _, _ = r._project(np.array([0.0, 0.0, 5.0]))
        sx_right, _, _ = r._project(np.array([1.0, 0.0, 5.0]))
        assert sx_right > sx_center

    def test_up_projects_up(self):
        r = ImageRenderer(100, 100)
        _, sy_center, _ = r._project(np.array([0.0, 0.0, 5.0]))
        _, sy_up, _ = r._project(np.array([0.0, 1.0, 5.0]))
        assert sy_up < sy_center  # screen y is inverted


# --- render_scene ---


def _h2_molecule() -> Molecule:
    H = get_element("H")
    atoms = [
        Atom(H, np.array([0.0, 0.0, 0.0])),
        Atom(H, np.array([0.74, 0.0, 0.0])),
    ]
    mol = Molecule(atoms=atoms, bonds=[])
    mol.detect_bonds()
    return mol


class TestRenderScene:
    def test_output_shape(self):
        mol = _h2_molecule()
        rot = rotation_matrix(0, 0, 0)
        pixels, hit = render_scene(64, 48, mol, rot, 5.0, ssaa=1)
        assert pixels.shape == (48, 64, 3)
        assert hit.shape == (48, 48) or hit.shape == (48, 64)

    def test_output_shape_exact(self):
        mol = _h2_molecule()
        rot = rotation_matrix(0, 0, 0)
        pixels, hit = render_scene(64, 48, mol, rot, 5.0, ssaa=1)
        assert pixels.shape == (48, 64, 3)
        assert hit.shape == (48, 64)

    def test_ssaa_downsamples(self):
        mol = _h2_molecule()
        rot = rotation_matrix(0, 0, 0)
        pixels, hit = render_scene(64, 48, mol, rot, 5.0, ssaa=2)
        assert pixels.shape == (48, 64, 3)

    def test_some_pixels_hit(self):
        mol = _h2_molecule()
        rot = rotation_matrix(0, 0, 0)
        _, hit = render_scene(64, 48, mol, rot, 5.0, ssaa=1)
        assert hit.any()

    def test_not_all_pixels_hit(self):
        mol = _h2_molecule()
        rot = rotation_matrix(0, 0, 0)
        _, hit = render_scene(64, 48, mol, rot, 5.0, ssaa=1)
        assert not hit.all()

    def test_bg_color_on_empty_pixels(self):
        mol = _h2_molecule()
        rot = rotation_matrix(0, 0, 0)
        bg = (30, 40, 50)
        pixels, hit = render_scene(64, 48, mol, rot, 5.0, ssaa=1, bg_color=bg)
        bg_pixels = pixels[~hit]
        if len(bg_pixels) > 0:
            np.testing.assert_array_equal(bg_pixels[0], list(bg))

    def test_lighting_params_accepted(self):
        mol = _h2_molecule()
        rot = rotation_matrix(0, 0, 0)
        pixels, _ = render_scene(
            32,
            24,
            mol,
            rot,
            5.0,
            ssaa=1,
            ambient=0.5,
            diffuse=0.8,
            specular=0.2,
            shininess=64.0,
        )
        assert pixels.shape == (24, 32, 3)

    def test_licorice_mode(self):
        mol = _h2_molecule()
        rot = rotation_matrix(0, 0, 0)
        pixels, hit = render_scene(64, 48, mol, rot, 5.0, ssaa=1, licorice=True)
        assert hit.any()

    def test_atom_scale_and_bond_radius(self):
        mol = _h2_molecule()
        rot = rotation_matrix(0, 0, 0)
        pixels, hit = render_scene(
            64,
            48,
            mol,
            rot,
            5.0,
            ssaa=1,
            atom_scale=0.8,
            bond_radius=0.15,
        )
        assert hit.any()


# --- Smoke test: render all example XYZ files ---

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
XYZ_FILES = sorted(EXAMPLES_DIR.glob("*.xyz"))


@pytest.mark.parametrize("xyz_file", XYZ_FILES, ids=lambda p: p.name)
def test_render_xyz_smoke(xyz_file: Path):
    from moltui.parsers import parse_xyz

    mol = parse_xyz(xyz_file)
    rot = rotation_matrix(-0.2, -0.5, 0.0)
    cam = max(4.0, mol.radius() * 3.0)
    pixels, hit = render_scene(32, 24, mol, rot, cam, ssaa=1)
    assert pixels.shape == (24, 32, 3)
    assert hit.any()
