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


def _bond_samples_in_bounds(
    renderer: ImageRenderer, p1: np.ndarray, p2: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return all in-bounds bond sample pixels and depths."""
    sx1, sy1, sz1 = renderer._project(p1)
    sx2, sy2, sz2 = renderer._project(p2)
    assert not np.isnan(sx1)
    assert not np.isnan(sx2)

    scale = min(renderer.width, renderer.height) / 2
    mid_z = (sz1 + sz2) / 2
    pr = renderer.bond_radius * renderer.fov / mid_z * scale

    dx = sx2 - sx1
    dy = sy2 - sy1
    length = np.sqrt(dx * dx + dy * dy)
    assert length >= 1.0

    nx, ny = -dy / length, dx / length
    half_w = max(1.0, pr)
    steps = int(length * 3) + 1

    hw = int(half_w + 1)
    ts = np.linspace(0, 1, steps + 1)
    offsets = np.arange(-hw, hw + 1, dtype=np.float64)
    d_norm = offsets / half_w
    off_mask = np.abs(d_norm) <= 1.0
    offsets = offsets[off_mask]
    d_norm = d_norm[off_mask]

    cxs = sx1 + dx * ts
    cys = sy1 + dy * ts
    czs = sz1 + (sz2 - sz1) * ts

    all_px = np.round(cxs[:, None] + nx * offsets[None, :]).astype(int)
    all_py = np.round(cys[:, None] + ny * offsets[None, :]).astype(int)
    cyl_nz = np.sqrt(1.0 - d_norm * d_norm)
    pz = czs[:, None] - renderer.bond_radius * cyl_nz[None, :]

    flat_px = all_px.ravel()
    flat_py = all_py.ravel()
    flat_pz = pz.ravel()

    valid = (
        (flat_px >= 0) & (flat_px < renderer.width) & (flat_py >= 0) & (flat_py < renderer.height)
    )
    return flat_px[valid], flat_py[valid], flat_pz[valid]


def _count_internal_pinholes(mask: np.ndarray) -> int:
    """Count empty pixels surrounded by mostly-filled 3x3 neighborhoods."""
    if not mask.any():
        return 0
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    sub = mask[y0 : y1 + 1, x0 : x1 + 1]
    holes = 0
    h, w = sub.shape
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if not sub[y, x] and sub[y - 1 : y + 2, x - 1 : x + 2].sum() >= 7:
                holes += 1
    return holes


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


class TestBondRasterization:
    def test_bond_keeps_nearest_depth_per_pixel_when_samples_overlap(self):
        r = ImageRenderer(120, 80)
        r.bond_radius = 0.22
        p1 = np.array([-0.8, -0.2, 4.2])
        p2 = np.array([0.9, 0.7, 5.1])

        flat_px, flat_py, flat_pz = _bond_samples_in_bounds(r, p1, p2)
        pixel_ids = flat_py * r.width + flat_px
        unique, counts = np.unique(pixel_ids, return_counts=True)
        assert (counts > 1).any(), "test setup must include duplicate pixel samples"

        expected_flat = np.full(r.width * r.height, np.inf, dtype=np.float64)
        np.minimum.at(expected_flat, pixel_ids, flat_pz)
        expected = expected_flat.reshape(r.height, r.width)

        r.render_bond(p1, p2, (255, 255, 255), (255, 255, 255))

        expected_hit = np.isfinite(expected)
        np.testing.assert_array_equal(np.isfinite(r.z_buf), expected_hit)
        np.testing.assert_allclose(r.z_buf[expected_hit], expected[expected_hit], rtol=0, atol=1e-9)

    def test_bond_hit_mask_has_no_internal_pinhole_for_steep_diagonal(self):
        r = ImageRenderer(160, 120)
        r.bond_radius = 0.08
        p1 = np.array([1.03759371, -0.78793903, 3.50547172])
        p2 = np.array([-0.30631733, 0.43301329, 5.7617411])

        r.render_bond(p1, p2, (255, 255, 255), (255, 255, 255))
        hit = np.isfinite(r.z_buf)
        assert _count_internal_pinholes(hit) == 0


# --- Smoke test: render temp XYZ files ---


def _write_xyz(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(body)
    return path


def test_render_xyz_smoke(tmp_path: Path):
    from moltui.parsers import parse_xyz

    xyz_files = [
        _write_xyz(
            tmp_path,
            "water.xyz",
            "3\nwater\nO 0.0 0.0 0.0\nH 0.7586 0.0 0.5043\nH -0.7586 0.0 0.5043\n",
        ),
        _write_xyz(
            tmp_path,
            "co2.xyz",
            "3\nco2\nO -1.16 0.0 0.0\nC 0.0 0.0 0.0\nO 1.16 0.0 0.0\n",
        ),
    ]

    for xyz_file in xyz_files:
        mol = parse_xyz(xyz_file)
        rot = rotation_matrix(-0.2, -0.5, 0.0)
        cam = max(4.0, mol.radius() * 3.0)
        pixels, hit = render_scene(32, 24, mol, rot, cam, ssaa=1)
        assert pixels.shape == (24, 32, 3)
        assert hit.any()
