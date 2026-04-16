#!/usr/bin/env python3
"""Tests for isosurface.py: marching cubes extraction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from moltui.isosurface import IsosurfaceMesh, extract_isosurfaces
from moltui.parsers import parse_cube_data

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
CUBE_FILES = sorted(EXAMPLES_DIR.glob("*.cube"))


@pytest.fixture
def cube_data():
    if not CUBE_FILES:
        pytest.skip("No .cube files in examples/")
    return parse_cube_data(CUBE_FILES[0])


class TestExtractIsosurfaces:
    def test_returns_list_of_meshes(self, cube_data):
        meshes = extract_isosurfaces(cube_data)
        assert isinstance(meshes, list)
        for m in meshes:
            assert isinstance(m, IsosurfaceMesh)

    def test_mesh_has_vertices_and_faces(self, cube_data):
        meshes = extract_isosurfaces(cube_data)
        assert len(meshes) > 0
        for m in meshes:
            assert m.vertices.ndim == 2 and m.vertices.shape[1] == 3
            assert m.faces.ndim == 2 and m.faces.shape[1] == 3
            assert m.normals.shape == m.vertices.shape

    def test_normals_are_unit_length(self, cube_data):
        meshes = extract_isosurfaces(cube_data)
        for m in meshes:
            norms = np.linalg.norm(m.normals, axis=1)
            np.testing.assert_array_almost_equal(norms, 1.0, decimal=3)

    def test_high_isovalue_produces_fewer_or_no_meshes(self, cube_data):
        meshes_low = extract_isosurfaces(cube_data, isovalue=0.01)
        meshes_high = extract_isosurfaces(cube_data, isovalue=0.5)
        total_verts_low = sum(len(m.vertices) for m in meshes_low)
        total_verts_high = sum(len(m.vertices) for m in meshes_high)
        assert total_verts_high <= total_verts_low

    def test_colors_are_positive_negative(self, cube_data):
        meshes = extract_isosurfaces(cube_data)
        if len(meshes) == 2:
            colors = {m.color for m in meshes}
            from moltui.isosurface import COLOR_NEGATIVE, COLOR_POSITIVE

            assert COLOR_POSITIVE in colors
            assert COLOR_NEGATIVE in colors


@pytest.mark.parametrize("cube_file", CUBE_FILES, ids=lambda p: p.name)
def test_extract_smoke(cube_file: Path):
    cd = parse_cube_data(cube_file)
    meshes = extract_isosurfaces(cd)
    assert isinstance(meshes, list)
