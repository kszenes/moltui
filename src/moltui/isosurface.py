from dataclasses import dataclass

import numpy as np
from scipy.ndimage import zoom
from skimage.measure import marching_cubes

from .parsers import CubeData, VolumetricData

COLOR_POSITIVE = (255, 135, 0)  # orange
COLOR_NEGATIVE = (30, 100, 255)  # blue


@dataclass
class IsosurfaceMesh:
    vertices: np.ndarray  # (N, 3) in Angstrom
    faces: np.ndarray  # (M, 3) triangle indices
    normals: np.ndarray  # (N, 3) vertex normals
    color: tuple[int, int, int]


def extract_isosurfaces(
    cube_data: CubeData | VolumetricData,
    isovalue: float = 0.05,
    step: int = 1,
    upsample: int = 1,
) -> list[IsosurfaceMesh]:
    volume = cube_data.to_volumetric_data() if isinstance(cube_data, CubeData) else cube_data
    data = volume.data[::step, ::step, ::step]
    axes = volume.axes * step
    if upsample > 1:
        data = zoom(data, upsample, order=3)
        axes = axes / upsample

    meshes = []
    for level, color in [(isovalue, COLOR_POSITIVE), (-isovalue, COLOR_NEGATIVE)]:
        if data.max() < level or data.min() > level:
            continue
        verts, faces, normals, _ = marching_cubes(data, level)
        world_verts = volume.origin + verts @ axes
        normals = normals @ np.linalg.pinv(axes)
        # Normalize normals
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        normals = normals / norms
        # marching_cubes normals are the gradient (toward increasing values).
        # For positive isovalue the gradient points inward (into the lobe),
        # so flip normals outward and reverse face winding to match.
        if level > 0:
            normals = -normals
            faces = faces[:, ::-1]
        meshes.append(IsosurfaceMesh(world_verts, faces, normals, color))

    return meshes
