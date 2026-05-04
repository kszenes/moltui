from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .elements import Molecule

BOHR_TO_ANGSTROM = 0.529177249


@dataclass
class VolumetricData:
    molecule: Molecule
    origin: np.ndarray  # (3,) in Angstrom
    axes: np.ndarray  # (3, 3) grid step vectors in Angstrom
    n_points: tuple[int, int, int]
    data: np.ndarray  # (n1, n2, n3) volumetric data
    periodic: bool | tuple[bool, bool, bool] = False


@dataclass
class CubeData:
    molecule: Molecule
    origin: np.ndarray  # (3,) in Bohr
    axes: np.ndarray  # (3, 3) step vectors in Bohr
    n_points: tuple[int, int, int]
    data: np.ndarray  # (n1, n2, n3) volumetric data
    periodic: bool | tuple[bool, bool, bool] = False

    def to_volumetric_data(self) -> VolumetricData:
        return VolumetricData(
            molecule=self.molecule,
            origin=self.origin * BOHR_TO_ANGSTROM,
            axes=self.axes * BOHR_TO_ANGSTROM,
            n_points=self.n_points,
            data=self.data,
            periodic=self.periodic,
        )


def default_volume_isovalue(volume: CubeData | VolumetricData) -> float:
    data = volume.data
    data_min = float(np.nanmin(data))
    data_max = float(np.nanmax(data))
    if data_min <= 0.05 <= data_max:
        return 0.05
    if data_min >= 0.0:
        return 0.5 * (data_min + data_max)
    return 0.25 * max(abs(data_min), abs(data_max))


def volume_isovalue_range(volume: CubeData | VolumetricData | None) -> tuple[float, float, float]:
    if volume is None:
        return (0.001, 0.10, 0.005)
    data = volume.data
    data_min = float(np.nanmin(data))
    data_max = float(np.nanmax(data))
    if data_min >= 0.0:
        span = max(data_max - data_min, 1e-6)
        return (data_min, data_max, span / 100.0)
    if data_max <= 0.0:
        min_abs = abs(data_max)
        max_abs = abs(data_min)
        span = max(max_abs - min_abs, 1e-6)
        return (min_abs, max_abs, span / 100.0)
    max_abs = max(abs(data_min), abs(data_max), 0.1)
    return (0.001, max_abs, max_abs / 100.0)
