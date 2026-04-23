"""Optional TREXIO file format support (HDF5 or text backend)."""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from .elements import Atom, Element, Molecule, get_element, get_element_by_number
from .parsers import BOHR_TO_ANGSTROM

_HDF5_TREXIO_SUFFIXES = frozenset({".h5", ".hdf5"})


def _hdf5_trexio_candidate(path: Path) -> bool:
    """True for regular files we classify as TREXIO solely by ``.h5`` / ``.hdf5`` extension."""
    return path.is_file() and path.suffix.lower() in _HDF5_TREXIO_SUFFIXES


def _not_trexio_hdf5_message(path: Path) -> str:
    return (
        "Only HDF5 files of TREXIO format are currently supported. "
        f"{path.name} is not a valid TREXIO file."
    )


def _raise_if_invalid_hdf5_trexio(path: Path) -> None:
    if _hdf5_trexio_candidate(path):
        raise ValueError(_not_trexio_hdf5_message(path))


@contextlib.contextmanager
def _suppress_stderr_fd() -> Iterator[None]:
    """Redirect the process stderr fd (2) so C libraries (e.g. HDF5) cannot spam diagnostics."""
    stderr_fd = 2
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
    except OSError:
        yield
        return
    saved = os.dup(stderr_fd)
    try:
        os.dup2(devnull, stderr_fd)
        yield
    finally:
        os.dup2(saved, stderr_fd)
        os.close(saved)
        os.close(devnull)


def is_readable_trexio_text_directory(path: Path) -> bool:
    """True if ``path`` is a directory readable as TREXIO text with nucleus coordinates."""
    if not path.is_dir():
        return False
    try:
        import trexio
    except ImportError:
        return False
    try:
        with trexio.File(str(path), "r", back_end=trexio.TREXIO_TEXT) as f:
            return bool(trexio.has_nucleus_num(f) and trexio.has_nucleus_coord(f))
    except trexio.Error:
        return False


def is_trexio_path(path: Path) -> bool:
    """Return True if ``path`` should be opened as TREXIO (HDF5 file or text-backend directory)."""
    suffix = path.suffix.lower()
    if suffix in (".h5", ".hdf5") and path.is_file():
        return True
    if suffix == ".trexio" and path.is_file():
        return True
    if suffix == ".trexio" and path.is_dir():
        return True
    if path.is_dir():
        return is_readable_trexio_text_directory(path)
    return False


def trexio_backend_for_path(path: Path) -> int:
    """HDF5 vs text backend constant for paths accepted by :func:`is_trexio_path`."""
    import trexio

    if path.is_dir():
        return trexio.TREXIO_TEXT
    return trexio.TREXIO_HDF5


def read_trexio_nucleus_elements(f, trexio, n: int, *, path: Path | None = None) -> list[Element]:
    """Map ``nucleus.label`` or ``nucleus.charge`` to ``n`` :class:`Element` instances.

    ``n`` must be ``nucleus.num`` as already read by the caller. When ``path`` is
    set and the file is HDF5-classified, invalid-data errors use the same generic
    message as :func:`load_molecule_from_trexio` before a specific :exc:`ValueError`.
    """
    if trexio.has_nucleus_label(f):
        labels = trexio.read_nucleus_label(f)
        if len(labels) != n:
            if path is not None:
                _raise_if_invalid_hdf5_trexio(path)
            raise ValueError("TREXIO nucleus.label length does not match nucleus.num.")
        return [get_element(str(lbl)) for lbl in labels]
    if trexio.has_nucleus_charge(f):
        charges = np.asarray(trexio.read_nucleus_charge(f), dtype=np.float64)
        if charges.shape[0] != n:
            if path is not None:
                _raise_if_invalid_hdf5_trexio(path)
            raise ValueError("TREXIO nucleus.charge length does not match nucleus.num.")
        return [get_element_by_number(int(round(float(z)))) for z in charges]
    if path is not None:
        _raise_if_invalid_hdf5_trexio(path)
    raise ValueError(
        "TREXIO file has neither nucleus.label nor nucleus.charge; cannot assign elements."
    )


def load_molecule_from_trexio(filepath: str | Path) -> Molecule:
    """Load nuclear coordinates from a TREXIO file (HDF5) or text-backend directory."""
    try:
        import trexio
    except ImportError as exc:
        raise ValueError(
            'Opening TREXIO files requires the optional "trexio" dependency '
            '(e.g. pip install "moltui[trexio]" or uv sync --extra trexio).'
        ) from exc

    path = Path(filepath)
    if not is_trexio_path(path):
        raise ValueError(f"Not a recognized TREXIO path: {path}")

    backend = trexio_backend_for_path(path)
    _stderr_guard = _suppress_stderr_fd() if backend == trexio.TREXIO_HDF5 else nullcontext()
    try:
        with _stderr_guard:
            with trexio.File(str(path), "r", back_end=backend) as f:
                if not trexio.has_nucleus_num(f) or not trexio.has_nucleus_coord(f):
                    _raise_if_invalid_hdf5_trexio(path)
                    raise ValueError("TREXIO file is missing nucleus.num or nucleus.coord data.")
                n = int(trexio.read_nucleus_num(f))
                coords_bohr = np.asarray(trexio.read_nucleus_coord(f), dtype=np.float64)
                if coords_bohr.shape != (n, 3):
                    _raise_if_invalid_hdf5_trexio(path)
                    raise ValueError(
                        f"TREXIO nucleus.coord has shape {coords_bohr.shape}, expected ({n}, 3)."
                    )
                coords_ang = coords_bohr * BOHR_TO_ANGSTROM

                elements = read_trexio_nucleus_elements(f, trexio, n, path=path)
    except trexio.Error as exc:
        if _hdf5_trexio_candidate(path):
            raise ValueError(_not_trexio_hdf5_message(path)) from exc
        raise ValueError(f"Invalid or unreadable TREXIO file: {exc}") from exc

    atoms = [Atom(element=elements[i], position=coords_ang[i].copy()) for i in range(n)]
    mol = Molecule(atoms=atoms, bonds=[])
    mol.detect_bonds()
    return mol
