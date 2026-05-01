#!/usr/bin/env python3
"""Cross-check moltui's CIF parser against ASE.

ASE is treated as the ground-truth reference for CIF parsing (cell parameters,
symmetry expansion, special-position dedupe). These tests are skipped when
``ase`` is not installed.

We compare structure-equivalent properties only — element counts, sorted
fractional coordinates modulo the lattice, and cell parameters — not atom
order, since ASE and moltui can legitimately enumerate symmetry images in
different orders.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from moltui.parsers import parse_cif

ase = pytest.importorskip("ase")
from ase.io import read as ase_read  # noqa: E402

DATA_DIR = Path(__file__).parent.parent / "data" / "crystal"


def _wrap_fracs(fracs: np.ndarray) -> np.ndarray:
    """Wrap fractional coords into [0, 1) modulo the lattice."""
    return fracs - np.floor(fracs)


def _greedy_match(
    a: np.ndarray, b: np.ndarray, tol: float = 1e-3
) -> tuple[bool, list[int], list[int]]:
    """Greedy nearest-neighbour pairing of two fractional point sets.

    Treats coords as periodic — the shortest distance under any unit-cell
    translation is used (so atoms near a face don't fail to match their
    counterparts on the opposite face). Returns ``(ok, unmatched_a,
    unmatched_b)``.
    """
    if len(a) != len(b):
        return False, list(range(len(a))), list(range(len(b)))
    used_b = [False] * len(b)
    unmatched_a: list[int] = []
    for ia, pa in enumerate(a):
        best = -1
        best_d = float("inf")
        for ib, pb in enumerate(b):
            if used_b[ib]:
                continue
            delta = pa - pb
            delta -= np.round(delta)
            d = float(np.linalg.norm(delta))
            if d < best_d:
                best_d = d
                best = ib
        if best >= 0 and best_d < tol:
            used_b[best] = True
        else:
            unmatched_a.append(ia)
    unmatched_b = [i for i, u in enumerate(used_b) if not u]
    return not unmatched_a and not unmatched_b, unmatched_a, unmatched_b


def _moltui_fractional(mol) -> np.ndarray:
    inv = np.linalg.inv(mol.lattice)
    return np.array([a.position @ inv for a in mol.atoms], dtype=np.float64)


def _ase_fractional(atoms) -> np.ndarray:
    return atoms.get_scaled_positions(wrap=False)


def _assert_structures_equivalent(mol, atoms, tol: float = 1e-3) -> None:
    """Assert moltui's mol matches ASE's atoms structurally."""
    # Element counts.
    moltui_counts = Counter(a.element.symbol for a in mol.atoms)
    ase_counts = Counter(atoms.get_chemical_symbols())
    assert moltui_counts == ase_counts, (
        f"element counts differ: moltui={dict(moltui_counts)} ase={dict(ase_counts)}"
    )

    # Cell parameters (a, b, c).
    a, b, c = (np.linalg.norm(mol.lattice[i]) for i in range(3))
    cellpar = atoms.cell.cellpar()
    np.testing.assert_allclose([a, b, c], cellpar[:3], atol=1e-3)

    # Per-element fractional-coordinate sets must match (periodic, tol-based).
    moltui_frac = _wrap_fracs(_moltui_fractional(mol))
    ase_frac = _wrap_fracs(_ase_fractional(atoms))
    moltui_syms = [a.element.symbol for a in mol.atoms]
    ase_syms = atoms.get_chemical_symbols()
    for sym in moltui_counts:
        moltui_subset = np.array([moltui_frac[i] for i, s in enumerate(moltui_syms) if s == sym])
        ase_subset = np.array([ase_frac[i] for i, s in enumerate(ase_syms) if s == sym])
        ok, unmatched_m, unmatched_a = _greedy_match(moltui_subset, ase_subset, tol=tol)
        assert ok, (
            f"{sym} fractional positions differ (tol={tol}):\n"
            f"  moltui-only rows: {[moltui_subset[i].tolist() for i in unmatched_m]}\n"
            f"  ase-only rows:    {[ase_subset[i].tolist() for i in unmatched_a]}"
        )


# --- Real-file tests ---


@pytest.mark.skipif(not (DATA_DIR / "graphite.cif").exists(), reason="graphite.cif missing")
def test_graphite_matches_ase():
    path = DATA_DIR / "graphite.cif"
    mol = parse_cif(path)
    atoms = ase_read(str(path))
    _assert_structures_equivalent(mol, atoms)


@pytest.mark.skipif(not (DATA_DIR / "aspirin.cif").exists(), reason="aspirin.cif missing")
def test_aspirin_matches_ase():
    path = DATA_DIR / "aspirin.cif"
    mol = parse_cif(path)
    atoms = ase_read(str(path))
    _assert_structures_equivalent(mol, atoms)


# --- Hand-written CIFs covering edge cases ---


def _write(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(body)
    return path


def test_bcc_iron_special_position(tmp_path: Path):
    """BCC iron: I-centered with one atom at (0,0,0) → 2 atoms total.

    The body-centering operation maps (0,0,0) to (1/2,1/2,1/2); both
    must be present after expansion, with no duplication of the origin.
    """
    body = (
        "data_bcc_fe\n"
        "_symmetry_space_group_name_H-M 'P 1'\n"
        "_cell_length_a 2.866\n"
        "_cell_length_b 2.866\n"
        "_cell_length_c 2.866\n"
        "_cell_angle_alpha 90\n"
        "_cell_angle_beta 90\n"
        "_cell_angle_gamma 90\n"
        "loop_\n_symmetry_equiv_pos_as_xyz\n"
        "'x,y,z'\n'1/2+x,1/2+y,1/2+z'\n"
        "loop_\n"
        "_atom_site_label\n_atom_site_type_symbol\n"
        "_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n"
        "Fe1 Fe 0.0 0.0 0.0\n"
    )
    path = _write(tmp_path, "bcc.cif", body)
    mol = parse_cif(path)
    atoms = ase_read(str(path))
    _assert_structures_equivalent(mol, atoms)
    assert len(mol.atoms) == 2


def test_fcc_copper_face_centering(tmp_path: Path):
    """FCC: 4 face-centering ops on a single (0,0,0) atom → 4 atoms."""
    body = (
        "data_fcc_cu\n"
        "_symmetry_space_group_name_H-M 'P 1'\n"
        "_cell_length_a 3.615\n"
        "_cell_length_b 3.615\n"
        "_cell_length_c 3.615\n"
        "_cell_angle_alpha 90\n"
        "_cell_angle_beta 90\n"
        "_cell_angle_gamma 90\n"
        "loop_\n_symmetry_equiv_pos_as_xyz\n"
        "'x,y,z'\n"
        "'x,1/2+y,1/2+z'\n"
        "'1/2+x,y,1/2+z'\n"
        "'1/2+x,1/2+y,z'\n"
        "loop_\n"
        "_atom_site_label\n_atom_site_type_symbol\n"
        "_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n"
        "Cu1 Cu 0.0 0.0 0.0\n"
    )
    path = _write(tmp_path, "fcc.cif", body)
    mol = parse_cif(path)
    atoms = ase_read(str(path))
    _assert_structures_equivalent(mol, atoms)
    assert len(mol.atoms) == 4


def test_rocksalt_two_special_positions(tmp_path: Path):
    """NaCl: each ion on a face-centering orbit → 4 Na + 4 Cl."""
    body = (
        "data_rocksalt\n"
        "_symmetry_space_group_name_H-M 'P 1'\n"
        "_cell_length_a 5.64\n_cell_length_b 5.64\n_cell_length_c 5.64\n"
        "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
        "loop_\n_symmetry_equiv_pos_as_xyz\n"
        "'x,y,z'\n'x,1/2+y,1/2+z'\n'1/2+x,y,1/2+z'\n'1/2+x,1/2+y,z'\n"
        "loop_\n"
        "_atom_site_label\n_atom_site_type_symbol\n"
        "_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n"
        "Na1 Na 0.0 0.0 0.0\n"
        "Cl1 Cl 0.5 0.5 0.5\n"
    )
    path = _write(tmp_path, "nacl.cif", body)
    mol = parse_cif(path)
    atoms = ase_read(str(path))
    _assert_structures_equivalent(mol, atoms)
    assert Counter(a.element.symbol for a in mol.atoms) == {"Na": 4, "Cl": 4}


def test_hexagonal_thirds(tmp_path: Path):
    """Hexagonal cell with 1/3, 2/3 translation vectors exercises rational
    fractions in the symop parser and dedupe."""
    body = (
        "data_hex\n"
        "_symmetry_space_group_name_H-M 'P 1'\n"
        "_cell_length_a 3.0\n_cell_length_b 3.0\n_cell_length_c 5.0\n"
        "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 120\n"
        "loop_\n_symmetry_equiv_pos_as_xyz\n"
        "'x,y,z'\n'1/3+x,2/3+y,z'\n'2/3+x,1/3+y,z'\n"
        "loop_\n"
        "_atom_site_label\n_atom_site_type_symbol\n"
        "_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n"
        "C1 C 0.0 0.0 0.0\n"
    )
    path = _write(tmp_path, "hex.cif", body)
    mol = parse_cif(path)
    atoms = ase_read(str(path))
    _assert_structures_equivalent(mol, atoms)
    assert len(mol.atoms) == 3


def test_p21a_general_position_matches_ase(tmp_path: Path):
    """Synthetic P 21/a structure with a single general-position atom — checks
    that the four monoclinic ops produce the same four images as ASE."""
    body = (
        "data_p21a\n"
        "_symmetry_space_group_name_H-M 'P 1'\n"
        "_cell_length_a 14.8\n_cell_length_b 16.7\n_cell_length_c 3.97\n"
        "_cell_angle_alpha 90\n_cell_angle_beta 97.0\n_cell_angle_gamma 90\n"
        "loop_\n_symmetry_equiv_pos_as_xyz\n"
        "'x,y,z'\n'1/2-x,1/2+y,-z'\n'-x,-y,-z'\n'1/2+x,1/2-y,z'\n"
        "loop_\n"
        "_atom_site_label\n_atom_site_type_symbol\n"
        "_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n"
        "C1 C 0.241 0.222 -0.099\n"
    )
    path = _write(tmp_path, "p21a.cif", body)
    mol = parse_cif(path)
    atoms = ase_read(str(path))
    _assert_structures_equivalent(mol, atoms)
    assert len(mol.atoms) == 4
