from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from moltui.parsers import load_molecule, parse_poscar, parse_xsf, parse_xyz
from tests.structure_compare import assert_molecule_matches_ase_atoms

ase = pytest.importorskip("ase")
from ase.io import read as ase_read  # noqa: E402


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


def _ase(path: Path, fmt: str):
    return ase_read(str(path), format=fmt)


POSCAR_DIRECT = """Si C
1.0
3.0 0.0 0.0
0.0 3.0 0.0
0.0 0.0 3.0
Si C
1 1
Direct
0.0 0.0 0.0
0.5 0.5 0.5
"""


def test_poscar_vasp5_direct_matches_ase(tmp_path: Path):
    path = _write(tmp_path, "POSCAR", POSCAR_DIRECT)
    mol = parse_poscar(path)
    assert_molecule_matches_ase_atoms(mol, _ase(path, "vasp"))
    assert mol.lattice is not None


def test_poscar_vasp5_cartesian_selective_nonorthogonal_matches_ase(tmp_path: Path):
    text = """Selective nonorthogonal
2.0
2.0 0.0 0.0
0.4 1.8 0.0
0.1 0.2 2.2
C H
1 1
Selective dynamics
Cartesian
0.0 0.0 0.0 T T T
0.55 0.0 0.0 F F F
"""
    path = _write(tmp_path, "case.vasp", text)
    mol = parse_poscar(path)
    assert_molecule_matches_ase_atoms(mol, _ase(path, "vasp"))


def test_poscar_vasp4_symbols_from_comment_matches_ase(tmp_path: Path):
    text = """Na Cl
1.0
5.64 0.0 0.0
0.0 5.64 0.0
0.0 0.0 5.64
1 1
Direct
0.0 0.0 0.0
0.5 0.5 0.5
"""
    path = _write(tmp_path, "CONTCAR", text)
    mol = parse_poscar(path)
    assert_molecule_matches_ase_atoms(mol, _ase(path, "vasp"))


def test_poscar_negative_volume_scale(tmp_path: Path):
    text = """C
-27.0
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0
C
1
Direct
0.25 0.25 0.25
"""
    path = _write(tmp_path, "neg.poscar", text)
    mol = parse_poscar(path)
    np.testing.assert_allclose(mol.lattice, np.diag([3.0, 3.0, 3.0]))
    np.testing.assert_allclose(mol.atoms[0].position, [0.75, 0.75, 0.75])


def test_load_molecule_dispatches_poscar_names_and_suffix(tmp_path: Path):
    assert len(load_molecule(_write(tmp_path, "POSCAR", POSCAR_DIRECT)).atoms) == 2
    assert len(load_molecule(_write(tmp_path, "sample.vasp", POSCAR_DIRECT)).atoms) == 2


def test_periodic_bonds_detected_for_poscar_boundary_bond(tmp_path: Path):
    text = """H
1.0
2.0 0.0 0.0
0.0 5.0 0.0
0.0 0.0 5.0
H
2
Direct
0.05 0.0 0.0
0.95 0.0 0.0
"""
    mol = parse_poscar(_write(tmp_path, "POSCAR", text))
    assert mol.bonds == [(0, 1)]
    assert mol.bond_shifts == [(-1, 0, 0)]
    assert mol.get_bond_lengths()[0][2] == pytest.approx(0.2)


def test_extxyz_z_and_extra_properties_matches_ase_and_uses_periodic_bonds(tmp_path: Path):
    text = (
        "2\n"
        'Lattice="2.0 0.0 0.0 0.0 5.0 0.0 0.0 0.0 5.0" '
        'Properties="forces:R:3:Z:I:1:pos:R:3:energy:R:1"\n'
        "0 0 0 1 0.1 0.0 0.0 -1.0\n"
        "0 0 0 1 1.9 0.0 0.0 -1.0\n"
    )
    path = _write(tmp_path, "h.extxyz", text)
    mol = parse_xyz(path)
    assert_molecule_matches_ase_atoms(mol, _ase(path, "extxyz"))
    assert mol.bond_shifts == [(-1, 0, 0)]


def test_extxyz_pbc_false_axis_suppresses_periodic_bonds(tmp_path: Path):
    text = (
        "2\n"
        'Lattice="5.0 0.0 0.0 0.0 5.0 0.0 0.0 0.0 2.0" '
        'pbc="T T F" Properties="species:S:1:pos:R:3"\n'
        "H 0.0 0.0 0.1\n"
        "H 0.0 0.0 1.9\n"
    )
    mol = parse_xyz(_write(tmp_path, "slab.extxyz", text))

    assert mol.pbc == (True, True, False)
    assert mol.bonds == []


def test_extxyz_pbc_true_axis_allows_periodic_bonds(tmp_path: Path):
    text = (
        "2\n"
        'Lattice="5.0 0.0 0.0 0.0 5.0 0.0 0.0 0.0 2.0" '
        'pbc="T T T" Properties="species:S:1:pos:R:3"\n'
        "H 0.0 0.0 0.1\n"
        "H 0.0 0.0 1.9\n"
    )
    mol = parse_xyz(_write(tmp_path, "bulk.extxyz", text))

    assert mol.pbc == (True, True, True)
    assert mol.bond_shifts == [(0, 0, -1)]


def test_xsf_crystal_structure_matches_ase(tmp_path: Path):
    text = """CRYSTAL
PRIMVEC
3.0 0.0 0.0
0.0 3.0 0.0
0.0 0.0 3.0
PRIMCOORD
2 1
C 0.0 0.0 0.0
8 1.5 1.5 1.5
"""
    path = _write(tmp_path, "structure.xsf", text)
    mol = parse_xsf(path)
    assert_molecule_matches_ase_atoms(mol, _ase(path, "xsf"))


def test_xsf_dispatch_and_periodic_bonds(tmp_path: Path):
    text = """CRYSTAL
PRIMVEC
2.0 0.0 0.0
0.0 5.0 0.0
0.0 0.0 5.0
PRIMCOORD
2 1
H 0.1 0.0 0.0
H 1.9 0.0 0.0
"""
    mol = load_molecule(_write(tmp_path, "boundary.xsf", text))
    assert mol.bond_shifts == [(-1, 0, 0)]
