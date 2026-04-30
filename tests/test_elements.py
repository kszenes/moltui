#!/usr/bin/env python3
"""Tests for elements.py: Element lookups, Atom, Molecule geometry."""

from __future__ import annotations

import numpy as np
import pytest

from moltui.elements import (
    DEFAULT_ELEMENT,
    ELEMENTS,
    Atom,
    Molecule,
    get_element,
    get_element_by_number,
)

# --- Element lookups ---


class TestGetElement:
    def test_known_symbol(self):
        e = get_element("C")
        assert e.symbol == "C"
        assert e.atomic_number == 6

    def test_case_insensitive(self):
        assert get_element("c").symbol == "C"
        assert get_element("FE").symbol == "Fe"

    def test_whitespace_stripped(self):
        assert get_element("  O  ").symbol == "O"

    def test_unknown_returns_default(self):
        e = get_element("Xx")
        assert e is DEFAULT_ELEMENT
        assert e.symbol == "X"


class TestGetElementByNumber:
    def test_known_number(self):
        e = get_element_by_number(1)
        assert e.symbol == "H"

    def test_unknown_number_returns_default(self):
        assert get_element_by_number(999) is DEFAULT_ELEMENT

    def test_all_elements_round_trip(self):
        for sym, elem in ELEMENTS.items():
            assert get_element_by_number(elem.atomic_number).symbol == sym


# --- Molecule geometry ---


def _water() -> Molecule:
    """Build a water molecule with known geometry."""
    O = get_element("O")
    H = get_element("H")
    atoms = [
        Atom(O, np.array([0.0, 0.0, 0.0])),
        Atom(H, np.array([0.757, 0.586, 0.0])),
        Atom(H, np.array([-0.757, 0.586, 0.0])),
    ]
    mol = Molecule(atoms=atoms, bonds=[(0, 1), (0, 2)])
    return mol


def _linear_triatomic() -> Molecule:
    """Three atoms in a line: A-B-C along the x-axis."""
    C = get_element("C")
    atoms = [
        Atom(C, np.array([0.0, 0.0, 0.0])),
        Atom(C, np.array([1.0, 0.0, 0.0])),
        Atom(C, np.array([2.0, 0.0, 0.0])),
    ]
    return Molecule(atoms=atoms, bonds=[(0, 1), (1, 2)])


class TestMoleculeCenter:
    def test_single_atom(self):
        H = get_element("H")
        mol = Molecule(atoms=[Atom(H, np.array([3.0, 4.0, 5.0]))], bonds=[])
        np.testing.assert_array_almost_equal(mol.center(), [3.0, 4.0, 5.0])

    def test_two_atoms(self):
        H = get_element("H")
        mol = Molecule(
            atoms=[
                Atom(H, np.array([0.0, 0.0, 0.0])),
                Atom(H, np.array([2.0, 0.0, 0.0])),
            ],
            bonds=[],
        )
        np.testing.assert_array_almost_equal(mol.center(), [1.0, 0.0, 0.0])

    def test_empty_molecule(self):
        mol = Molecule(atoms=[], bonds=[])
        np.testing.assert_array_almost_equal(mol.center(), [0.0, 0.0, 0.0])


class TestMoleculeRadius:
    def test_single_atom_radius_is_zero(self):
        H = get_element("H")
        mol = Molecule(atoms=[Atom(H, np.array([1.0, 2.0, 3.0]))], bonds=[])
        assert mol.radius() == 0.0

    def test_two_atoms_radius(self):
        H = get_element("H")
        mol = Molecule(
            atoms=[
                Atom(H, np.array([0.0, 0.0, 0.0])),
                Atom(H, np.array([4.0, 0.0, 0.0])),
            ],
            bonds=[],
        )
        assert mol.radius() == pytest.approx(2.0)

    def test_empty_molecule(self):
        mol = Molecule(atoms=[], bonds=[])
        assert mol.radius() == 1.0  # default


class TestBondLengths:
    def test_water_bond_lengths(self):
        mol = _water()
        lengths = mol.get_bond_lengths()
        assert len(lengths) == 2
        for i, j, dist in lengths:
            assert dist == pytest.approx(0.9584, abs=0.01)

    def test_no_bonds(self):
        H = get_element("H")
        mol = Molecule(atoms=[Atom(H, np.array([0.0, 0.0, 0.0]))], bonds=[])
        assert mol.get_bond_lengths() == []


class TestAngles:
    def test_water_angle(self):
        mol = _water()
        angles = mol.get_angles()
        assert len(angles) == 1
        _, _, _, angle = angles[0]
        assert angle == pytest.approx(104.5, abs=1.0)

    def test_linear_angle(self):
        mol = _linear_triatomic()
        angles = mol.get_angles()
        assert len(angles) == 1
        assert angles[0][3] == pytest.approx(180.0, abs=0.1)

    def test_no_angles_for_diatomic(self):
        H = get_element("H")
        mol = Molecule(
            atoms=[
                Atom(H, np.array([0.0, 0.0, 0.0])),
                Atom(H, np.array([1.0, 0.0, 0.0])),
            ],
            bonds=[(0, 1)],
        )
        assert mol.get_angles() == []


class TestDihedrals:
    def test_butane_like_dihedral(self):
        """Four atoms in a known dihedral arrangement."""
        C = get_element("C")
        atoms = [
            Atom(C, np.array([0.0, 0.0, 0.0])),
            Atom(C, np.array([1.0, 0.0, 0.0])),
            Atom(C, np.array([1.5, 1.0, 0.0])),
            Atom(C, np.array([2.5, 1.0, 0.5])),
        ]
        mol = Molecule(atoms=atoms, bonds=[(0, 1), (1, 2), (2, 3)])
        dihedrals = mol.get_dihedrals()
        assert len(dihedrals) == 1
        _, _, _, _, angle = dihedrals[0]
        assert 0.0 <= angle <= 180.0

    def test_planar_dihedral_is_zero_or_180(self):
        """Four coplanar atoms should have dihedral 0 or 180."""
        C = get_element("C")
        atoms = [
            Atom(C, np.array([0.0, 0.0, 0.0])),
            Atom(C, np.array([1.0, 0.0, 0.0])),
            Atom(C, np.array([2.0, 1.0, 0.0])),
            Atom(C, np.array([3.0, 1.0, 0.0])),
        ]
        mol = Molecule(atoms=atoms, bonds=[(0, 1), (1, 2), (2, 3)])
        dihedrals = mol.get_dihedrals()
        assert len(dihedrals) == 1
        angle = dihedrals[0][4]
        assert angle == pytest.approx(0.0, abs=0.5) or angle == pytest.approx(180.0, abs=0.5)

    def test_no_dihedrals_for_three_atoms(self):
        mol = _linear_triatomic()
        assert mol.get_dihedrals() == []


class TestDetectBonds:
    def test_h2_bonded(self):
        H = get_element("H")
        mol = Molecule(
            atoms=[
                Atom(H, np.array([0.0, 0.0, 0.0])),
                Atom(H, np.array([0.74, 0.0, 0.0])),
            ],
            bonds=[],
        )
        mol.detect_bonds()
        assert len(mol.bonds) == 1
        assert mol.bonds[0] == (0, 1)

    def test_far_atoms_not_bonded(self):
        H = get_element("H")
        mol = Molecule(
            atoms=[
                Atom(H, np.array([0.0, 0.0, 0.0])),
                Atom(H, np.array([10.0, 0.0, 0.0])),
            ],
            bonds=[],
        )
        mol.detect_bonds()
        assert len(mol.bonds) == 0

    def test_detect_bonds_clears_previous(self):
        H = get_element("H")
        mol = Molecule(
            atoms=[
                Atom(H, np.array([0.0, 0.0, 0.0])),
                Atom(H, np.array([0.74, 0.0, 0.0])),
            ],
            bonds=[(0, 1), (0, 1)],  # duplicates
        )
        mol.detect_bonds()
        assert len(mol.bonds) == 1


class TestPeriodicImages:
    def test_no_lattice_returns_self(self):
        H = get_element("H")
        mol = Molecule(atoms=[Atom(H, np.array([0.0, 0.0, 0.0]))], bonds=[])
        assert mol.with_periodic_images() is mol

    def test_corner_atom_replicates_to_8(self):
        H = get_element("H")
        lattice = np.eye(3) * 4.0
        mol = Molecule(
            atoms=[Atom(H, np.array([0.0, 0.0, 0.0]))],
            bonds=[],
            lattice=lattice,
        )
        replicated = mol.with_periodic_images()
        assert len(replicated.atoms) == 8

    def test_face_atom_replicates_to_2(self):
        H = get_element("H")
        lattice = np.eye(3) * 4.0
        mol = Molecule(
            atoms=[Atom(H, np.array([0.0, 2.0, 2.0]))],
            bonds=[],
            lattice=lattice,
        )
        replicated = mol.with_periodic_images()
        assert len(replicated.atoms) == 2

    def test_interior_atom_unchanged(self):
        H = get_element("H")
        lattice = np.eye(3) * 4.0
        mol = Molecule(
            atoms=[Atom(H, np.array([2.0, 2.0, 2.0]))],
            bonds=[],
            lattice=lattice,
        )
        replicated = mol.with_periodic_images()
        assert len(replicated.atoms) == 1

    def test_lattice_preserved(self):
        H = get_element("H")
        lattice = np.eye(3) * 4.0
        mol = Molecule(
            atoms=[Atom(H, np.array([0.0, 2.0, 2.0]))],
            bonds=[],
            lattice=lattice,
        )
        replicated = mol.with_periodic_images()
        np.testing.assert_array_equal(replicated.lattice, lattice)


class TestSupercell:
    def test_no_lattice_returns_self(self):
        H = get_element("H")
        mol = Molecule(atoms=[Atom(H, np.array([0.0, 0.0, 0.0]))], bonds=[])
        assert mol.supercell(2, 2, 2) is mol

    def test_unit_dims_returns_self(self):
        H = get_element("H")
        mol = Molecule(
            atoms=[Atom(H, np.array([1.0, 1.0, 1.0]))],
            bonds=[],
            lattice=np.eye(3) * 4.0,
        )
        assert mol.supercell(1, 1, 1) is mol

    def test_2x1x1_doubles_atoms_and_a_axis(self):
        H = get_element("H")
        mol = Molecule(
            atoms=[Atom(H, np.array([1.0, 1.0, 1.0]))],
            bonds=[],
            lattice=np.eye(3) * 4.0,
        )
        sc = mol.supercell(2, 1, 1)
        assert len(sc.atoms) == 2
        np.testing.assert_array_equal(sc.atoms[0].position, [1.0, 1.0, 1.0])
        np.testing.assert_array_equal(sc.atoms[1].position, [5.0, 1.0, 1.0])
        np.testing.assert_array_equal(sc.lattice[0], [8.0, 0.0, 0.0])
        np.testing.assert_array_equal(sc.lattice[1], [0.0, 4.0, 0.0])

    def test_2x2x2_count(self):
        H = get_element("H")
        mol = Molecule(
            atoms=[
                Atom(H, np.array([1.0, 1.0, 1.0])),
                Atom(H, np.array([2.0, 2.0, 2.0])),
            ],
            bonds=[],
            lattice=np.eye(3) * 4.0,
        )
        sc = mol.supercell(2, 2, 2)
        assert len(sc.atoms) == 16

    def test_invalid_dims_raises(self):
        H = get_element("H")
        mol = Molecule(
            atoms=[Atom(H, np.array([0.0, 0.0, 0.0]))],
            bonds=[],
            lattice=np.eye(3),
        )
        with pytest.raises(ValueError, match=">= 1"):
            mol.supercell(0, 1, 1)
