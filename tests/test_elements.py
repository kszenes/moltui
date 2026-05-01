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
from moltui.geometry_panel import GeometryPanel

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

    def test_periodic_angle_uses_bond_shifts(self):
        H = get_element("H")
        lattice = np.diag([2.0, 2.0, 2.0])
        mol = Molecule(
            atoms=[
                Atom(H, np.array([0.1, 0.1, 0.1])),
                Atom(H, np.array([1.9, 0.1, 0.1])),
                Atom(H, np.array([0.1, 1.0, 0.1])),
            ],
            bonds=[(0, 1), (0, 2)],
            lattice=lattice,
            bond_shifts=[(-1, 0, 0), (0, 0, 0)],
        )
        angles = mol.get_angles()
        assert len(angles) == 1
        assert angles[0][3] == pytest.approx(90.0, abs=0.5)


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

    def test_periodic_dihedral_uses_bond_shifts(self):
        C = get_element("C")
        lattice = np.diag([2.0, 4.0, 4.0])
        mol = Molecule(
            atoms=[
                Atom(C, np.array([0.0, 0.0, 0.0])),
                Atom(C, np.array([1.0, 0.0, 0.0])),
                Atom(C, np.array([1.5, 1.0, 0.0])),
                Atom(C, np.array([0.5, 1.0, 0.5])),
            ],
            bonds=[(0, 1), (1, 2), (2, 3)],
            lattice=lattice,
            bond_shifts=[(0, 0, 0), (0, 0, 0), (1, 0, 0)],
        )
        reference = Molecule(
            atoms=[
                Atom(C, np.array([0.0, 0.0, 0.0])),
                Atom(C, np.array([1.0, 0.0, 0.0])),
                Atom(C, np.array([1.5, 1.0, 0.0])),
                Atom(C, np.array([2.5, 1.0, 0.5])),
            ],
            bonds=[(0, 1), (1, 2), (2, 3)],
        )
        dihedrals = mol.get_dihedrals()
        assert len(dihedrals) == 1
        assert dihedrals[0][4] == pytest.approx(reference.get_dihedrals()[0][4], abs=0.5)


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

    def test_detect_bonds_periodic_across_boundary(self):
        H = get_element("H")
        mol = Molecule(
            atoms=[
                Atom(H, np.array([0.0, 0.0, 0.0])),
                Atom(H, np.array([1.9, 0.0, 0.0])),
            ],
            bonds=[],
            lattice=np.diag([2.0, 2.0, 2.0]),
        )
        mol.detect_bonds_periodic()
        assert mol.bonds == [(0, 1)]
        assert mol.bond_shifts == [(-1, 0, 0)]

    def test_detect_bonds_periodic_raises_without_lattice(self):
        H = get_element("H")
        mol = Molecule(atoms=[Atom(H, np.array([0.0, 0.0, 0.0]))], bonds=[])
        with pytest.raises(ValueError, match="requires a lattice"):
            mol.detect_bonds_periodic()

    def test_detect_bonds_periodic_empty_molecule(self):
        mol = Molecule(atoms=[], bonds=[], lattice=np.eye(3))
        mol.detect_bonds_periodic()
        assert mol.bonds == []
        assert mol.bond_shifts is None

    def test_detect_bonds_auto_uses_nonperiodic_without_lattice(self):
        H = get_element("H")
        mol = Molecule(
            atoms=[
                Atom(H, np.array([0.0, 0.0, 0.0])),
                Atom(H, np.array([0.74, 0.0, 0.0])),
            ],
            bonds=[],
        )
        mol.detect_bonds_auto()
        assert mol.bonds == [(0, 1)]
        assert mol.bond_shifts is None

    def test_detect_bonds_auto_uses_periodic_with_lattice(self):
        H = get_element("H")
        mol = Molecule(
            atoms=[
                Atom(H, np.array([0.0, 0.0, 0.0])),
                Atom(H, np.array([1.9, 0.0, 0.0])),
            ],
            bonds=[],
            lattice=np.diag([2.0, 2.0, 2.0]),
        )
        mol.detect_bonds_auto()
        assert mol.bonds == [(0, 1)]
        assert mol.bond_shifts == [(-1, 0, 0)]

    def test_detect_bonds_periodic_nacl_like_has_six_cation_neighbors(self):
        C = get_element("C")
        H = get_element("H")
        mol = Molecule(
            atoms=[
                Atom(C, np.array([0.0, 0.0, 0.0])),
                Atom(H, np.array([1.0, 0.0, 0.0])),
                Atom(H, np.array([0.0, 1.0, 0.0])),
                Atom(H, np.array([0.0, 0.0, 1.0])),
            ],
            bonds=[],
            lattice=np.diag([2.0, 2.0, 2.0]),
        )
        mol.detect_bonds_periodic()

        assert mol.bond_shifts is not None
        cation_bonds = [
            (bond, shift)
            for bond, shift in zip(mol.bonds, mol.bond_shifts, strict=False)
            if bond[0] == 0 or bond[1] == 0
        ]
        assert len(cation_bonds) == 6

        vectors = []
        for (i, j), shift in cation_bonds:
            if i == 0:
                base = mol.atoms[i].position
                other = mol.atoms[j].position
                sign = 1.0
            else:
                base = mol.atoms[j].position
                other = mol.atoms[i].position
                sign = -1.0
            disp = shift[0] * mol.lattice[0] + shift[1] * mol.lattice[1] + shift[2] * mol.lattice[2]
            vectors.append(tuple(np.round(sign * (other + disp - base), 6)))

        assert set(vectors) == {
            (-1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, -1.0),
            (0.0, 0.0, 1.0),
        }


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

    def test_with_bonded_periodic_images_graphite(self):
        lattice = np.array(
            [
                [2.46, 0.0, 0.0],
                [-1.23, 2.13042249, 0.0],
                [0.0, 0.0, 6.71],
            ]
        )
        C = get_element("C")
        mol = Molecule(
            atoms=[
                Atom(C, np.array([0.0, 0.0, 0.0])),
                Atom(C, np.array([0.0, 1.42028166, 0.0])),
                Atom(C, np.array([0.0, 0.0, 3.355])),
                Atom(C, np.array([1.23, 0.71014083, 3.355])),
            ],
            bonds=[],
            lattice=lattice,
        )
        mol.detect_bonds_periodic()

        replicated = mol.with_bonded_periodic_images()
        assert len(replicated.atoms) == 36

        degrees = [0] * len(replicated.atoms)
        for i, j in replicated.bonds:
            degrees[i] += 1
            degrees[j] += 1
        assert all(degree == 3 for degree in degrees[:4])


class TestGeometryPanelLabels:
    def test_parent_indices_control_atom_label(self):
        C = get_element("C")
        panel = GeometryPanel()
        panel._molecule = Molecule(
            atoms=[Atom(C, np.zeros(3)) for _ in range(5)],
            bonds=[],
        )
        panel._parent_indices = [0, 1, 0, 1, 0]
        assert panel._atom_label(2) == "1:C"
