"""Tests comparing MO data loaded via cclib against the reference n2.molden."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cclib")

from moltui.gto import parse_molden
from moltui.molden import OrbitalData, evaluate_mo
from moltui.parsers import load_orbital_data_from_cclib

DATA = Path(__file__).parent.parent / "data"
N2_OUT = DATA / "orca/mo/n2_nosym.out"
N2_MOLDEN = DATA / "orca/n2.molden.input"


def _require(path: Path) -> Path:
    if not path.exists():
        pytest.skip(f"data file not found: {path}")
    return path


@pytest.fixture(scope="module")
def od_cclib() -> OrbitalData:
    return load_orbital_data_from_cclib(_require(N2_OUT))


@pytest.fixture(scope="module")
def od_molden() -> OrbitalData:
    from moltui.elements import Atom, Molecule, get_element
    from moltui.parsers import BOHR_TO_ANGSTROM

    basis = parse_molden(_require(N2_MOLDEN))
    atoms = [
        Atom(
            element=get_element(sym),
            position=basis.atom_coords_bohr[i] * BOHR_TO_ANGSTROM,
        )
        for i, sym in enumerate(basis.atom_symbols)
    ]
    mol = Molecule(atoms=atoms, bonds=[])
    mol.detect_bonds()
    return OrbitalData.from_gto_basis(basis, mol)


def test_n_mos(od_cclib, od_molden):
    assert od_cclib.n_mos == od_molden.n_mos


def test_mo_energies(od_cclib, od_molden):
    np.testing.assert_allclose(od_cclib.mo_energies, od_molden.mo_energies, atol=1e-4)


def test_mo_occupations(od_cclib, od_molden):
    # cclib truncates fractional occupations to 4 decimal places in the output file
    np.testing.assert_allclose(od_cclib.mo_occupations, od_molden.mo_occupations, atol=1e-4)


def test_ao_basis_shells(od_cclib, od_molden):
    shells_cclib = od_cclib._basis.shells
    shells_molden = od_molden._basis.shells
    assert len(shells_cclib) == len(shells_molden)
    for i, (sc, sm) in enumerate(zip(shells_cclib, shells_molden)):
        assert sc.l == sm.l, f"shell {i}: l mismatch {sc.l} != {sm.l}"
        np.testing.assert_allclose(sc.center, sm.center, atol=1e-5, err_msg=f"shell {i} center")
        np.testing.assert_allclose(
            sc.exponents, sm.exponents, atol=1e-5, err_msg=f"shell {i} exponents"
        )
        # Coefficients are not compared: for l >= 2, ORCA's molden omits the
        # double-factorial from the primitive norm while gto._prim_norm includes it,
        # leading to a sqrt(dfact(2l-1)) difference in stored values. Both conventions
        # produce identical normalized AOs after _contraction_norm — verified by
        # test_homo_cube.


def test_mo_coefficients(od_cclib, od_molden):
    # MO coefficients may differ by an overall sign per orbital
    coeff_c = od_cclib._basis.mo_coefficients
    coeff_m = od_molden._basis.mo_coefficients
    assert coeff_c.shape == coeff_m.shape
    for mo in range(coeff_c.shape[1]):
        col_c = coeff_c[:, mo]
        col_m = coeff_m[:, mo]
        # Align sign: pick the sign of the largest-magnitude element
        sign = np.sign(col_c[np.argmax(np.abs(col_c))]) * np.sign(col_m[np.argmax(np.abs(col_m))])
        np.testing.assert_allclose(col_c, sign * col_m, atol=1e-5, err_msg=f"MO {mo}")


def test_homo_cube(od_cclib, od_molden):
    homo = od_cclib.homo_idx
    assert homo == od_molden.homo_idx
    cube_c = evaluate_mo(od_cclib, homo)
    cube_m = evaluate_mo(od_molden, homo)
    assert cube_c.data.shape == cube_m.data.shape
    # Allow for global sign flip
    sign = np.sign(cube_c.data.flat[np.argmax(np.abs(cube_c.data))]) * np.sign(
        cube_m.data.flat[np.argmax(np.abs(cube_m.data))]
    )
    np.testing.assert_allclose(cube_c.data, sign * cube_m.data, atol=1e-4)
