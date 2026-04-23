"""Integration tests exercising real file parsers against the data/ directory."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

DATA = Path(__file__).parent.parent / "data"
TREXIO_DATA = DATA / "trexio" / "n2_sp"
BOHR = 0.529177249


def _require(path: Path) -> Path:
    if not path.exists():
        pytest.skip(f"data file not found: {path}")
    return path


def _bond(mol, i=0, j=1):
    return np.linalg.norm(mol.atoms[i].position - mol.atoms[j].position)


def _angle_deg(mol, i, j, k):
    """Angle i-j-k in degrees."""
    u = mol.atoms[i].position - mol.atoms[j].position
    v = mol.atoms[k].position - mol.atoms[j].position
    cos = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


# ---------------------------------------------------------------------------
# Geom: XYZ vs Z-matrix (water)
# ---------------------------------------------------------------------------


def test_geom_water_xyz_zmat_same_geometry():
    """water.xyz and water.zmat encode the same geometry: O-H, H-H, and H-O-H angle."""
    from moltui.parsers import parse_xyz, parse_zmat

    mol_xyz = parse_xyz(_require(DATA / "geom" / "water.xyz"))
    mol_zmat = parse_zmat(_require(DATA / "geom" / "water.zmat"))

    for mol in (mol_xyz, mol_zmat):
        assert [a.element.symbol for a in mol.atoms] == ["O", "H", "H"]
        # O-H bond lengths
        np.testing.assert_allclose(_bond(mol, 0, 1), 0.9572, atol=1e-5)
        np.testing.assert_allclose(_bond(mol, 0, 2), 0.9572, atol=1e-5)
        # H-H distance: 2r·sin(θ/2) with θ=104.52°
        expected_hh = 2 * 0.9572 * np.sin(np.radians(104.52 / 2))
        np.testing.assert_allclose(_bond(mol, 1, 2), expected_hh, atol=1e-5)
        # H-O-H angle
        np.testing.assert_allclose(_angle_deg(mol, 1, 0, 2), 104.52, atol=1e-3)


# ---------------------------------------------------------------------------
# Geom: XYZ trajectory (benzene)
# ---------------------------------------------------------------------------


def test_geom_benzene_trajectory():
    from moltui.parsers import parse_xyz_trajectory

    trj = parse_xyz_trajectory(_require(DATA / "geom" / "benzene_trj.xyz"))

    assert len(trj.molecule.atoms) == 12
    assert [a.element.symbol for a in trj.molecule.atoms] == ["C"] * 6 + ["H"] * 6
    assert trj.frames.shape == (12, 12, 3)

    # First frame: exact atom positions from the file
    np.testing.assert_allclose(trj.frames[0, 0], [1.4, 0.0, 0.32], atol=1e-8)

    # Last frame: first carbon position
    np.testing.assert_allclose(trj.frames[-1, 0], [1.404335e00, 0.0, 8.0e-06], atol=1e-6)

    # molecule attribute reflects frame 0 coordinates
    np.testing.assert_allclose(trj.molecule.atoms[0].position, trj.frames[0, 0], atol=1e-10)


# ---------------------------------------------------------------------------
# Molcas: N2 SCF molden
# ---------------------------------------------------------------------------


def test_molcas_n2():
    from moltui.molden import load_molden_data

    orb = load_molden_data(_require(DATA / "molcas" / "n2.scf.molden"))

    # Geometry
    assert len(orb.molecule.atoms) == 2
    assert all(a.element.symbol == "N" for a in orb.molecule.atoms)
    # 2.07491929 Bohr → Angstrom
    np.testing.assert_allclose(_bond(orb.molecule), 2.07491929 * BOHR, atol=1e-5)

    # Orbital data
    assert orb.n_mos == 28
    assert orb.homo_idx == 22
    np.testing.assert_allclose(orb.mo_energies[orb.homo_idx], -0.608, atol=1e-3)

    # First 5 MO energies (limited precision in Molcas molden format)
    np.testing.assert_allclose(
        orb.mo_energies[:5], [-15.687, -15.683, -1.471, -0.7742, -0.6262], atol=1e-4
    )
    # All MO energies (limited precision in Molcas molden format)
    np.testing.assert_allclose(
        orb.mo_energies,
        [
            -15.687,
            -15.683,
            -1.471,
            -0.7742,
            -0.6262,
            0.5945,
            0.8202,
            0.9922,
            1.1439,
            1.6429,
            1.7575,
            2.2988,
            2.8741,
            3.2831,
            -0.608,
            0.1755,
            0.8723,
            1.051,
            1.8817,
            2.9961,
            1.7575,
            2.2988,
            -0.608,
            0.1755,
            0.8723,
            1.051,
            1.8817,
            2.9961,
        ],
        atol=1e-4,
    )
    # All occupations
    np.testing.assert_allclose(
        orb.mo_occupations,
        [
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            2.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            2.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
    )
    # All symmetry labels
    assert orb.mo_symmetries == (["A1"] * 14 + ["B1"] * 6 + ["A2"] * 2 + ["B2"] * 6)


# ---------------------------------------------------------------------------
# Orca: h2o.hess and n2.molden.input
# ---------------------------------------------------------------------------


def test_orca_h2o_hess():
    from moltui.parsers import parse_orca_hess_data

    hess = parse_orca_hess_data(_require(DATA / "orca" / "h2o.hess"))

    # Geometry
    assert len(hess.molecule.atoms) == 3
    assert hess.molecule.atoms[0].element.symbol == "O"
    assert {a.element.symbol for a in hess.molecule.atoms} == {"H", "O"}

    expected_pos = np.array(
        [
            [-0.04013049, 0.0, 0.05184791],  # O (Bohr → Ang)
            [0.91706958, 0.0, 0.05184791],  # H
            [-0.28011751, 0.0, -0.87477916],  # H
        ]
    )
    for i, atom in enumerate(hess.molecule.atoms):
        np.testing.assert_allclose(atom.position, expected_pos[i], atol=1e-6)

    # Frequencies: first 6 are zero (translations/rotations), last 3 are real vibrations
    assert hess.frequencies is not None
    assert len(hess.frequencies) == 9
    np.testing.assert_allclose(hess.frequencies[:6], 0.0, atol=1e-8)
    np.testing.assert_allclose(
        hess.frequencies[6:],
        [1530.7546082017, 3912.3240091405, 4029.8961503744],
        atol=1e-4,
    )

    # Normal modes
    assert hess.normal_modes is not None
    assert hess.normal_modes.shape == (9, 3, 3)


def test_orca_n2_molden():
    from moltui.molden import load_molden_data

    orb = load_molden_data(_require(DATA / "orca" / "n2.molden.input"))

    # Geometry: symmetric about origin, ±1.0374596475 Bohr
    assert len(orb.molecule.atoms) == 2
    assert all(a.element.symbol == "N" for a in orb.molecule.atoms)
    np.testing.assert_allclose(_bond(orb.molecule), 2 * 1.0374596475 * BOHR, atol=1e-6)

    # Orbital data
    assert orb.n_mos == 28
    assert orb.homo_idx == 6
    np.testing.assert_allclose(orb.mo_energies[orb.homo_idx], -1.01651068178511, atol=1e-8)

    # All MO energies (high precision in Orca molden format)
    np.testing.assert_allclose(
        orb.mo_energies,
        [
            -15.6449058610371,
            -15.64215461089,
            -1.03995052337999,
            -0.758523735526424,
            -0.585576851041608,
            -0.585576851041606,
            -1.01651068178511,
            0.312622537259467,
            0.312622537259465,
            1.27378440477447,
            0.707365890027976,
            0.827755818271803,
            0.878723391374256,
            0.878723391374252,
            0.931721040080917,
            0.931721040080924,
            1.00500408396294,
            1.55693939695611,
            1.76897117072112,
            1.76897117072112,
            1.89447591291943,
            1.89447591291943,
            2.30886256692502,
            2.30886256692503,
            2.88667706857542,
            2.99803308211133,
            2.99803308211133,
            3.17167282790304,
        ],
        atol=1e-8,
    )
    # All occupations
    np.testing.assert_allclose(
        orb.mo_occupations,
        [
            2.0,
            2.0,
            2.0,
            2.0,
            1.941708,
            1.941708,
            1.982237,
            0.058205,
            0.058205,
            0.017936,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        atol=1e-6,
    )
    # Orca does not write symmetry labels — all default to "A"
    assert orb.mo_symmetries == ["A"] * 28


# ---------------------------------------------------------------------------
# PySCF: H2O with normal modes
# ---------------------------------------------------------------------------


def test_pyscf_h2o_normal_modes():
    from moltui.molden import load_molden_data

    orb = load_molden_data(_require(DATA / "pyscf" / "h2o_mo_normal_modes.molden"))

    # Geometry
    assert len(orb.molecule.atoms) == 3
    assert {a.element.symbol for a in orb.molecule.atoms} == {"H", "O"}

    # MOs
    assert orb.n_mos == 7
    assert orb.homo_idx == 4
    np.testing.assert_allclose(orb.mo_energies[orb.homo_idx], -0.3912330487, atol=1e-8)
    np.testing.assert_allclose(orb.mo_occupations[:5], [2.0, 2.0, 2.0, 2.0, 2.0])
    np.testing.assert_allclose(orb.mo_occupations[5:], [0.0, 0.0])

    # Vibrational frequencies (exact values from file)
    assert orb.mode_frequencies is not None
    assert orb.normal_modes is not None
    assert orb.normal_modes.shape == (3, 3, 3)
    np.testing.assert_allclose(
        orb.mode_frequencies,
        [2042.2778651037, 4490.1568540958, 4792.6324264867],
        atol=1e-6,
    )


# ---------------------------------------------------------------------------
# PySCF: benzene — molden vs cube MO evaluation
# ---------------------------------------------------------------------------


def test_pyscf_benzene_molden():
    from moltui.molden import load_molden_data

    orb = load_molden_data(_require(DATA / "pyscf" / "benzene.molden"))

    assert len(orb.molecule.atoms) == 12
    symbols = [a.element.symbol for a in orb.molecule.atoms]
    assert symbols.count("C") == 6
    assert symbols.count("H") == 6
    # 21 occupied MOs (benzene: 42 electrons / 2)
    assert orb.homo_idx == 20
    assert orb.mo_occupations[orb.homo_idx] == pytest.approx(2.0)
    assert orb.mo_occupations[orb.homo_idx + 1] == pytest.approx(0.0)


def test_pyscf_benzene_cube_grid_params():
    """benzene_mo_12.cube: 80³ orthogonal grid with exact step sizes and origin."""
    from moltui.parsers import parse_cube_data

    cube = parse_cube_data(_require(DATA / "pyscf" / "benzene_mo_12.cube"))

    assert cube.n_points == (80, 80, 80)
    # Step sizes along each axis (in Bohr)
    np.testing.assert_allclose(cube.axes[0, 0], 0.178664, atol=1e-6)
    np.testing.assert_allclose(cube.axes[1, 1], 0.194548, atol=1e-6)
    np.testing.assert_allclose(cube.axes[2, 2], 0.075949, atol=1e-6)
    # Grid is orthogonal
    np.testing.assert_allclose(cube.axes[0, 1:], 0.0, atol=1e-10)
    np.testing.assert_allclose(cube.axes[1, ::2], 0.0, atol=1e-10)
    np.testing.assert_allclose(cube.axes[2, :2], 0.0, atol=1e-10)
    # Origin (in Bohr)
    np.testing.assert_allclose(cube.origin, [-7.057242, -7.684631, -3.0], atol=1e-5)
    # 12 atoms in cube
    assert len(cube.molecule.atoms) == 12


def test_pyscf_benzene_molden_cube_mo_agreement():
    """Evaluating MO 12 (0-indexed) from benzene.molden reproduces benzene_mo_12.cube."""
    from moltui.gto import eval_gto
    from moltui.molden import load_molden_data
    from moltui.parsers import parse_cube_data

    cube = parse_cube_data(_require(DATA / "pyscf" / "benzene_mo_12.cube"))
    orb = load_molden_data(_require(DATA / "pyscf" / "benzene.molden"))
    basis = orb._basis

    nx, ny, nz = cube.n_points
    ix, iy, iz = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    grid_points = (
        cube.origin
        + ix[..., None] * cube.axes[0]
        + iy[..., None] * cube.axes[1]
        + iz[..., None] * cube.axes[2]
    ).reshape(-1, 3)

    ao = eval_gto(basis.shells, grid_points, basis.spherical)
    mo_vals = (ao @ basis.mo_coefficients[:, 12]).reshape(nx, ny, nz)

    # Correct for a possible global sign flip
    ref_idx = np.argmax(np.abs(cube.data))
    if np.sign(mo_vals.flat[ref_idx]) != np.sign(cube.data.flat[ref_idx]):
        mo_vals = -mo_vals

    np.testing.assert_allclose(mo_vals, cube.data, atol=1e-4, rtol=1e-3)


# ---------------------------------------------------------------------------
# TrexIO: txt / HDF5 / molden geometry and MO agreement
# ---------------------------------------------------------------------------


def test_trexio_n2_geometry():
    """txt, HDF5, and molden TREXIO sources all give the same N2 geometry."""
    pytest.importorskip("trexio")
    from moltui.molden import load_molden_data
    from moltui.trexio_support import load_molecule_from_trexio

    mol_txt = load_molecule_from_trexio(_require(TREXIO_DATA / "n2"))
    mol_h5 = load_molecule_from_trexio(_require(TREXIO_DATA / "n2.h5"))
    mol_molden = load_molden_data(_require(TREXIO_DATA / "n2.molden")).molecule

    for mol in (mol_txt, mol_h5, mol_molden):
        assert len(mol.atoms) == 2
        assert all(a.element.symbol == "N" for a in mol.atoms)
        # N2 bond from N 0 0 0; N 0 0 1.098 in main.py → 2.07491928 Bohr
        np.testing.assert_allclose(_bond(mol), 2.07491928477244 * BOHR, atol=1e-5)

    # All three formats agree on positions
    for mol in (mol_h5, mol_molden):
        for i in range(2):
            np.testing.assert_allclose(mol.atoms[i].position, mol_txt.atoms[i].position, atol=1e-5)


def test_trexio_mo_data():
    """MO counts, energies, and density matrices agree across txt, HDF5, and molden."""
    pytest.importorskip("trexio")
    from moltui.molden import load_molden_data
    from moltui.trexio_molden import load_trexio_orbital_data

    orb_txt = load_trexio_orbital_data(_require(TREXIO_DATA / "n2"))
    orb_h5 = load_trexio_orbital_data(_require(TREXIO_DATA / "n2.h5"))
    orb_molden = load_molden_data(_require(TREXIO_DATA / "n2.molden"))

    assert orb_txt is not None
    assert orb_h5 is not None

    # All sources: 110 MOs (cc-pvqz basis), HOMO at index 6
    for orb in (orb_txt, orb_h5, orb_molden):
        assert orb.n_mos == 110
        assert orb.homo_idx == 6

    # Molden MO energies (exact from file)
    np.testing.assert_allclose(
        orb_molden.mo_energies[:5],
        [-15.68219196, -15.67865699, -1.469969434, -0.7785386818, -0.6336316255],
        atol=1e-8,
    )
    np.testing.assert_allclose(orb_molden.mo_energies[orb_molden.homo_idx], -0.613128476, atol=1e-8)
    np.testing.assert_allclose(orb_molden.mo_occupations[:7], [2.0] * 7)
    assert orb_molden.mo_occupations[7] == pytest.approx(0.0)

    # Symmetry labels agree across all three backends (all 110 MOs)
    for orb in (orb_txt, orb_h5):
        assert orb.mo_symmetries == orb_molden.mo_symmetries

    # Density matrices agree across all three backends
    def _density(orb):
        C = orb._basis.mo_coefficients
        return C @ np.diag(orb.mo_occupations) @ C.T

    P_txt = _density(orb_txt)
    P_h5 = _density(orb_h5)
    P_molden = _density(orb_molden)

    np.testing.assert_allclose(P_txt, P_h5, atol=1e-8)
    np.testing.assert_allclose(P_txt, P_molden, atol=1e-4)
