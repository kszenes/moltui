from __future__ import annotations

from pathlib import Path

import numpy as np

from moltui.gto import eval_gto, parse_molden
from moltui.molden import evaluate_mo, load_molden_data, parse_molden_atoms


def _write_molden(tmp_path: Path, contents: str) -> Path:
    path = tmp_path / "sample.molden"
    path.write_text(contents)
    return path


def test_eval_gto_cartesian_d_shell(tmp_path: Path) -> None:
    """Cartesian d-shells (default without [5D]) evaluate without index errors."""
    path = _write_molden(
        tmp_path,
        """[Molden Format]
[Atoms] AU
H 1 1 0.0 0.0 0.0
[GTO]
1 0
d 1 1.0
1.0 1.0

[MO]
Sym= A1
Ene= -0.1
Spin= Alpha
Occup= 2.0
1 0.10
2 0.20
3 0.30
4 0.40
5 0.50
6 0.60
""",
    )

    basis = parse_molden(path)
    ao = eval_gto(basis.shells, np.array([[0.1, 0.2, 0.3]]), basis.spherical)

    assert ao.shape == (1, 6)


def test_parse_molden_mos_without_symmetry_blocks(tmp_path: Path) -> None:
    """Some Molden writers start each orbital with Ene= and omit Sym=."""
    path = _write_molden(
        tmp_path,
        """[Molden Format]
[Atoms] AU
H 1 1 0.0 0.0 0.0
[GTO]
1 0
s 1 1.0
1.0 1.0

[MO]
Ene= -0.5
Spin= Alpha
Occup= 2.0
1 1.0
Ene= 0.2
Spin= Alpha
Occup= 0.0
1 0.5
""",
    )

    basis = parse_molden(path)

    assert basis.mo_coefficients.shape == (1, 2)
    np.testing.assert_allclose(basis.mo_energies, [-0.5, 0.2])
    assert basis.mo_symmetries == ["A", "A"]


def test_parse_molden_accepts_h_shells(tmp_path: Path) -> None:
    path = _write_molden(
        tmp_path,
        """[Molden Format]
[Atoms] AU
H 1 1 0.0 0.0 0.0
[GTO]
1 0
h 1 1.0
1.0 1.0
""",
    )

    basis = parse_molden(path)

    assert basis.shells[0].l == 5


def test_parse_molden_spherical_flags(tmp_path: Path) -> None:
    """Extended Molden flags should update spherical/cartesian behavior."""
    path = _write_molden(
        tmp_path,
        """[Molden Format]
[ATOMS] AU
H 1 1 0.0 0.0 0.0
[GTO]
1 0
s 1 1.0
1.0 1.0

[5D10F]
[9G]
[MO]
Sym= A1
Ene= -0.1
Spin= Alpha
Occup= 2.0
1 1.0
""",
    )

    basis = parse_molden(path)

    assert basis.spherical[2] is True
    assert basis.spherical[3] is False
    assert basis.spherical[4] is True


def test_parse_molden_reads_vibrational_sections(tmp_path: Path) -> None:
    """Vibrational sections should be parsed alongside MO data."""
    path = _write_molden(
        tmp_path,
        """[Molden Format]
[Atoms] Angs
H 1 1 0.0 0.0 0.0
[GTO]
1 0
s 1 1.0
1.0 1.0

[FREQ]
1000.0
[FR-COORD]
H 0.0 0.0 0.0
[FR-NORM-COORD]
vibration 1
0.1 0.0 0.0

[MO]
Sym= A1
Ene= -0.1
Spin= Alpha
Occup= 2.0
1 1.0
""",
    )

    basis = parse_molden(path)

    assert basis.mo_coefficients.shape == (1, 1)
    assert basis.atom_coords_bohr.shape == (1, 3)
    assert basis.frequencies is not None
    assert basis.normal_modes is not None
    assert np.isclose(basis.frequencies[0], 1000.0)
    assert basis.normal_modes.shape == (1, 1, 3)


def test_eval_gto_cartesian_g_shell(tmp_path: Path) -> None:
    """Cartesian g-shells (default without [9G]) should evaluate to 15 AOs."""
    path = _write_molden(
        tmp_path,
        """[Molden Format]
[Atoms] AU
H 1 1 0.0 0.0 0.0
[GTO]
1 0
g 1 1.0
1.0 1.0

[MO]
Sym= A1
Ene= -0.1
Spin= Alpha
Occup= 2.0
1 0.10
2 0.20
3 0.30
4 0.40
5 0.50
6 0.60
7 0.70
8 0.80
9 0.90
10 1.00
11 1.10
12 1.20
13 1.30
14 1.40
15 1.50
""",
    )

    basis = parse_molden(path)
    ao = eval_gto(basis.shells, np.array([[0.1, 0.2, 0.3]]), basis.spherical)

    assert ao.shape == (1, 15)


def test_eval_gto_spherical_g_shell(tmp_path: Path) -> None:
    """Spherical g-shells ([9G]) should evaluate to 9 AOs."""
    path = _write_molden(
        tmp_path,
        """[Molden Format]
[Atoms] AU
H 1 1 0.0 0.0 0.0
[GTO]
1 0
g 1 1.0
1.0 1.0

[9G]
[MO]
Sym= A1
Ene= -0.1
Spin= Alpha
Occup= 2.0
1 0.10
2 0.20
3 0.30
4 0.40
5 0.50
6 0.60
7 0.70
8 0.80
9 0.90
""",
    )

    basis = parse_molden(path)
    ao = eval_gto(basis.shells, np.array([[0.1, 0.2, 0.3]]), basis.spherical)

    assert ao.shape == (1, 9)


def test_load_molden_data_accepts_normal_modes_only_file(tmp_path: Path) -> None:
    """Normal-modes-only Molden files should load geometry and mode vectors."""
    path = _write_molden(
        tmp_path,
        """[MOLDEN FORMAT]
[N_FREQ]
9
[FREQ]
0.0
0.0
0.0
0.0
0.0
0.0
1626.44
3645.22
3751.11
[NATOM]
3
[FR-COORD]
O -0.57 -0.43 0.30
H -0.02 1.27 0.76
H 0.60 -0.85 -1.06
[FR-NORM-COORD]
vibration 1
0.0 0.0 0.0
0.0 0.0 0.0
0.0 0.0 0.0
[INT]
0.0
""",
    )

    data = load_molden_data(path)
    assert len(data.molecule.atoms) == 3
    assert data.n_mos == 0
    assert data.normal_modes is not None
    assert data.normal_modes.shape[0] == 1
    assert data.mode_frequencies is not None
    assert np.isclose(data.mode_frequencies[-1], 0.0)


def _write_two_mo_molden(tmp_path: Path) -> Path:
    """Minimal molden with two s-shells and two MOs for cache testing."""
    return _write_molden(
        tmp_path,
        """[Molden Format]
[Atoms] AU
H 1 1 0.0 0.0 0.0
H 2 1 0.0 0.0 1.4
[GTO]
1 0
s 1 1.0
1.24 1.0

2 0
s 1 1.0
1.24 1.0

[MO]
Sym= A1
Ene= -0.5
Spin= Alpha
Occup= 2.0
1 0.7071
2 0.7071
Sym= A1
Ene= 0.3
Spin= Alpha
Occup= 0.0
1  0.7071
2 -0.7071
""",
    )


def test_ao_cache_populated_after_first_evaluate_mo(tmp_path: Path) -> None:
    """AO matrix cache should be None before the first call and float32 after."""
    data = load_molden_data(_write_two_mo_molden(tmp_path))

    assert data._ao_cache_values is None
    assert data._gto_prepared_cache is None

    evaluate_mo(data, 0, grid_shape=(10, 10, 10))

    assert data._ao_cache_values is not None
    assert data._ao_cache_values.dtype == np.float32
    assert data._ao_cache_values.shape == (1000, data._basis.mo_coefficients.shape[0])
    assert data._gto_prepared_cache is not None


def test_ao_cache_hit_produces_same_mo_values(tmp_path: Path) -> None:
    """MO values from a cache hit must match a fresh evaluation for all MOs."""
    data = load_molden_data(_write_two_mo_molden(tmp_path))
    shape = (12, 12, 12)

    # First call populates cache; compute MO 0 without cache for reference.
    ref0 = evaluate_mo(data, 0, grid_shape=shape)
    ref1 = evaluate_mo(data, 1, grid_shape=shape)  # cache hit

    # Reset cache and recompute both MOs independently.
    data._ao_cache_values = None
    data._ao_cache_key = None
    fresh0 = evaluate_mo(data, 0, grid_shape=shape)
    data._ao_cache_values = None
    data._ao_cache_key = None
    fresh1 = evaluate_mo(data, 1, grid_shape=shape)

    np.testing.assert_allclose(ref0.data, fresh0.data, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(ref1.data, fresh1.data, rtol=1e-5, atol=1e-7)


def test_ao_cache_invalidated_on_grid_change(tmp_path: Path) -> None:
    """Changing grid_shape must trigger a fresh AO matrix computation."""
    data = load_molden_data(_write_two_mo_molden(tmp_path))

    evaluate_mo(data, 0, grid_shape=(8, 8, 8))
    first_cache = data._ao_cache_values
    assert data._ao_cache_key == ((8, 8, 8), 5.0)

    evaluate_mo(data, 0, grid_shape=(10, 10, 10))
    assert data._ao_cache_values is not first_cache
    assert data._ao_cache_key == ((10, 10, 10), 5.0)


def test_parse_molden_openmolcas_style_gto_and_atom_labels(tmp_path: Path) -> None:
    """OpenMolcas uses one integer per GTO center (no trailing 0) and atom names like C1."""
    path = _write_molden(
        tmp_path,
        """[Molden Format]
[Atoms] AU
C1 6 12.0 0.0 0.0 0.0
H2 1 1.0 0.0 0.0 1.0
[GTO]
1
s 1
1.0 1.0
2
s 1
1.0 1.0

[MO]
Sym= A1
Ene= -0.5
Spin= Alpha
Occup= 2.0
1 0.7
2 0.7
""",
    )

    basis = parse_molden(path)
    assert basis.atom_symbols == ["C1", "H2"]
    assert len(basis.shells) == 2
    assert np.allclose(basis.shells[0].center, [0.0, 0.0, 0.0])
    assert np.allclose(basis.shells[1].center, [0.0, 0.0, 1.0])
    assert basis.mo_coefficients.shape == (2, 1)

    data = load_molden_data(path)
    assert data.molecule.atoms[0].element.symbol == "C"
    assert data.molecule.atoms[1].element.symbol == "H"

    mol = parse_molden_atoms(path)
    assert mol.atoms[0].element.symbol == "C"
    assert mol.atoms[1].element.symbol == "H"
