from __future__ import annotations

from pathlib import Path

import numpy as np

from moltui.gto import eval_gto, parse_molden


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


def test_parse_molden_ignores_vibrational_sections(tmp_path: Path) -> None:
    """Vibrational sections should be skipped while parsing MO data."""
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
