#!/usr/bin/env python3
"""Tests for QC-input geometry parsers and their detection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from moltui.app import _detect_filetype
from moltui.parsers import load_molecule
from moltui.qc_inputs import (
    BOHR_TO_ANGSTROM,
    parse_gaussian_input,
    parse_molcas_input,
    parse_molpro_input,
    parse_nwchem_input,
    parse_orca_input,
    parse_psi4_input,
    parse_qchem_input,
    parse_turbomole_coord,
    sniff_qc_input,
)

WATER_SYMBOLS = ["O", "H", "H"]
WATER_COORDS_ANG = np.array(
    [
        [0.000000, 0.000000, 0.117790],
        [0.000000, 0.755453, -0.471161],
        [0.000000, -0.755453, -0.471161],
    ]
)


def _assert_water(mol, atol: float = 1e-4) -> None:
    assert [a.element.symbol for a in mol.atoms] == WATER_SYMBOLS
    coords = np.array([a.position for a in mol.atoms])
    assert coords.shape == (3, 3)
    np.testing.assert_allclose(coords, WATER_COORDS_ANG, atol=atol)


# ---------------------------------------------------------------------------
# Orca
# ---------------------------------------------------------------------------

ORCA_INPUT = """! BP86 def2-SVP Opt
%pal nprocs 4 end

* xyz 0 1
O    0.000000   0.000000   0.117790
H    0.000000   0.755453  -0.471161
H    0.000000  -0.755453  -0.471161
*
"""


def test_orca_input_parse(tmp_path: Path):
    p = tmp_path / "water.inp"
    p.write_text(ORCA_INPUT)
    _assert_water(parse_orca_input(p))


def test_orca_input_detected(tmp_path: Path):
    p = tmp_path / "water.inp"
    p.write_text(ORCA_INPUT)
    assert _detect_filetype(str(p)) == "orca-input"


ORCA_INT_INPUT = """! HF STO-3G
* int 0 1
 O 0 0 0   0.0    0.0    0.0
 H 1 0 0   0.957  0.0    0.0
 H 1 2 0   0.957  104.5  0.0
*
"""

ORCA_GZMT_INPUT = """! HF STO-3G
* gzmt 0 1
O
H 1 0.957
H 1 0.957 2 104.5
*
"""


def test_orca_int_zmatrix(tmp_path: Path):
    p = tmp_path / "water.inp"
    p.write_text(ORCA_INT_INPUT)
    mol = parse_orca_input(p)
    assert [a.element.symbol for a in mol.atoms] == ["O", "H", "H"]
    coords = np.array([a.position for a in mol.atoms])
    # OH bonds should be ~0.957 Å
    np.testing.assert_allclose(np.linalg.norm(coords[1] - coords[0]), 0.957, atol=1e-3)
    np.testing.assert_allclose(np.linalg.norm(coords[2] - coords[0]), 0.957, atol=1e-3)
    # HOH angle ≈ 104.5°
    v1 = coords[1] - coords[0]
    v2 = coords[2] - coords[0]
    cos_ang = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    assert abs(np.degrees(np.arccos(cos_ang)) - 104.5) < 1e-2


def test_orca_gzmt_zmatrix(tmp_path: Path):
    p = tmp_path / "water.inp"
    p.write_text(ORCA_GZMT_INPUT)
    mol = parse_orca_input(p)
    assert [a.element.symbol for a in mol.atoms] == ["O", "H", "H"]
    coords = np.array([a.position for a in mol.atoms])
    np.testing.assert_allclose(np.linalg.norm(coords[1] - coords[0]), 0.957, atol=1e-3)
    np.testing.assert_allclose(np.linalg.norm(coords[2] - coords[0]), 0.957, atol=1e-3)


def test_orca_int_bohr_distances(tmp_path: Path):
    """`! Bohrs` should scale Z-matrix bond lengths from a.u. to Å."""
    body = (
        "! HF STO-3G Bohrs\n"
        "* int 0 1\n"
        " O 0 0 0   0.0      0.0    0.0\n"
        " H 1 0 0   1.808846 0.0    0.0\n"
        " H 1 2 0   1.808846 104.5  0.0\n"
        "*\n"
    )
    # 1.808846 Bohr ≈ 0.957 Å
    p = tmp_path / "water.inp"
    p.write_text(body)
    mol = parse_orca_input(p)
    coords = np.array([a.position for a in mol.atoms])
    np.testing.assert_allclose(np.linalg.norm(coords[1] - coords[0]), 0.957, atol=1e-3)


def test_orca_paras_block_rejected(tmp_path: Path):
    body = """! HF
%paras
   B1 = 1.39
end
* int 0 1
 C 0 0 0 0.0   0.0   0.0
 C 1 0 0 {B1}  0.0   0.0
*
"""
    p = tmp_path / "x.inp"
    p.write_text(body)
    with pytest.raises(ValueError, match=r"%paras"):
        parse_orca_input(p)


def test_orca_coords_block_rejected(tmp_path: Path):
    body = """! HF
%coords
 ctyp xyz
 charge 0
 mult 1
 coords
   O 0.0 0.0 0.0
   H 0.957 0.0 0.0
   H 0.0 0.957 0.0
 end
end
"""
    p = tmp_path / "x.inp"
    p.write_text(body)
    with pytest.raises(ValueError, match=r"%coords"):
        parse_orca_input(p)


def test_orca_compound_block_rejected(tmp_path: Path):
    body = """%compound
   for r from 0.8 to 1.2 by 0.1 do
     # ...
   endfor
end
"""
    p = tmp_path / "x.inp"
    p.write_text(body)
    with pytest.raises(ValueError, match=r"%compound"):
        parse_orca_input(p)


def test_orca_xyzfile_external(tmp_path: Path):
    xyz = tmp_path / "geom.xyz"
    xyz.write_text(
        "3\nwater\n"
        "O    0.000000   0.000000   0.117790\n"
        "H    0.000000   0.755453  -0.471161\n"
        "H    0.000000  -0.755453  -0.471161\n"
    )
    inp = tmp_path / "job.inp"
    inp.write_text("! HF\n* xyzfile 0 1 geom.xyz\n")
    _assert_water(parse_orca_input(inp))


def test_orca_xyzfile_missing(tmp_path: Path):
    p = tmp_path / "x.inp"
    p.write_text("! HF\n* xyzfile 0 1 missing.xyz\n")
    with pytest.raises(ValueError, match="not found"):
        parse_orca_input(p)


# ---------------------------------------------------------------------------
# Q-Chem
# ---------------------------------------------------------------------------

QCHEM_INPUT = """$molecule
0 1
O    0.000000   0.000000   0.117790
H    0.000000   0.755453  -0.471161
H    0.000000  -0.755453  -0.471161
$end

$rem
   METHOD   HF
   BASIS    sto-3g
$end
"""


def test_qchem_input_parse(tmp_path: Path):
    p = tmp_path / "water.in"
    p.write_text(QCHEM_INPUT)
    _assert_water(parse_qchem_input(p))


def test_qchem_input_detected(tmp_path: Path):
    p = tmp_path / "water.qcin"
    p.write_text(QCHEM_INPUT)
    assert _detect_filetype(str(p)) == "qchem-input"


def test_qchem_zmatrix_with_variables(tmp_path: Path):
    body = """$rem
   METHOD HF
   BASIS sto-3g
$end

$molecule
0 1
O
H 1 R1
H 1 R1 2 A1

R1=0.957
A1=104.5
$end
"""
    p = tmp_path / "water.in"
    p.write_text(body)
    mol = parse_qchem_input(p)
    assert [a.element.symbol for a in mol.atoms] == ["O", "H", "H"]
    coords = np.array([a.position for a in mol.atoms])
    np.testing.assert_allclose(np.linalg.norm(coords[1] - coords[0]), 0.957, atol=1e-3)
    np.testing.assert_allclose(np.linalg.norm(coords[2] - coords[0]), 0.957, atol=1e-3)
    v1 = coords[1] - coords[0]
    v2 = coords[2] - coords[0]
    cos_ang = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    assert abs(np.degrees(np.arccos(cos_ang)) - 104.5) < 1e-2


def test_qchem_input_detected_from_in(tmp_path: Path):
    p = tmp_path / "water.in"
    p.write_text(QCHEM_INPUT)
    assert _detect_filetype(str(p)) == "qchem-input"


# ---------------------------------------------------------------------------
# Gaussian
# ---------------------------------------------------------------------------

GAUSSIAN_INPUT = """%mem=2GB
%nproc=4
%chk=water.chk
# HF/6-31G(d) Opt

water single point

0 1
O    0.000000   0.000000   0.117790
H    0.000000   0.755453  -0.471161
H    0.000000  -0.755453  -0.471161

"""


def test_gaussian_input_parse(tmp_path: Path):
    p = tmp_path / "water.gjf"
    p.write_text(GAUSSIAN_INPUT)
    _assert_water(parse_gaussian_input(p))


def test_gaussian_input_detected_gjf(tmp_path: Path):
    p = tmp_path / "water.gjf"
    p.write_text(GAUSSIAN_INPUT)
    assert _detect_filetype(str(p)) == "gaussian-input"


def test_gaussian_input_detected_com(tmp_path: Path):
    p = tmp_path / "water.com"
    p.write_text(GAUSSIAN_INPUT)
    assert _detect_filetype(str(p)) == "gaussian-input"


def test_gaussian_link1_rejected(tmp_path: Path):
    body = """%mem=1GB
# HF/STO-3G

job 1

0 1
H 0 0 0
H 0 0 0.74

--Link1--
%mem=1GB
# HF/STO-3G Geom=Check Guess=Read

job 2

0 1
"""
    p = tmp_path / "x.gjf"
    p.write_text(body)
    with pytest.raises(ValueError, match=r"--Link1--"):
        parse_gaussian_input(p)


def test_gaussian_input_with_freeze_flag(tmp_path: Path):
    body = """%mem=1GB
# HF/STO-3G

title

0 1
O 0   0.000000   0.000000   0.117790
H 0   0.000000   0.755453  -0.471161
H -1  0.000000  -0.755453  -0.471161

"""
    p = tmp_path / "frozen.gjf"
    p.write_text(body)
    _assert_water(parse_gaussian_input(p))


# ---------------------------------------------------------------------------
# NWChem
# ---------------------------------------------------------------------------

NWCHEM_INPUT = """echo
title "water"
start water

geometry units angstroms
  O    0.000000   0.000000   0.117790
  H    0.000000   0.755453  -0.471161
  H    0.000000  -0.755453  -0.471161
end

basis
  * library STO-3G
end

task scf
"""


def test_nwchem_input_parse(tmp_path: Path):
    p = tmp_path / "water.nw"
    p.write_text(NWCHEM_INPUT)
    _assert_water(parse_nwchem_input(p))


def test_nwchem_input_detected(tmp_path: Path):
    p = tmp_path / "water.nw"
    p.write_text(NWCHEM_INPUT)
    assert _detect_filetype(str(p)) == "nwchem-input"


def test_nwchem_input_bohr_units(tmp_path: Path):
    body = """start water
geometry units bohr
  O    0.000000   0.000000   0.222595
  H    0.000000   1.427521  -0.890345
  H    0.000000  -1.427521  -0.890345
end
"""
    p = tmp_path / "water.nw"
    p.write_text(body)
    mol = parse_nwchem_input(p)
    coords = np.array([a.position for a in mol.atoms])
    np.testing.assert_allclose(coords, WATER_COORDS_ANG, atol=1e-3)


# ---------------------------------------------------------------------------
# Turbomole
# ---------------------------------------------------------------------------

TURBOMOLE_COORD = """$coord
    0.000000000000    0.000000000000    0.222595000000      o
    0.000000000000    1.427521000000   -0.890345000000      h
    0.000000000000   -1.427521000000   -0.890345000000      h
$end
"""


def test_turbomole_coord_parse(tmp_path: Path):
    p = tmp_path / "coord"
    p.write_text(TURBOMOLE_COORD)
    mol = parse_turbomole_coord(p)
    coords = np.array([a.position for a in mol.atoms])
    np.testing.assert_allclose(coords, WATER_COORDS_ANG, atol=1e-3)
    assert [a.element.symbol for a in mol.atoms] == WATER_SYMBOLS


def test_turbomole_coord_detected(tmp_path: Path):
    p = tmp_path / "coord"
    p.write_text(TURBOMOLE_COORD)
    assert _detect_filetype(str(p)) == "turbomole-input"


def test_turbomole_coord_angs_directive(tmp_path: Path):
    body = """$coord angs
    0.000000   0.000000   0.117790  o
    0.000000   0.755453  -0.471161  h
    0.000000  -0.755453  -0.471161  h
$end
"""
    p = tmp_path / "coord.tmp"
    p.write_text(body)
    _assert_water(parse_turbomole_coord(p))


# ---------------------------------------------------------------------------
# Molcas
# ---------------------------------------------------------------------------

MOLCAS_INLINE = """&GATEWAY
Coord
3
water
O    0.000000   0.000000   0.117790
H    0.000000   0.755453  -0.471161
H    0.000000  -0.755453  -0.471161
Basis
sto-3g
Group
C1
&SEWARD
&SCF
"""


def test_molcas_inline_parse(tmp_path: Path):
    p = tmp_path / "water.input"
    p.write_text(MOLCAS_INLINE)
    _assert_water(parse_molcas_input(p))


def test_molcas_input_detected(tmp_path: Path):
    p = tmp_path / "water.input"
    p.write_text(MOLCAS_INLINE)
    assert _detect_filetype(str(p)) == "molcas-input"


def test_molcas_external_coord(tmp_path: Path):
    xyz = tmp_path / "water.xyz"
    xyz.write_text(
        "3\nwater\n"
        "O    0.000000   0.000000   0.117790\n"
        "H    0.000000   0.755453  -0.471161\n"
        "H    0.000000  -0.755453  -0.471161\n"
    )
    inp = tmp_path / "job.input"
    inp.write_text("&GATEWAY\nCoord = water.xyz\nBasis = sto-3g\nGroup = C1\n&SEWARD\n&SCF\n")
    _assert_water(parse_molcas_input(inp))


# ---------------------------------------------------------------------------
# Molpro
# ---------------------------------------------------------------------------

MOLPRO_XYZ_STYLE = """***, water
memory,30,m
geometry={
3
water
O    0.000000   0.000000   0.117790
H    0.000000   0.755453  -0.471161
H    0.000000  -0.755453  -0.471161
}
basis=cc-pvdz
hf
"""


def test_molpro_input_parse_xyz_style(tmp_path: Path):
    p = tmp_path / "water.com"
    p.write_text(MOLPRO_XYZ_STYLE)
    _assert_water(parse_molpro_input(p))


def test_molpro_input_detected(tmp_path: Path):
    p = tmp_path / "water.com"
    p.write_text(MOLPRO_XYZ_STYLE)
    assert _detect_filetype(str(p)) == "molpro-input"


def test_molpro_cartesian_angstrom(tmp_path: Path):
    body = """***, water
angstrom
geometry={
O,    0.000000,   0.000000,   0.117790
H,    0.000000,   0.755453,  -0.471161
H,    0.000000,  -0.755453,  -0.471161
}
hf
"""
    p = tmp_path / "water.inp"
    p.write_text(body)
    _assert_water(parse_molpro_input(p))


def test_molpro_cartesian_bohr_default(tmp_path: Path):
    bohr = WATER_COORDS_ANG / BOHR_TO_ANGSTROM
    lines = [f"{sym} {x:.10f} {y:.10f} {z:.10f}" for sym, (x, y, z) in zip(WATER_SYMBOLS, bohr)]
    body = "***, water\ngeometry={\n" + "\n".join(lines) + "\n}\nhf\n"
    p = tmp_path / "water.inp"
    p.write_text(body)
    _assert_water(parse_molpro_input(p), atol=1e-3)


# ---------------------------------------------------------------------------
# Psi4
# ---------------------------------------------------------------------------


def test_psi4_dynamic_zmatrix_variables_error(tmp_path: Path):
    body = """Rvals = [0.9, 1.0, 1.1]
Avals = range(102, 106, 2)

molecule h2o {
    O
    H 1 R
    H 1 R 2 A
}

for R in Rvals:
    h2o.R = R
    for A in Avals:
        h2o.A = A
"""
    p = tmp_path / "scan.dat"
    p.write_text(body)
    with pytest.raises(ValueError, match="dynamic molecule variables are not supported"):
        parse_psi4_input(p)


# ---------------------------------------------------------------------------
# load_molecule integration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filename, content",
    [
        ("water.inp", ORCA_INPUT),
        ("water.qcin", QCHEM_INPUT),
        ("water.gjf", GAUSSIAN_INPUT),
        ("water.nw", NWCHEM_INPUT),
        ("coord", TURBOMOLE_COORD),
        ("water.input", MOLCAS_INLINE),
        ("water.com", MOLPRO_XYZ_STYLE),
    ],
)
def test_load_molecule_dispatches_qc_inputs(tmp_path: Path, filename: str, content: str):
    p = tmp_path / filename
    p.write_text(content)
    mol = load_molecule(p)
    assert len(mol.atoms) == 3
    assert [a.element.symbol for a in mol.atoms] == WATER_SYMBOLS


def test_sniff_returns_none_for_random_text(tmp_path: Path):
    p = tmp_path / "random.txt"
    p.write_text("just some random text\nwithout any markers\n")
    assert sniff_qc_input(p) is None


# ---------------------------------------------------------------------------
# Sniffer accumulator path: weak-signal-only inputs
# ---------------------------------------------------------------------------


def test_sniff_accumulator_orca_keyword_lines_only(tmp_path: Path):
    """An Orca file with keyword lines but no `* xyz` block in the prefix."""
    p = tmp_path / "x.inp"
    p.write_text("! HF DEF2-SVP\n! TightSCF Opt\n! NoFrozenCore\n")
    assert sniff_qc_input(p) == "orca-input"


def test_sniff_accumulator_gaussian_link0_only(tmp_path: Path):
    """Gaussian Link 0 directives with no route line yet."""
    p = tmp_path / "x.com"
    p.write_text("%mem=2GB\n%nprocshared=4\n%chk=foo.chk\n")
    assert sniff_qc_input(p) == "gaussian-input"


def test_sniff_accumulator_nwchem_top_directives(tmp_path: Path):
    """NWChem top-level `start`/`title`/`echo` with no `geometry` block."""
    p = tmp_path / "x.in"
    p.write_text('start mymol\ntitle "preflight"\necho\n')
    assert sniff_qc_input(p) == "nwchem-input"


def test_sniff_accumulator_majority_wins(tmp_path: Path):
    """Mixed weak signals — most common one wins."""
    p = tmp_path / "x.inp"
    # 3 orca-style lines and 1 nwchem-style "title" line — orca should win.
    p.write_text("! HF\n! Opt\n! TightSCF\ntitle some\n")
    assert sniff_qc_input(p) == "orca-input"


# ---------------------------------------------------------------------------
# Format-specific quirks identified from upstream docs
# ---------------------------------------------------------------------------


def test_nwchem_atom_tag_inferred_element(tmp_path: Path):
    """NWChem atom tags resolve to elements via prefix-match (Heavy1 → He)."""
    body = """start tagged
geometry units angstroms
  Heavy1   0.0 0.0 0.0
  Cl_a     0.0 0.0 1.0
  Oxy1     0.0 1.0 0.0
end
"""
    p = tmp_path / "tags.nw"
    p.write_text(body)
    mol = parse_nwchem_input(p)
    assert [a.element.symbol for a in mol.atoms] == ["He", "Cl", "O"]


def test_gaussian_input_comma_separated_coords(tmp_path: Path):
    body = (
        "%mem=1GB\n# HF/STO-3G\n\ntitle\n\n0,1\n"
        "O,0.000000,0.000000,0.117790\n"
        "H,0.000000,0.755453,-0.471161\n"
        "H,0.000000,-0.755453,-0.471161\n\n"
    )
    p = tmp_path / "water.gjf"
    p.write_text(body)
    _assert_water(parse_gaussian_input(p))


def test_molpro_semicolon_separated_coords(tmp_path: Path):
    body = (
        "***, water\nangstrom\ngeometry={\n"
        "O;0.000000;0.000000;0.117790\n"
        "H;0.000000;0.755453;-0.471161\n"
        "H;0.000000;-0.755453;-0.471161\n"
        "}\nhf\n"
    )
    p = tmp_path / "water.com"
    p.write_text(body)
    _assert_water(parse_molpro_input(p))


def test_molcas_4char_keyword_stem(tmp_path: Path):
    """OpenMolcas matches keywords on their 4-char stem (`Coor` works)."""
    body = (
        "&GATEWAY\nCoor\n3\nwater\n"
        "O    0.000000   0.000000   0.117790\n"
        "H    0.000000   0.755453  -0.471161\n"
        "H    0.000000  -0.755453  -0.471161\n"
        "&SEWARD\n"
    )
    p = tmp_path / "water.input"
    p.write_text(body)
    _assert_water(parse_molcas_input(p))


def test_molcas_inline_bohr_in_comment_line(tmp_path: Path):
    body = "&GATEWAY\nCoord\n3\nwater (bohr)\n" + _bohr_water_block() + "\n&SEWARD\n"
    p = tmp_path / "water.input"
    p.write_text(body)
    _assert_water(parse_molcas_input(p), atol=1e-3)


def test_molcas_inline_au_token_in_comment(tmp_path: Path):
    body = "&GATEWAY\nCoord\n3\nwater a.u.\n" + _bohr_water_block() + "\n&SEWARD\n"
    p = tmp_path / "water.input"
    p.write_text(body)
    _assert_water(parse_molcas_input(p), atol=1e-3)


def test_molcas_inline_trans_token_rejected(tmp_path: Path):
    body = (
        "&GATEWAY\nCoord\n3\nTRANS 1.0 0.0 0.0\n"
        "O    0.000000   0.000000   0.117790\n"
        "H    0.000000   0.755453  -0.471161\n"
        "H    0.000000  -0.755453  -0.471161\n"
        "&SEWARD\n"
    )
    p = tmp_path / "water.input"
    p.write_text(body)
    with pytest.raises(ValueError, match="TRANS"):
        parse_molcas_input(p)


# ---------------------------------------------------------------------------
# Bohr-unit handling
# ---------------------------------------------------------------------------


def _bohr_water_block() -> str:
    bohr = WATER_COORDS_ANG / BOHR_TO_ANGSTROM
    return "\n".join(
        f"{sym}    {x:.10f}   {y:.10f}   {z:.10f}" for sym, (x, y, z) in zip(WATER_SYMBOLS, bohr)
    )


def test_orca_input_bohr_keyword(tmp_path: Path):
    body = f"! HF Bohrs\n* xyz 0 1\n{_bohr_water_block()}\n*\n"
    p = tmp_path / "water.inp"
    p.write_text(body)
    _assert_water(parse_orca_input(p), atol=1e-3)


def test_qchem_input_bohr_rem(tmp_path: Path):
    body = (
        "$molecule\n0 1\n"
        + _bohr_water_block()
        + "\n$end\n\n$rem\n   METHOD HF\n   INPUT_BOHR true\n$end\n"
    )
    p = tmp_path / "water.in"
    p.write_text(body)
    _assert_water(parse_qchem_input(p), atol=1e-3)


def test_gaussian_input_bohr_units(tmp_path: Path):
    body = "%mem=1GB\n# HF/STO-3G Units=Bohr\n\ntitle\n\n0 1\n" + _bohr_water_block() + "\n\n"
    p = tmp_path / "water.gjf"
    p.write_text(body)
    _assert_water(parse_gaussian_input(p), atol=1e-3)


# ---------------------------------------------------------------------------
# Real-world fixtures under data/qc_inputs/
# ---------------------------------------------------------------------------

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "data" / "qc_inputs"

# (relative path, expected detected kind, expected coordinates in Angstrom)
N2_BOND = 1.098
N2_NWCHEM_BOND = 1.08  # nwchem fixture uses 1.08
_N2_COORDS_LONG = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, N2_BOND]])
_N2_COORDS_SHORT = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, N2_NWCHEM_BOND]])
# Z-matrix-derived coords place atom 1 along the X axis instead of Z.
_N2_COORDS_LONG_X = np.array([[0.0, 0.0, 0.0], [N2_BOND, 0.0, 0.0]])


@pytest.mark.parametrize(
    "relpath, expected_kind, expected_coords",
    [
        ("orca.inp", "orca-input", _N2_COORDS_LONG),
        ("qchem.in", "qchem-input", _N2_COORDS_LONG),
        ("gaussian.gjf", "gaussian-input", _N2_COORDS_LONG),
        ("nwchem.nwi", "nwchem-input", _N2_COORDS_SHORT),
        ("turbomole/coord", "turbomole-input", _N2_COORDS_LONG),
        ("molcas.inp", "molcas-input", _N2_COORDS_LONG),
        ("molpro.com", "molpro-input", _N2_COORDS_LONG),
        ("mrcc/MINP", "mrcc-input", _N2_COORDS_LONG),
        ("cfour/ZMAT", "cfour-input", _N2_COORDS_LONG_X),
        ("psi4.dat", "psi4-input", _N2_COORDS_LONG),
        ("gamess.inp", "gamess-input", _N2_COORDS_LONG),
        ("jaguar.in", "jaguar-input", _N2_COORDS_LONG),
    ],
)
def test_real_fixture_detection_and_parsing(
    relpath: str, expected_kind: str, expected_coords: np.ndarray
):
    path = FIXTURE_DIR / relpath
    assert path.exists(), f"missing fixture: {path}"
    assert _detect_filetype(str(path)) == expected_kind
    mol = load_molecule(path)
    assert [a.element.symbol for a in mol.atoms] == ["N", "N"]
    coords = np.array([a.position for a in mol.atoms])
    np.testing.assert_allclose(coords, expected_coords, atol=1e-3)


def test_uppercase_keywords_all_formats(tmp_path: Path):
    """All QC parsers should be case-insensitive on directives/keywords."""
    cases = {
        "water.inp": ORCA_INPUT.upper(),
        "water.qcin": QCHEM_INPUT.upper(),
        "water.gjf": GAUSSIAN_INPUT.upper(),
        "water.nw": NWCHEM_INPUT.upper(),
        "COORD": TURBOMOLE_COORD.upper(),
        "water.input": MOLCAS_INLINE.upper(),
        "water.com": MOLPRO_XYZ_STYLE.upper(),
    }
    for name, content in cases.items():
        p = tmp_path / name
        p.write_text(content)
        mol = load_molecule(p)
        assert [a.element.symbol for a in mol.atoms] == WATER_SYMBOLS, name


def test_molcas_input_bohr_directive(tmp_path: Path):
    body = (
        "&GATEWAY\nBohr\nCoord\n3\nwater\n"
        + _bohr_water_block()
        + "\nBasis\nsto-3g\nGroup\nC1\n&SEWARD\n&SCF\n"
    )
    p = tmp_path / "water.input"
    p.write_text(body)
    _assert_water(parse_molcas_input(p), atol=1e-3)
