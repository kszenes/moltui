"""Equivalence tests for the Gaussian formatted-checkpoint parser.

Each fchk in ``data/gaussian/`` has a sibling ``.molden`` (see that directory's
README for provenance). The parsers should agree on geometry, MO energies, and
the value of each MO at arbitrary points in space.

Comparing on a grid (rather than coefficient-by-coefficient) is the right
abstraction because the two formats use different AO orderings and any
generalized contractions in the fchk get split into segmented shells in the
molden. Only the *physical* MO ought to match.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from moltui.gto import (
    component_permutation,
    eval_gto,
    gaussian_cartesian_component_labels,
    molden_cartesian_component_labels,
    prepare_gto_cache,
    pure_spherical_component_labels,
)

DATA = Path(__file__).resolve().parent.parent / "data" / "gaussian"

PAIRS = [
    "h2o_sto3g",
    "ch3_hf_sto3g",
    "water_ccpvdz_pure_hf_g03",
    "o2_cc_pvtz_cart",
]


def _eval_basis_on_points(basis, points_bohr: np.ndarray) -> np.ndarray:
    """Return MO values shape (npts, nmo) for ``basis`` at ``points_bohr``."""
    cache = prepare_gto_cache(basis.shells, basis.spherical)
    ao = eval_gto(basis.shells, points_bohr, basis.spherical, prepared_cache=cache)
    return ao.astype(np.float64) @ basis.mo_coefficients


def _sample_points(coords_bohr: np.ndarray, n: int = 24, seed: int = 0) -> np.ndarray:
    """Random points in the molecule's bbox plus a small padding (Bohr units)."""
    rng = np.random.default_rng(seed)
    pad = 2.0  # Bohr
    lo = coords_bohr.min(axis=0) - pad
    hi = coords_bohr.max(axis=0) + pad
    return rng.uniform(lo, hi, size=(n, 3))


@pytest.fixture(scope="module")
def fchk_module():
    return pytest.importorskip("moltui.fchk")


def _fchk_scalar(label: str, type_code: str, value: int | float) -> str:
    if type_code == "I":
        return f"{label:<40}   {type_code}   {value:12d}\n"
    return f"{label:<40}   {type_code}   {value:16.8E}\n"


def _fchk_array(label: str, type_code: str, values: list[int] | np.ndarray) -> str:
    arr = np.asarray(values)
    lines = [f"{label:<40}   {type_code}   N={arr.size:12d}\n"]
    per_line = 6 if type_code == "I" else 5
    for start in range(0, arr.size, per_line):
        chunk = arr[start : start + per_line]
        if type_code == "I":
            lines.append("".join(f"{int(v):12d}" for v in chunk) + "\n")
        else:
            lines.append("".join(f"{float(v):16.8E}" for v in chunk) + "\n")
    return "".join(lines)


def _shell_nbasis(shell_type: int) -> int:
    l = abs(shell_type)
    return 2 * l + 1 if shell_type < 0 else (l + 1) * (l + 2) // 2


def _write_minimal_shell_fchk(
    path: Path,
    *,
    shell_type: int,
    n_alpha: int = 0,
    n_beta: int = 0,
    include_pure_flags: bool = False,
    mo_coefficients: np.ndarray | None = None,
) -> None:
    """Write a tiny one-atom fchk with a single shell."""
    nbasis = _shell_nbasis(shell_type)
    nmo = nbasis
    if mo_coefficients is None:
        mo_coefficients = np.eye(nbasis)
    # fchk stores MO-major coefficients; tests pass coefficients in AO-major
    # shape (nbasis, nmo), matching the parser's post-reshape layout.
    mo_coeff = mo_coefficients.T.reshape(-1)
    text = "test\nSP HF/STO-3G\n"
    text += _fchk_scalar("Number of atoms", "I", 1)
    text += _fchk_scalar("Number of alpha electrons", "I", n_alpha)
    text += _fchk_scalar("Number of beta electrons", "I", n_beta)
    text += _fchk_scalar("Number of basis functions", "I", nbasis)
    if include_pure_flags:
        text += _fchk_scalar("Pure/Cartesian d shells", "I", 0 if shell_type < 0 else 1)
        text += _fchk_scalar("Pure/Cartesian f shells", "I", 0)
    text += _fchk_array("Atomic numbers", "I", [8])
    text += _fchk_array("Current cartesian coordinates", "R", [0.0, 0.0, 0.0])
    text += _fchk_array("Shell types", "I", [shell_type])
    text += _fchk_array("Number of primitives per shell", "I", [1])
    text += _fchk_array("Shell to atom map", "I", [1])
    text += _fchk_array("Primitive exponents", "R", [1.0])
    text += _fchk_array("Contraction coefficients", "R", [1.0])
    text += _fchk_array("Alpha Orbital Energies", "R", np.arange(nmo, dtype=float))
    text += _fchk_array("Alpha MO coefficients", "R", mo_coeff)
    path.write_text(text)


@pytest.mark.parametrize(("shell_type", "expected_spherical"), [(-2, True), (2, False)])
def test_missing_pure_cartesian_flags_are_inferred_from_shell_types(
    tmp_path: Path, fchk_module, shell_type: int, expected_spherical: bool
) -> None:
    path = tmp_path / "missing-pure-flags.fchk"
    _write_minimal_shell_fchk(path, shell_type=shell_type)

    basis = fchk_module.parse_fchk(path)

    assert basis.spherical[2] is expected_spherical
    assert basis.mo_coefficients.shape[0] == (5 if expected_spherical else 6)


def test_restricted_open_shell_occupations_are_doubly_then_singly_occupied(
    tmp_path: Path, fchk_module
) -> None:
    path = tmp_path / "rohf.fchk"
    _write_minimal_shell_fchk(path, shell_type=-2, n_alpha=5, n_beta=4, include_pure_flags=True)

    basis = fchk_module.parse_fchk(path)

    np.testing.assert_array_equal(basis.mo_occupations, [2.0, 2.0, 2.0, 2.0, 1.0])


def test_cartesian_component_labels_match_iodata_fchk_convention() -> None:
    assert gaussian_cartesian_component_labels(4) == [
        "zzzz",
        "yzzz",
        "yyzz",
        "yyyz",
        "yyyy",
        "xzzz",
        "xyzz",
        "xyyz",
        "xyyy",
        "xxzz",
        "xxyz",
        "xxyy",
        "xxxz",
        "xxxy",
        "xxxx",
    ]
    assert gaussian_cartesian_component_labels(5)[:6] == [
        "zzzzz",
        "yzzzz",
        "yyzzz",
        "yyyzz",
        "yyyyz",
        "yyyyy",
    ]
    assert len(gaussian_cartesian_component_labels(9)) == 55
    assert pure_spherical_component_labels(9) == [
        "c0",
        "c1",
        "s1",
        "c2",
        "s2",
        "c3",
        "s3",
        "c4",
        "s4",
        "c5",
        "s5",
        "c6",
        "s6",
        "c7",
        "s7",
        "c8",
        "s8",
        "c9",
        "s9",
    ]


def test_fchk_cartesian_g_coefficients_are_reordered_to_moltui_convention(
    tmp_path: Path, fchk_module
) -> None:
    path = tmp_path / "cartesian-g.fchk"
    fchk_labels = gaussian_cartesian_component_labels(4)
    moltui_labels = molden_cartesian_component_labels(4)
    coeff = np.eye(len(fchk_labels))
    _write_minimal_shell_fchk(path, shell_type=4, include_pure_flags=True, mo_coefficients=coeff)

    basis = fchk_module.parse_fchk(path)

    expected_rows = component_permutation(fchk_labels, moltui_labels)
    np.testing.assert_array_equal(basis.mo_coefficients, coeff[expected_rows, :])


def _write_minimal_trajectory_fchk(
    path: Path, *, prefix: str = "Opt point", counts: list[int] | None = None
) -> np.ndarray:
    if counts is None:
        counts = [2, 1]
    z = [1, 8]
    frames_bohr = []
    text = "trajectory\nFOpt HF/STO-3G\n"
    text += _fchk_scalar("Number of atoms", "I", len(z))
    text += _fchk_array("Atomic numbers", "I", z)
    for frame_idx in range(sum(counts)):
        frames_bohr.append(
            np.array(
                [
                    [float(frame_idx), 0.0, 0.0],
                    [float(frame_idx), 0.0, 1.0],
                ]
            )
        )
    text += _fchk_array("Current cartesian coordinates", "R", frames_bohr[-1].reshape(-1))
    count_label = (
        "IRC Number of geometries" if prefix == "IRC point" else "Optimization Number of geometries"
    )
    text += _fchk_array(count_label, "I", counts)
    offset = 0
    for point_idx, n_frames in enumerate(counts, start=1):
        point_frames = np.array(frames_bohr[offset : offset + n_frames])
        text += _fchk_array(f"{prefix} {point_idx:7d} Geometries", "R", point_frames.reshape(-1))
        offset += n_frames
    path.write_text(text)
    return np.array(frames_bohr)


@pytest.mark.parametrize("prefix", ["Opt point", "IRC point"])
def test_fchk_trajectory_parser_flattens_point_geometries(
    tmp_path: Path, fchk_module, prefix: str
) -> None:
    path = tmp_path / "trajectory.fchk"
    frames_bohr = _write_minimal_trajectory_fchk(path, prefix=prefix)

    trajectory = fchk_module.parse_fchk_trajectory(path)

    assert trajectory.frames.shape == (3, 2, 3)
    np.testing.assert_allclose(trajectory.frames, frames_bohr * 0.529177249)
    np.testing.assert_allclose(trajectory.molecule.atoms[0].position, trajectory.frames[0, 0])


def test_fchk_trajectory_parser_reports_missing_trajectory(tmp_path: Path, fchk_module) -> None:
    path = tmp_path / "not-trajectory.fchk"
    _write_minimal_shell_fchk(path, shell_type=0)

    with pytest.raises(ValueError, match="does not contain"):
        fchk_module.parse_fchk_trajectory(path)


@pytest.mark.parametrize("stem", PAIRS)
def test_geometry_matches_molden(stem: str, fchk_module) -> None:
    from moltui.molden import load_molden_data

    fchk_data = fchk_module.load_fchk_data(DATA / f"{stem}.fchk")
    molden_data = load_molden_data(DATA / f"{stem}.molden")

    assert len(fchk_data.molecule.atoms) == len(molden_data.molecule.atoms)
    for a, b in zip(fchk_data.molecule.atoms, molden_data.molecule.atoms):
        assert a.element.symbol == b.element.symbol
        assert np.allclose(a.position, b.position, atol=1e-6)


@pytest.mark.parametrize("stem", PAIRS)
def test_mo_energies_match_molden(stem: str, fchk_module) -> None:
    from moltui.molden import load_molden_data

    fchk_data = fchk_module.load_fchk_data(DATA / f"{stem}.fchk")
    molden_data = load_molden_data(DATA / f"{stem}.molden")

    assert fchk_data.mo_energies.shape == molden_data.mo_energies.shape
    np.testing.assert_allclose(fchk_data.mo_energies, molden_data.mo_energies, atol=1e-6)


@pytest.mark.parametrize("stem", PAIRS)
def test_mo_values_match_molden_on_grid(stem: str, fchk_module) -> None:
    """The strongest test: parsed basis + MOs must produce the same wavefunction."""
    from moltui.molden import load_molden_data

    fchk_data = fchk_module.load_fchk_data(DATA / f"{stem}.fchk")
    molden_data = load_molden_data(DATA / f"{stem}.molden")

    coords_bohr = fchk_data._basis.atom_coords_bohr
    points = _sample_points(coords_bohr)

    fchk_vals = _eval_basis_on_points(fchk_data._basis, points)
    molden_vals = _eval_basis_on_points(molden_data._basis, points)

    assert fchk_vals.shape == molden_vals.shape
    # MOs in the paired molden are the same coefficients as in the fchk (no
    # re-diagonalisation), so they should agree element-wise up to float
    # precision. atol stays loose because GTO values vary wildly near nuclei.
    np.testing.assert_allclose(fchk_vals, molden_vals, atol=1e-5, rtol=1e-4)


def test_normal_modes_match_molden(fchk_module) -> None:
    """Vib-Modes / Vib-E2 must agree with the paired molden's freq sections.

    Mode eigenvectors carry an arbitrary sign per mode, so we compare absolute
    values element-wise.
    """
    from moltui.molden import load_molden_data

    fchk_data = fchk_module.load_fchk_data(DATA / "peroxide_tsopt.fchk")
    molden_data = load_molden_data(DATA / "peroxide_tsopt.molden")

    assert fchk_data.normal_modes is not None
    assert molden_data.normal_modes is not None
    assert fchk_data.mode_frequencies is not None
    assert molden_data.mode_frequencies is not None

    np.testing.assert_allclose(fchk_data.mode_frequencies, molden_data.mode_frequencies, atol=1e-3)

    assert fchk_data.normal_modes.shape == molden_data.normal_modes.shape
    np.testing.assert_allclose(
        np.abs(fchk_data.normal_modes), np.abs(molden_data.normal_modes), atol=1e-6
    )


def test_freqs_from_hessian_match_vib_e2(fchk_module) -> None:
    """Hessian-derived frequencies must match Gaussian's own Vib-E2 analysis.

    ``peroxide_tsopt.fchk`` contains both ``Cartesian Force Constants`` and the
    precomputed ``Vib-E2`` frequencies. We force the parser onto the Hessian
    path (mass-weight, project translations + rotations, diagonalise, convert
    to cm^-1) and compare against Vib-E2 as ground truth — same molecule, same
    underlying SCF, so any deviation > a few hundredths of a cm^-1 indicates a
    bug in mass-weighting, projection, unit conversion, or sign handling.
    """
    freqs, _ = fchk_module.compute_freqs_from_hessian(DATA / "peroxide_tsopt.fchk")

    # Read the Vib-E2 oracle the same way the main parser does.
    fchk_data = fchk_module.load_fchk_data(DATA / "peroxide_tsopt.fchk")
    expected = fchk_data.mode_frequencies
    assert expected is not None
    assert freqs.shape == expected.shape
    np.testing.assert_allclose(freqs, expected, atol=0.05)


def test_modes_from_hessian_match_vib_modes(fchk_module) -> None:
    """Hessian-derived eigenvectors must agree with Vib-Modes (per-mode cosine).

    Eigenvectors carry an arbitrary sign per mode and, within any degenerate
    subspace, an arbitrary rotation. ``peroxide_tsopt`` (H2O2 at a TS) has no
    degeneracies among its 6 modes, so a per-mode |cosine| ≈ 1 is achievable
    and is the strongest agreement we can expect.
    """
    _, modes = fchk_module.compute_freqs_from_hessian(DATA / "peroxide_tsopt.fchk")

    fchk_data = fchk_module.load_fchk_data(DATA / "peroxide_tsopt.fchk")
    expected = fchk_data._basis.normal_modes
    assert expected is not None
    assert modes.shape == expected.shape

    n_modes = modes.shape[0]
    for k in range(n_modes):
        a = modes[k].reshape(-1)
        b = expected[k].reshape(-1)
        cos = abs(float(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        assert cos > 0.999, f"mode {k} cosine {cos:.6f} < 0.999"


def test_hessian_only_fchk_uses_fallback(fchk_module) -> None:
    """``peroxide_irc.fchk`` has no Vib-Modes/Vib-E2 — only the Hessian. The
    parser should auto-fall-back and produce the same output as calling
    ``compute_freqs_from_hessian`` directly."""
    data = fchk_module.load_fchk_data(DATA / "peroxide_irc.fchk")
    assert data.normal_modes is not None
    assert data.mode_frequencies is not None
    assert data.normal_modes.shape[1] == len(data.molecule.atoms)
    assert np.all(np.isfinite(data.mode_frequencies))

    expected_freqs, expected_modes = fchk_module.compute_freqs_from_hessian(
        DATA / "peroxide_irc.fchk"
    )
    np.testing.assert_allclose(data.mode_frequencies, expected_freqs, atol=0.0)
    # ``load_fchk_data`` converts modes from Bohr to Å (as for molden); compare
    # directions via per-mode cosine to dodge that scaling and arbitrary signs.
    for k in range(expected_freqs.shape[0]):
        a = data.normal_modes[k].reshape(-1)
        b = expected_modes[k].reshape(-1)
        cos = abs(float(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        assert cos > 0.9999, f"fallback mode {k} cosine {cos:.6f}"
