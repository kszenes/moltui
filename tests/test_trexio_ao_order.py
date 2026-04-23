"""Golden tests for PySCF-style TREXIO spherical AO ordering (no optional deps beyond numpy)."""

from __future__ import annotations

import numpy as np

from moltui.trexio_ao_order import (
    inverse_permutation,
    pyscf_molden_ao_index_order,
    pyscf_trexio_spherical_ao_index_order,
)


def test_spherical_ao_index_golden_small_shells() -> None:
    np.testing.assert_array_equal(pyscf_trexio_spherical_ao_index_order(np.array([1])), [2, 0, 1])
    np.testing.assert_array_equal(
        pyscf_trexio_spherical_ao_index_order(np.array([0, 1])),
        [0, 3, 1, 2],
    )
    np.testing.assert_array_equal(
        pyscf_trexio_spherical_ao_index_order(np.array([0, 1, 2])),
        [0, 3, 1, 2, 6, 7, 5, 8, 4],
    )


def test_molden_ao_index_golden_spherical() -> None:
    """Matches ``pyscf.tools.molden.order_ao_index`` (one contraction per shell)."""
    np.testing.assert_array_equal(
        pyscf_molden_ao_index_order(np.array([1]), cartesian=False),
        [0, 1, 2],
    )
    np.testing.assert_array_equal(
        pyscf_molden_ao_index_order(np.array([0, 1]), cartesian=False),
        [0, 1, 2, 3],
    )
    np.testing.assert_array_equal(
        pyscf_molden_ao_index_order(np.array([0, 1, 2]), cartesian=False),
        [0, 1, 2, 3, 6, 7, 5, 8, 4],
    )


def test_molden_ao_index_golden_cartesian_d() -> None:
    np.testing.assert_array_equal(
        pyscf_molden_ao_index_order(np.array([2]), cartesian=True),
        [0, 3, 5, 1, 2, 4],
    )


def test_inverse_permutation_roundtrip() -> None:
    idx = pyscf_trexio_spherical_ao_index_order(np.array([0, 1, 2, 3]))
    inv = inverse_permutation(idx)
    n = idx.shape[0]
    assert np.array_equal(idx[inv], np.arange(n))
    assert np.array_equal(inv[idx], np.arange(n))
