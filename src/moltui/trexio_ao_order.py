"""PySCF/TREXIO spherical AO index ordering (pure NumPy; no ``trexio`` dependency)."""

from __future__ import annotations

import numpy as np


def pyscf_trexio_spherical_ao_index_order(shell_ang_moms: np.ndarray) -> np.ndarray:
    """Spherical AO index order PySCF uses when writing TREXIO ``mo.coefficient``.

    Matches ``pyscf.tools.trexio._order_ao_index`` for segmented shells (one contraction
    block per ``mol._bas`` row). PySCF stores ``M[mo, ao] = mo_coeff[idx[ao], mo]``;
    MolTUI's evaluator expects native PySCF/Molden row order, so apply
    ``mo_coeff[inverse(idx), :]`` after reading TREXIO.
    """
    ang = np.asarray(shell_ang_moms, dtype=np.int64).reshape(-1)
    if ang.size == 0:
        return np.zeros(0, dtype=np.int64)
    lmax = int(ang.max())
    cache_by_l: list[np.ndarray] = [
        np.array([0], dtype=np.int64),
        np.array([2, 0, 1], dtype=np.int64),
    ]
    for l in range(2, lmax + 1):
        idx_block = np.empty(l * 2 + 1, dtype=np.int64)
        idx_block[::2] = l - np.arange(0, l + 1, dtype=np.int64)
        idx_block[1::2] = np.arange(l + 1, l + l + 1, dtype=np.int64)
        cache_by_l.append(idx_block)

    blocks: list[np.ndarray] = []
    off = 0
    for l in ang:
        l = int(l)
        blocks.append(cache_by_l[l] + off)
        off += l * 2 + 1
    return np.concatenate(blocks)


def inverse_permutation(p: np.ndarray) -> np.ndarray:
    inv = np.empty_like(p)
    inv[p] = np.arange(p.shape[0], dtype=p.dtype)
    return inv


def pyscf_molden_ao_index_order(shell_ang_moms: np.ndarray, *, cartesian: bool) -> np.ndarray:
    """AO row order PySCF uses when writing Molden ``[MO]`` lines (``order_ao_index``).

    ``result[i]`` is the PySCF-internal AO row index printed as Molden AO ``i+1``.
    So ``mo_molden[i, :] == mo_native[result[i], :]`` and ``mo_native`` can be recovered
    only in combination with TREXIO's separate convention; for MolTUI,
    ``mo_molden = mo_native[result, :]`` maps native coefficients to Molden file row order.
    """
    ang = np.asarray(shell_ang_moms, dtype=np.int64).reshape(-1)
    if ang.size == 0:
        return np.zeros(0, dtype=np.int64)
    idx: list[int] = []
    off = 0
    for l_raw in ang:
        l = int(l_raw)
        if cartesian:
            if l == 2:
                idx.extend([off + 0, off + 3, off + 5, off + 1, off + 2, off + 4])
            elif l == 3:
                idx.extend(
                    [
                        off + 0,
                        off + 6,
                        off + 9,
                        off + 3,
                        off + 1,
                        off + 2,
                        off + 5,
                        off + 8,
                        off + 7,
                        off + 4,
                    ]
                )
            elif l == 4:
                idx.extend(
                    [
                        off + 0,
                        off + 10,
                        off + 14,
                        off + 1,
                        off + 2,
                        off + 6,
                        off + 11,
                        off + 9,
                        off + 13,
                        off + 3,
                        off + 5,
                        off + 12,
                        off + 4,
                        off + 7,
                        off + 8,
                    ]
                )
            elif l > 4:
                raise ValueError(f"Molden AO order: unsupported cartesian l={l}")
            else:
                ncart = (l + 1) * (l + 2) // 2
                idx.extend(range(off, off + ncart))
            off += (l + 1) * (l + 2) // 2
        else:
            if l == 2:
                idx.extend([off + 2, off + 3, off + 1, off + 4, off + 0])
            elif l == 3:
                idx.extend([off + 3, off + 4, off + 2, off + 5, off + 1, off + 6, off + 0])
            elif l == 4:
                idx.extend(
                    [
                        off + 4,
                        off + 5,
                        off + 3,
                        off + 6,
                        off + 2,
                        off + 7,
                        off + 1,
                        off + 8,
                        off + 0,
                    ]
                )
            elif l > 4:
                raise ValueError(f"Molden AO order: unsupported spherical l={l}")
            else:
                idx.extend(range(off, off + l * 2 + 1))
            off += l * 2 + 1
    return np.asarray(idx, dtype=np.int64)
