from __future__ import annotations

from collections import Counter

import numpy as np


def wrap_fracs(fracs: np.ndarray) -> np.ndarray:
    return fracs - np.floor(fracs)


def fractional_positions_from_molecule(mol) -> np.ndarray:
    inv = np.linalg.inv(mol.lattice)
    return np.array([atom.position @ inv for atom in mol.atoms], dtype=np.float64)


def greedy_periodic_match(a: np.ndarray, b: np.ndarray, tol: float = 1e-3) -> bool:
    if len(a) != len(b):
        return False
    used = np.zeros(len(b), dtype=bool)
    for pa in a:
        best = -1
        best_d = float("inf")
        for i, pb in enumerate(b):
            if used[i]:
                continue
            delta = pa - pb
            delta -= np.round(delta)
            dist = float(np.linalg.norm(delta))
            if dist < best_d:
                best = i
                best_d = dist
        if best < 0 or best_d >= tol:
            return False
        used[best] = True
    return bool(np.all(used))


def assert_molecule_matches_ase_atoms(mol, atoms, tol: float = 1e-3) -> None:
    assert Counter(a.element.symbol for a in mol.atoms) == Counter(atoms.get_chemical_symbols())
    np.testing.assert_allclose(mol.lattice, np.asarray(atoms.cell.array), atol=tol)

    mol_frac = wrap_fracs(fractional_positions_from_molecule(mol))
    ase_frac = wrap_fracs(atoms.get_scaled_positions(wrap=False))
    mol_symbols = [a.element.symbol for a in mol.atoms]
    ase_symbols = atoms.get_chemical_symbols()
    for sym in set(mol_symbols):
        lhs = np.array([mol_frac[i] for i, s in enumerate(mol_symbols) if s == sym])
        rhs = np.array([ase_frac[i] for i, s in enumerate(ase_symbols) if s == sym])
        assert greedy_periodic_match(lhs, rhs, tol=tol), f"fractional coordinates differ for {sym}"
