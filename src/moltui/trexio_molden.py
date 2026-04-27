"""Build :class:`molden.OrbitalData` from TREXIO when basis and MO coefficients are present."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import numpy as np

from .elements import Atom, Molecule, get_element
from .gto import GtoBasis, PrimShell, _n_components
from .molden import OrbitalData
from .parsers import BOHR_TO_ANGSTROM
from .trexio_ao_order import (
    inverse_permutation,
    pyscf_molden_ao_index_order,
    pyscf_trexio_spherical_ao_index_order,
)
from .trexio_support import (
    _suppress_stderr_fd,
    is_trexio_path,
    read_trexio_nucleus_elements,
    trexio_backend_for_path,
)


def load_trexio_orbital_data(filepath: str | Path) -> OrbitalData | None:
    """Load TREXIO as :class:`OrbitalData` when basis and MOs exist; else ``None``."""
    try:
        import trexio
    except ImportError:
        return None

    path = Path(filepath)
    if not is_trexio_path(path):
        return None

    backend = trexio_backend_for_path(path)
    guard = _suppress_stderr_fd() if backend == trexio.TREXIO_HDF5 else nullcontext()

    try:
        with guard:
            with trexio.File(str(path), "r", back_end=backend) as f:
                if not _trexio_has_mo_basis(f, trexio):
                    return None
                basis, missing_fields = _trexio_file_to_gto_basis(f, trexio)
    except trexio.Error:
        return None

    atoms = [
        Atom(element=get_element(sym), position=c * BOHR_TO_ANGSTROM)
        for sym, c in zip(basis.atom_symbols, basis.atom_coords_bohr, strict=False)
    ]
    molecule = Molecule(atoms=atoms, bonds=[])
    molecule.detect_bonds()

    od = OrbitalData.from_gto_basis(basis, molecule)
    od.has_mo_energies = "mo_energy" not in missing_fields
    od.has_mo_occupations = "mo_occupation" not in missing_fields
    return od


def _trexio_has_mo_basis(f, trexio) -> bool:
    if not (
        trexio.has_mo_num(f)
        and trexio.has_mo_coefficient(f)
        and trexio.has_basis_shell_num(f)
        and trexio.has_basis_prim_num(f)
        and trexio.has_basis_nucleus_index(f)
        and trexio.has_basis_shell_ang_mom(f)
        and trexio.has_basis_shell_index(f)
        and trexio.has_basis_exponent(f)
        and trexio.has_basis_coefficient(f)
        and trexio.has_nucleus_num(f)
        and trexio.has_nucleus_coord(f)
    ):
        return False
    if int(trexio.read_mo_num(f)) <= 0 or int(trexio.read_basis_shell_num(f)) <= 0:
        return False
    if trexio.has_basis_r_power(f):
        rp = np.asarray(trexio.read_basis_r_power(f), dtype=np.int64)
        if np.any(rp != 0):
            return False
    if trexio.has_mo_coefficient_im(f):
        im = np.asarray(trexio.read_mo_coefficient_im(f), dtype=np.float64)
        if np.any(np.abs(im) > 1e-12):
            return False
    if trexio.has_basis_type(f):
        btype = str(trexio.read_basis_type(f)).lower()
        if "gauss" not in btype:
            return False
    return True


def _trexio_file_to_gto_basis(f, trexio) -> tuple[GtoBasis, list[str]]:
    n_nucl = int(trexio.read_nucleus_num(f))
    coords_bohr = np.asarray(trexio.read_nucleus_coord(f), dtype=np.float64)
    if coords_bohr.shape != (n_nucl, 3):
        raise ValueError("Invalid nucleus.coord shape in TREXIO file.")

    elements = read_trexio_nucleus_elements(f, trexio, n_nucl)
    atom_symbols = [e.symbol for e in elements]

    n_shell = int(trexio.read_basis_shell_num(f))
    n_prim = int(trexio.read_basis_prim_num(f))

    nucleus_idx = np.asarray(trexio.read_basis_nucleus_index(f), dtype=np.int64).reshape(-1)
    shell_ang = np.asarray(trexio.read_basis_shell_ang_mom(f), dtype=np.int64).reshape(-1)
    if nucleus_idx.min() >= 1:
        nucleus_idx = nucleus_idx - 1
    if np.any(nucleus_idx < 0) or np.any(nucleus_idx >= n_nucl):
        raise ValueError("Invalid basis.nucleus_index in TREXIO file.")

    shell_idx = np.asarray(trexio.read_basis_shell_index(f), dtype=np.int64).reshape(-1)
    if int(shell_idx.min()) >= 1:
        shell_idx = shell_idx - 1
    if shell_idx.shape[0] != n_prim or np.any(shell_idx < 0) or np.any(shell_idx >= n_shell):
        raise ValueError("Invalid basis.shell_index in TREXIO file.")

    exponent = np.asarray(trexio.read_basis_exponent(f), dtype=np.float64).reshape(-1)
    coefficient = np.asarray(trexio.read_basis_coefficient(f), dtype=np.float64).reshape(-1)
    if exponent.shape[0] != n_prim or coefficient.shape[0] != n_prim:
        raise ValueError("Mismatch between basis.prim_num and exponent/coefficient arrays.")

    if trexio.has_ao_cartesian(f):
        use_cartesian = int(trexio.read_ao_cartesian(f)) == 1
    else:
        use_cartesian = False
    spherical = {2: not use_cartesian, 3: not use_cartesian, 4: not use_cartesian}

    shells: list[PrimShell] = []
    for s in range(n_shell):
        prim_mask = np.where(shell_idx == s)[0]
        if prim_mask.size == 0:
            raise ValueError(f"TREXIO basis shell {s} has no primitives.")
        ia = int(nucleus_idx[s])
        center = coords_bohr[ia].copy()
        l = int(shell_ang[s])
        if l < 0 or l > 4:
            raise ValueError(f"Unsupported angular momentum l={l} in TREXIO basis.")
        exps = exponent[prim_mask]
        # Use ``basis.coefficient`` only: PySCF/Molden-consistent exports store the same
        # contracted primitive weights as in Molden; ``prim_factor`` / ``shell_factor``
        # are auxiliary normalizations for integral backends, not extra Molden scaling.
        coeffs = coefficient[prim_mask]
        shells.append(PrimShell(center=center, l=l, exponents=exps, coefficients=coeffs))

    nao_expected = sum(_n_components(sh.l, spherical.get(sh.l, sh.l <= 1)) for sh in shells)
    nmo = int(trexio.read_mo_num(f))
    mo_c = np.asarray(trexio.read_mo_coefficient(f), dtype=np.float64)
    if mo_c.ndim != 2:
        raise ValueError("TREXIO mo.coefficient must be a 2D array.")
    # TREXIO row-major layout is [mo.num, ao.num]; GtoBasis uses (nao, nmo).
    if mo_c.shape[0] == nmo and mo_c.shape[1] == nao_expected:
        mo_c = mo_c.T
    elif mo_c.shape[0] == nao_expected and mo_c.shape[1] == nmo:
        pass
    else:
        raise ValueError(
            f"TREXIO MO coefficient shape {mo_c.shape} does not match basis "
            f"(nao={nao_expected}, nmo={nmo})."
        )

    # PySCF TREXIO export: rows in TREX AO order (spherical only); undo to PySCF native.
    if not use_cartesian:
        idx = pyscf_trexio_spherical_ao_index_order(shell_ang)
        if idx.shape[0] != mo_c.shape[0]:
            raise ValueError("Internal error: AO index permutation length does not match nao.")
        mo_c = mo_c[inverse_permutation(idx), :]

    # PySCF Molden export uses a different AO shuffle than TREXIO (``order_ao_index``):
    # e.g. spherical p stays native in Molden but TREXIO permutes to Pz,Px,Py. Map native
    # rows to the same order ``parse_molden`` uses for a PySCF-generated Molden file.
    molden_rows = pyscf_molden_ao_index_order(shell_ang, cartesian=use_cartesian)
    if molden_rows.shape[0] != mo_c.shape[0]:
        raise ValueError("Internal error: Molden AO permutation length does not match nao.")
    mo_c = mo_c[molden_rows, :]

    missing_fields: list[str] = []

    has_mo_energy = trexio.has_mo_energy(f)
    if not has_mo_energy:
        missing_fields.append("mo_energy")
    mo_e = (
        np.asarray(trexio.read_mo_energy(f), dtype=np.float64).reshape(-1)
        if has_mo_energy
        else np.zeros(nmo, dtype=np.float64)
    )
    if mo_e.size < nmo:
        mo_e = np.concatenate([mo_e, np.zeros(nmo - mo_e.size, dtype=np.float64)])
    elif mo_e.size > nmo:
        mo_e = mo_e[:nmo]

    has_mo_occ = trexio.has_mo_occupation(f)
    if not has_mo_occ:
        missing_fields.append("mo_occupation")
    mo_occ = (
        np.asarray(trexio.read_mo_occupation(f), dtype=np.float64).reshape(-1)
        if has_mo_occ
        else np.zeros(nmo, dtype=np.float64)
    )
    if mo_occ.size < nmo:
        mo_occ = np.concatenate([mo_occ, np.zeros(nmo - mo_occ.size, dtype=np.float64)])
    elif mo_occ.size > nmo:
        mo_occ = mo_occ[:nmo]
    if trexio.has_mo_symmetry(f):
        mo_sym = [str(x).strip() for x in trexio.read_mo_symmetry(f)]
        if len(mo_sym) < nmo:
            mo_sym = mo_sym + ["A"] * (nmo - len(mo_sym))
        elif len(mo_sym) > nmo:
            mo_sym = mo_sym[:nmo]
    else:
        mo_sym = ["A"] * nmo
    if trexio.has_mo_spin(f):
        spins_i = np.asarray(trexio.read_mo_spin(f), dtype=np.int64).reshape(-1)
        mo_spins = ["Alpha" if int(x) == 0 else "Beta" for x in spins_i]
        if len(mo_spins) < nmo:
            mo_spins = mo_spins + ["Alpha"] * (nmo - len(mo_spins))
        elif len(mo_spins) > nmo:
            mo_spins = mo_spins[:nmo]
    else:
        mo_spins = ["Alpha"] * nmo

    basis = GtoBasis(
        atom_symbols=atom_symbols,
        atom_coords_bohr=coords_bohr,
        shells=shells,
        mo_energies=mo_e,
        mo_occupations=mo_occ,
        mo_coefficients=mo_c,
        mo_symmetries=mo_sym,
        mo_spins=mo_spins,
        spherical=spherical,
        frequencies=None,
        normal_modes=None,
    )
    return basis, missing_fields
