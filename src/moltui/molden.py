from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .elements import Atom, Molecule, get_element
from .gto import GtoBasis, eval_gto, parse_molden, prepare_gto_cache
from .parsers import BOHR_TO_ANGSTROM, CubeData


@dataclass
class OrbitalData:
    """GTO basis, MO metadata, and optional normal modes (any source: Molden file, TREXIO, …)."""

    molecule: Molecule
    mo_energies: np.ndarray
    mo_occupations: np.ndarray
    mo_symmetries: list[str]
    mo_spins: list[str]
    n_mos: int
    homo_idx: int
    _basis: GtoBasis = field(repr=False)
    has_mo_energies: bool = True
    has_mo_occupations: bool = True
    mode_frequencies: np.ndarray | None = None  # (n_modes,)
    normal_modes: np.ndarray | None = None  # (n_modes, n_atoms, 3) in Angstrom
    # AO evaluation caches — populated lazily by evaluate_mo, keyed by (grid_shape, padding).
    # _ao_cache_values stores the float32 AO matrix (npts, nao); halves memory vs float64.
    _gto_prepared_cache: tuple[list, np.ndarray] | None = field(default=None, repr=False)
    _ao_cache_key: tuple[tuple[int, int, int], float] | None = field(default=None, repr=False)
    _ao_cache_values: np.ndarray | None = field(default=None, repr=False)

    @classmethod
    def from_gto_basis(
        cls,
        basis: GtoBasis,
        molecule: Molecule,
        *,
        mode_frequencies: np.ndarray | None = None,
        normal_modes: np.ndarray | None = None,
    ) -> OrbitalData:
        """Build from a :class:`~moltui.gto.GtoBasis` and matching ``molecule``."""
        n_mos = basis.mo_coefficients.shape[1] if basis.mo_coefficients.ndim == 2 else 0
        occ_indices = np.where(basis.mo_occupations > 0.5)[0] if n_mos > 0 else np.array([])
        homo_idx = int(occ_indices[-1]) if len(occ_indices) > 0 else 0
        return cls(
            molecule=molecule,
            mo_energies=basis.mo_energies,
            mo_occupations=basis.mo_occupations,
            mo_symmetries=basis.mo_symmetries,
            mo_spins=basis.mo_spins,
            n_mos=n_mos,
            homo_idx=homo_idx,
            _basis=basis,
            mode_frequencies=mode_frequencies,
            normal_modes=normal_modes,
        )


def parse_molden_atoms(filepath: str | Path) -> Molecule:
    """Parse just the atoms from a molden file."""
    filepath = Path(filepath)
    atoms = []
    in_atoms = False
    is_angstrom = False

    with open(filepath) as f:
        for line in f:
            if "[Atoms]" in line:
                in_atoms = True
                is_angstrom = "Angs" in line
                continue
            if line.startswith("[") and in_atoms:
                break
            if in_atoms and line.strip():
                parts = line.split()
                symbol = parts[0]
                x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
                if not is_angstrom:
                    x *= BOHR_TO_ANGSTROM
                    y *= BOHR_TO_ANGSTROM
                    z *= BOHR_TO_ANGSTROM
                atoms.append(Atom(element=get_element(symbol), position=np.array([x, y, z])))

    mol = Molecule(atoms=atoms, bonds=[])
    mol.detect_bonds()
    return mol


def load_molden_data(filepath: str | Path) -> OrbitalData:
    """Load full molden data including MO coefficients."""
    filepath = Path(filepath)
    basis = parse_molden(filepath)
    if basis.atom_coords_bohr.size == 0 or not basis.atom_symbols:
        raise ValueError("Molden file does not contain atomic coordinates")

    # Build Molecule
    atoms = []
    for i, symbol in enumerate(basis.atom_symbols):
        coord = basis.atom_coords_bohr[i] * BOHR_TO_ANGSTROM
        atoms.append(Atom(element=get_element(symbol), position=coord))

    molecule = Molecule(atoms=atoms, bonds=[])
    molecule.detect_bonds()

    normal_modes: np.ndarray | None = None
    mode_frequencies: np.ndarray | None = None
    if basis.normal_modes is not None:
        if basis.normal_modes.shape[1] != len(atoms):
            raise ValueError("Normal-mode vectors do not match atom count")
        normal_modes = basis.normal_modes * BOHR_TO_ANGSTROM
        if basis.frequencies is not None:
            mode_frequencies = basis.frequencies[: normal_modes.shape[0]]

    return OrbitalData.from_gto_basis(
        basis,
        molecule,
        mode_frequencies=mode_frequencies,
        normal_modes=normal_modes,
    )


def evaluate_mo(
    orbital_data: OrbitalData,
    mo_index: int,
    grid_shape: tuple[int, int, int] = (60, 60, 60),
    padding: float = 5.0,
) -> CubeData:
    """Evaluate a molecular orbital on a 3D grid. Returns CubeData.

    The AO matrix (npts, nao) is cached as float32 on orbital_data after the
    first call for a given grid_shape/padding.  Subsequent calls for the same
    grid (different MO index) skip the expensive GTO evaluation and only do a
    float32 matrix-vector multiply — roughly 100x faster.

    TODO: for very large grids (e.g. 160³ export) the float32 AO matrix can
    exceed available RAM (~5 GiB for 32 atoms at 160³).  If that becomes a
    problem, add chunked z-slice evaluation via eval_mo_direct so the full
    matrix is never materialised.
    """
    if orbital_data.n_mos == 0:
        raise ValueError("No molecular orbitals are available in this data")
    basis = orbital_data._basis
    coords = basis.atom_coords_bohr

    padding_bohr = padding / BOHR_TO_ANGSTROM
    min_c = coords.min(axis=0) - padding_bohr
    max_c = coords.max(axis=0) + padding_bohr

    nx, ny, nz = grid_shape

    # Lazily build the prepared-shell cache (normalisations + deduped centres).
    if orbital_data._gto_prepared_cache is None:
        orbital_data._gto_prepared_cache = prepare_gto_cache(basis.shells, basis.spherical)

    cache_key = (grid_shape, float(padding))
    if orbital_data._ao_cache_values is not None and orbital_data._ao_cache_key == cache_key:
        ao = orbital_data._ao_cache_values
    else:
        x = np.linspace(min_c[0], max_c[0], nx)
        y = np.linspace(min_c[1], max_c[1], ny)
        z = np.linspace(min_c[2], max_c[2], nz)
        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
        grid_points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
        # eval_gto returns float32; store directly without an extra copy.
        ao = eval_gto(
            basis.shells,
            grid_points,
            basis.spherical,
            prepared_cache=orbital_data._gto_prepared_cache,
        )
        orbital_data._ao_cache_values = ao
        orbital_data._ao_cache_key = cache_key

    mo_vec = basis.mo_coefficients[:, mo_index].astype(np.float32, copy=False)
    mo_vals = ao @ mo_vec

    origin = min_c
    axes = np.diag(
        [
            (max_c[0] - min_c[0]) / (nx - 1),
            (max_c[1] - min_c[1]) / (ny - 1),
            (max_c[2] - min_c[2]) / (nz - 1),
        ]
    )

    return CubeData(
        molecule=orbital_data.molecule,
        origin=origin,
        axes=axes,
        n_points=grid_shape,
        data=mo_vals.reshape(grid_shape),
    )
