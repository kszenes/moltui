from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Element:
    symbol: str
    name: str
    atomic_number: int
    cpk_color: tuple[int, int, int]
    covalent_radius: float  # Angstrom
    vdw_radius: float  # Angstrom


ELEMENTS: dict[str, Element] = {
    # Period 1
    "H": Element("H", "Hydrogen", 1, (255, 255, 255), 0.31, 1.20),
    "He": Element("He", "Helium", 2, (217, 255, 255), 0.28, 1.40),
    # Period 2
    "Li": Element("Li", "Lithium", 3, (204, 128, 255), 1.28, 1.82),
    "Be": Element("Be", "Beryllium", 4, (194, 255, 0), 0.96, 1.98),
    "B": Element("B", "Boron", 5, (255, 181, 181), 0.84, 1.92),
    "C": Element("C", "Carbon", 6, (144, 144, 144), 0.76, 1.70),
    "N": Element("N", "Nitrogen", 7, (48, 80, 248), 0.71, 1.55),
    "O": Element("O", "Oxygen", 8, (255, 13, 13), 0.66, 1.52),
    "F": Element("F", "Fluorine", 9, (144, 224, 80), 0.57, 1.47),
    "Ne": Element("Ne", "Neon", 10, (179, 227, 245), 0.58, 1.54),
    # Period 3
    "Na": Element("Na", "Sodium", 11, (171, 92, 242), 1.66, 2.27),
    "Mg": Element("Mg", "Magnesium", 12, (138, 255, 0), 1.41, 1.73),
    "Al": Element("Al", "Aluminum", 13, (191, 166, 166), 1.21, 1.84),
    "Si": Element("Si", "Silicon", 14, (240, 200, 160), 1.11, 2.10),
    "P": Element("P", "Phosphorus", 15, (255, 128, 0), 1.07, 1.80),
    "S": Element("S", "Sulfur", 16, (255, 255, 48), 1.05, 1.80),
    "Cl": Element("Cl", "Chlorine", 17, (31, 240, 31), 1.02, 1.75),
    "Ar": Element("Ar", "Argon", 18, (128, 209, 227), 1.06, 1.88),
    # Period 4
    "K": Element("K", "Potassium", 19, (143, 64, 212), 2.03, 2.75),
    "Ca": Element("Ca", "Calcium", 20, (61, 255, 0), 1.76, 2.31),
    "Sc": Element("Sc", "Scandium", 21, (230, 230, 230), 1.70, 2.15),
    "Ti": Element("Ti", "Titanium", 22, (191, 194, 199), 1.60, 2.11),
    "V": Element("V", "Vanadium", 23, (166, 166, 171), 1.53, 2.07),
    "Cr": Element("Cr", "Chromium", 24, (138, 153, 199), 1.39, 2.06),
    "Mn": Element("Mn", "Manganese", 25, (156, 122, 199), 1.50, 2.05),
    "Fe": Element("Fe", "Iron", 26, (224, 102, 51), 1.42, 2.04),
    "Co": Element("Co", "Cobalt", 27, (240, 144, 160), 1.38, 2.00),
    "Ni": Element("Ni", "Nickel", 28, (80, 208, 80), 1.24, 1.97),
    "Cu": Element("Cu", "Copper", 29, (200, 128, 51), 1.32, 1.96),
    "Zn": Element("Zn", "Zinc", 30, (125, 128, 176), 1.22, 2.01),
    "Ga": Element("Ga", "Gallium", 31, (194, 143, 143), 1.22, 1.87),
    "Ge": Element("Ge", "Germanium", 32, (102, 143, 143), 1.20, 2.11),
    "As": Element("As", "Arsenic", 33, (189, 128, 227), 1.19, 1.85),
    "Se": Element("Se", "Selenium", 34, (255, 161, 0), 1.20, 1.90),
    "Br": Element("Br", "Bromine", 35, (166, 41, 41), 1.20, 1.85),
    "Kr": Element("Kr", "Krypton", 36, (92, 184, 209), 1.16, 2.02),
    # Period 5
    "Rb": Element("Rb", "Rubidium", 37, (112, 46, 176), 2.20, 3.03),
    "Sr": Element("Sr", "Strontium", 38, (0, 255, 0), 1.95, 2.49),
    "Y": Element("Y", "Yttrium", 39, (148, 255, 255), 1.90, 2.19),
    "Zr": Element("Zr", "Zirconium", 40, (148, 224, 224), 1.75, 2.23),
    "Nb": Element("Nb", "Niobium", 41, (115, 194, 201), 1.64, 2.18),
    "Mo": Element("Mo", "Molybdenum", 42, (84, 181, 181), 1.54, 2.17),
    "Tc": Element("Tc", "Technetium", 43, (59, 158, 158), 1.47, 2.16),
    "Ru": Element("Ru", "Ruthenium", 44, (36, 143, 143), 1.46, 2.13),
    "Rh": Element("Rh", "Rhodium", 45, (10, 125, 140), 1.42, 2.10),
    "Pd": Element("Pd", "Palladium", 46, (0, 105, 133), 1.39, 2.10),
    "Ag": Element("Ag", "Silver", 47, (192, 192, 192), 1.45, 2.11),
    "Cd": Element("Cd", "Cadmium", 48, (255, 217, 143), 1.44, 2.18),
    "In": Element("In", "Indium", 49, (166, 117, 115), 1.42, 1.93),
    "Sn": Element("Sn", "Tin", 50, (102, 128, 128), 1.39, 2.17),
    "Sb": Element("Sb", "Antimony", 51, (158, 99, 181), 1.39, 2.06),
    "Te": Element("Te", "Tellurium", 52, (212, 122, 0), 1.38, 2.06),
    "I": Element("I", "Iodine", 53, (148, 0, 148), 1.39, 1.98),
    "Xe": Element("Xe", "Xenon", 54, (66, 158, 176), 1.40, 2.16),
    # Period 6
    "Cs": Element("Cs", "Cesium", 55, (87, 23, 143), 2.44, 3.43),
    "Ba": Element("Ba", "Barium", 56, (0, 201, 0), 2.15, 2.68),
    "La": Element("La", "Lanthanum", 57, (112, 212, 255), 2.07, 2.43),
    "Ce": Element("Ce", "Cerium", 58, (255, 255, 199), 2.04, 2.42),
    "Pr": Element("Pr", "Praseodymium", 59, (217, 255, 199), 2.03, 2.40),
    "Nd": Element("Nd", "Neodymium", 60, (199, 255, 199), 2.01, 2.39),
    "Pm": Element("Pm", "Promethium", 61, (163, 255, 199), 1.99, 2.38),
    "Sm": Element("Sm", "Samarium", 62, (143, 255, 199), 1.98, 2.36),
    "Eu": Element("Eu", "Europium", 63, (97, 255, 199), 1.98, 2.35),
    "Gd": Element("Gd", "Gadolinium", 64, (69, 255, 199), 1.96, 2.34),
    "Tb": Element("Tb", "Terbium", 65, (48, 255, 199), 1.94, 2.33),
    "Dy": Element("Dy", "Dysprosium", 66, (31, 255, 199), 1.92, 2.31),
    "Ho": Element("Ho", "Holmium", 67, (0, 255, 156), 1.92, 2.30),
    "Er": Element("Er", "Erbium", 68, (0, 230, 117), 1.89, 2.29),
    "Tm": Element("Tm", "Thulium", 69, (0, 212, 82), 1.90, 2.27),
    "Yb": Element("Yb", "Ytterbium", 70, (0, 191, 56), 1.87, 2.26),
    "Lu": Element("Lu", "Lutetium", 71, (0, 171, 36), 1.87, 2.24),
    "Hf": Element("Hf", "Hafnium", 72, (77, 194, 255), 1.75, 2.23),
    "Ta": Element("Ta", "Tantalum", 73, (77, 166, 255), 1.70, 2.22),
    "W": Element("W", "Tungsten", 74, (33, 148, 214), 1.62, 2.18),
    "Re": Element("Re", "Rhenium", 75, (38, 125, 171), 1.51, 2.16),
    "Os": Element("Os", "Osmium", 76, (38, 102, 150), 1.44, 2.16),
    "Ir": Element("Ir", "Iridium", 77, (23, 84, 135), 1.41, 2.20),
    "Pt": Element("Pt", "Platinum", 78, (208, 208, 224), 1.36, 2.13),
    "Au": Element("Au", "Gold", 79, (255, 209, 35), 1.36, 2.14),
    "Hg": Element("Hg", "Mercury", 80, (184, 184, 208), 1.32, 2.09),
    "Tl": Element("Tl", "Thallium", 81, (166, 84, 77), 1.45, 1.96),
    "Pb": Element("Pb", "Lead", 82, (87, 89, 97), 1.46, 2.02),
    "Bi": Element("Bi", "Bismuth", 83, (158, 79, 181), 1.48, 2.07),
    "Po": Element("Po", "Polonium", 84, (171, 92, 0), 1.40, 1.97),
    "At": Element("At", "Astatine", 85, (117, 79, 69), 1.50, 2.02),
    "Rn": Element("Rn", "Radon", 86, (66, 130, 150), 1.50, 2.20),
    # Period 7
    "Fr": Element("Fr", "Francium", 87, (66, 0, 102), 2.60, 3.48),
    "Ra": Element("Ra", "Radium", 88, (0, 125, 0), 2.21, 2.83),
    "Ac": Element("Ac", "Actinium", 89, (112, 171, 250), 2.15, 2.47),
    "Th": Element("Th", "Thorium", 90, (0, 186, 255), 2.06, 2.45),
    "Pa": Element("Pa", "Protactinium", 91, (0, 161, 255), 2.00, 2.43),
    "U": Element("U", "Uranium", 92, (0, 143, 255), 1.96, 2.41),
    "Np": Element("Np", "Neptunium", 93, (0, 128, 255), 1.90, 2.39),
    "Pu": Element("Pu", "Plutonium", 94, (0, 107, 255), 1.87, 2.43),
}

DEFAULT_ELEMENT = Element("X", "Unknown", 0, (255, 20, 147), 1.00, 1.70)

ATOMIC_NUMBER_TO_SYMBOL: dict[int, str] = {e.atomic_number: e.symbol for e in ELEMENTS.values()}


def get_element(symbol: str) -> Element:
    # In molden files from Molcas the symbol may contain numbers (eg. C1):
    alpha_only = "".join(c for c in symbol.strip() if c.isalpha())
    key = alpha_only.capitalize()
    return ELEMENTS.get(key, DEFAULT_ELEMENT)


def get_element_by_number(atomic_number: int) -> Element:
    symbol = ATOMIC_NUMBER_TO_SYMBOL.get(atomic_number)
    if symbol:
        return ELEMENTS[symbol]
    return DEFAULT_ELEMENT


@dataclass
class Atom:
    element: Element
    position: np.ndarray  # shape (3,)


@dataclass
class Molecule:
    atoms: list[Atom]
    bonds: list[tuple[int, int]]
    lattice: np.ndarray | None = None  # (3, 3) row-vectors in Angstrom, or None
    # Per-bond integer cell shift for the j atom relative to i; (0, 0, 0) when
    # both ends are in the same cell. Length matches `bonds` when populated.
    bond_shifts: list[tuple[int, int, int]] | None = None

    def center(self) -> np.ndarray:
        if not self.atoms:
            return np.zeros(3)
        positions = np.array([a.position for a in self.atoms])
        return positions.mean(axis=0)

    def radius(self) -> float:
        if not self.atoms:
            return 1.0
        positions = np.array([a.position for a in self.atoms])
        centroid = positions.mean(axis=0)
        distances = np.linalg.norm(positions - centroid, axis=1)
        return float(distances.max()) if len(distances) > 0 else 1.0

    def _bond_displacement(self, bond_index: int) -> np.ndarray:
        if self.bond_shifts is None or self.lattice is None or bond_index >= len(self.bond_shifts):
            return np.zeros(3, dtype=np.float64)
        s0, s1, s2 = self.bond_shifts[bond_index]
        return s0 * self.lattice[0] + s1 * self.lattice[1] + s2 * self.lattice[2]

    def _adjacency(self) -> dict[int, list[int]]:
        adj: dict[int, list[int]] = {i: [] for i in range(len(self.atoms))}
        for i, j in self.bonds:
            adj[i].append(j)
            adj[j].append(i)
        return adj

    def _directed_bond_vectors(self) -> dict[tuple[int, int], np.ndarray]:
        """Map each bonded pair to the directed vector from atom i to atom j."""
        directed: dict[tuple[int, int], np.ndarray] = {}
        for bond_index, (i, j) in enumerate(self.bonds):
            disp = self._bond_displacement(bond_index)
            vec_ij = self.atoms[j].position + disp - self.atoms[i].position
            directed[(i, j)] = vec_ij
            directed[(j, i)] = -vec_ij
        return directed

    def get_bond_lengths(self) -> list[tuple[int, int, float]]:
        results = []
        for k, (i, j) in enumerate(self.bonds):
            disp = self._bond_displacement(k)
            dist = float(np.linalg.norm(self.atoms[j].position + disp - self.atoms[i].position))
            results.append((i, j, dist))
        return results

    def get_angles(self) -> list[tuple[int, int, int, float]]:
        adj = self._adjacency()
        directed = self._directed_bond_vectors()
        results = []
        for j, neighbors in adj.items():
            for ni, i in enumerate(neighbors):
                for k in neighbors[ni + 1 :]:
                    v1 = directed[(j, i)]
                    v2 = directed[(j, k)]
                    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
                    angle = float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))
                    results.append((i, j, k, angle))
        return results

    def get_dihedrals(self) -> list[tuple[int, int, int, int, float]]:
        adj = self._adjacency()
        directed = self._directed_bond_vectors()
        results = []
        seen = set()
        for j, k in self.bonds:
            for i in adj[j]:
                if i == k:
                    continue
                for l in adj[k]:
                    if l == j or l == i:
                        continue
                    key = (i, j, k, l) if i < l else (l, k, j, i)
                    if key in seen:
                        continue
                    seen.add(key)
                    b1 = directed[(i, j)]
                    b2 = directed[(j, k)]
                    b3 = directed[(k, l)]
                    n1 = np.cross(b1, b2)
                    n2 = np.cross(b2, b3)
                    n1_norm = np.linalg.norm(n1) + 1e-10
                    n2_norm = np.linalg.norm(n2) + 1e-10
                    cos_d = np.dot(n1, n2) / (n1_norm * n2_norm)
                    angle = float(np.degrees(np.arccos(np.clip(cos_d, -1.0, 1.0))))
                    results.append((i, j, k, l, angle))
        return results

    def supercell(self, nx: int, ny: int, nz: int) -> "Molecule":
        """Tile the unit cell into an ``nx * ny * nz`` supercell.

        Atom positions are replicated at ``i*a + j*b + k*c`` for
        ``i ∈ [0, nx)``, ``j ∈ [0, ny)``, ``k ∈ [0, nz)``. The returned
        molecule's lattice is scaled accordingly. Bonds are recomputed.
        Returns ``self`` if dimensions are ``(1, 1, 1)`` or no lattice.
        """
        if nx < 1 or ny < 1 or nz < 1:
            raise ValueError("supercell dimensions must be >= 1")
        if (nx, ny, nz) == (1, 1, 1) or self.lattice is None:
            return self

        from itertools import product

        new_atoms: list[Atom] = []
        for i, j, k in product(range(nx), range(ny), range(nz)):
            disp = i * self.lattice[0] + j * self.lattice[1] + k * self.lattice[2]
            for atom in self.atoms:
                new_atoms.append(Atom(element=atom.element, position=atom.position + disp))

        new_lattice = self.lattice.copy()
        new_lattice[0] *= nx
        new_lattice[1] *= ny
        new_lattice[2] *= nz

        replicated = Molecule(atoms=new_atoms, bonds=[], lattice=new_lattice)
        replicated.detect_bonds_periodic()
        return replicated

    def with_bonded_periodic_images(
        self, tol: float = 1e-3, bond_tolerance: float = 1.3
    ) -> "Molecule":
        """Return a copy that also includes periodic-image atoms bonded across cell faces.

        Extends :meth:`with_periodic_images` (boundary replicas) with images of
        any atom within bonding distance of an in-cell atom along any of the
        26 neighboring cell shifts. The image atoms appear as "ghosts" that
        keep boundary-crossing bonds visible.
        """
        if self.lattice is None or not self.atoms:
            return self

        from itertools import product

        base = self.with_periodic_images(tol=tol)

        existing: dict[tuple[int, int, int], int] = {}
        new_atoms: list[Atom] = list(base.atoms)
        for idx, atom in enumerate(new_atoms):
            key = tuple(np.round(atom.position * 1000).astype(int).tolist())
            existing[key] = idx  # type: ignore[assignment]

        # Distance check against the full augmented set so boundary replicas
        # also acquire their cross-cell bonded partners.
        target_positions = np.array([a.position for a in new_atoms], dtype=np.float64)
        target_radii = np.array([a.element.covalent_radius for a in new_atoms], dtype=np.float64)

        for s0, s1, s2 in product((-1, 0, 1), repeat=3):
            if (s0, s1, s2) == (0, 0, 0):
                continue
            disp = s0 * self.lattice[0] + s1 * self.lattice[1] + s2 * self.lattice[2]
            for atom_j in self.atoms:
                shifted = atom_j.position + disp
                deltas = target_positions - shifted
                dist_sq = np.einsum("ij,ij->i", deltas, deltas)
                max_bond = (target_radii + atom_j.element.covalent_radius) * bond_tolerance
                if not np.any(dist_sq < max_bond * max_bond):
                    continue
                key = tuple(np.round(shifted * 1000).astype(int).tolist())
                if key in existing:  # type: ignore[comparison-overlap]
                    continue
                existing[key] = len(new_atoms)  # type: ignore[assignment]
                new_atoms.append(Atom(element=atom_j.element, position=shifted))

        replicated = Molecule(atoms=new_atoms, bonds=[], lattice=self.lattice)
        replicated.detect_bonds(tolerance=bond_tolerance)
        return replicated

    def with_periodic_images(self, tol: float = 1e-3) -> "Molecule":
        """Return a copy with atoms on cell boundaries replicated across faces.

        Atoms whose fractional coordinates sit within ``tol`` of 0 or 1 along
        any axis are duplicated at the equivalent boundary on the opposite
        side, so the unit cell appears visually complete. Bonds are recomputed
        on the replicated set. If the molecule has no lattice, returns self.
        """
        if self.lattice is None or not self.atoms:
            return self

        from itertools import product

        positions = np.array([a.position for a in self.atoms], dtype=np.float64)
        inv = np.linalg.inv(self.lattice)
        fracs = positions @ inv  # rows: fractional coords

        new_atoms: list[Atom] = []
        for atom, frac in zip(self.atoms, fracs, strict=False):
            shifts_per_dim: list[list[int]] = []
            for d in range(3):
                f = frac[d] - np.floor(frac[d])
                shifts = [0]
                if f < tol:
                    shifts.append(1)
                elif 1.0 - f < tol:
                    shifts.append(-1)
                shifts_per_dim.append(shifts)
            for s0, s1, s2 in product(*shifts_per_dim):
                disp = s0 * self.lattice[0] + s1 * self.lattice[1] + s2 * self.lattice[2]
                new_atoms.append(Atom(element=atom.element, position=atom.position + disp))

        replicated = Molecule(atoms=new_atoms, bonds=[], lattice=self.lattice)
        replicated.detect_bonds()
        return replicated

    def detect_bonds(self, tolerance: float = 1.3):
        n = len(self.atoms)
        if n < 2:
            self.bonds = []
            self.bond_shifts = None
            return

        positions = np.array([atom.position for atom in self.atoms], dtype=np.float64)
        covalent_radii = np.array(
            [atom.element.covalent_radius for atom in self.atoms], dtype=np.float64
        )

        # Same pair ordering as nested loops: i outer, j inner (i < j).
        i_idx, j_idx = np.triu_indices(n, k=1)
        delta = positions[i_idx] - positions[j_idx]
        dist_sq = np.einsum("ij,ij->i", delta, delta)

        max_bond = (covalent_radii[i_idx] + covalent_radii[j_idx]) * tolerance
        is_bonded = dist_sq < (max_bond * max_bond)

        self.bonds = [
            (int(i), int(j)) for i, j in zip(i_idx[is_bonded], j_idx[is_bonded], strict=False)
        ]
        self.bond_shifts = None

    def detect_bonds_periodic(self, tolerance: float = 1.3):
        """Detect bonds including periodic-image neighbours.

        Records each unique bond once with the integer cell shift applied to
        the j atom. The (0, 0, 0) shift recovers in-cell bonds; non-zero
        shifts cover bonds that cross unit-cell faces.
        """
        if self.lattice is None:
            raise ValueError("detect_bonds_periodic requires a lattice")

        from itertools import product

        n = len(self.atoms)
        if n == 0:
            self.bonds = []
            self.bond_shifts = None
            return

        positions = np.array([atom.position for atom in self.atoms], dtype=np.float64)
        radii = np.array([atom.element.covalent_radius for atom in self.atoms], dtype=np.float64)

        bonds: list[tuple[int, int]] = []
        shifts: list[tuple[int, int, int]] = []
        for s0, s1, s2 in product((-1, 0, 1), repeat=3):
            disp = s0 * self.lattice[0] + s1 * self.lattice[1] + s2 * self.lattice[2]
            shifted = positions + disp
            for i in range(n):
                # Restrict to unique unordered pairs:
                #   * For different atoms, take only i < j (any shift).
                #   * For the same atom, take only the lex-positive half of
                #     the shift to avoid double-counting (i,i,s) and (i,i,-s);
                #     drop the trivial zero shift.
                if (s0, s1, s2) == (0, 0, 0):
                    j_start = i + 1
                else:
                    j_start = i
                for j in range(j_start, n):
                    if i == j:
                        if (s0, s1, s2) <= (0, 0, 0):
                            continue
                    delta = shifted[j] - positions[i]
                    dist_sq = float(delta @ delta)
                    cutoff = (radii[i] + radii[j]) * tolerance
                    if dist_sq < cutoff * cutoff:
                        bonds.append((i, j))
                        shifts.append((s0, s1, s2))

        self.bonds = bonds
        self.bond_shifts = shifts
