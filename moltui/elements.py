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
    "Be": Element("Be", "Beryllium", 4, (194, 255, 0), 0.96, 1.53),
    "B": Element("B", "Boron", 5, (255, 181, 181), 0.84, 1.92),
    "C": Element("C", "Carbon", 6, (150, 150, 150), 0.76, 1.70),
    "N": Element("N", "Nitrogen", 7, (48, 80, 248), 0.71, 1.55),
    "O": Element("O", "Oxygen", 8, (255, 60, 60), 0.66, 1.52),
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
    "Mn": Element("Mn", "Manganese", 25, (156, 122, 199), 1.39, 2.05),
    "Fe": Element("Fe", "Iron", 26, (224, 102, 51), 1.32, 2.04),
    "Co": Element("Co", "Cobalt", 27, (240, 144, 160), 1.26, 2.00),
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
    "Ir": Element("Ir", "Iridium", 77, (23, 124, 62), 1.41, 2.20),
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
    key = symbol.strip().capitalize()
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

    def _adjacency(self) -> dict[int, list[int]]:
        adj: dict[int, list[int]] = {i: [] for i in range(len(self.atoms))}
        for i, j in self.bonds:
            adj[i].append(j)
            adj[j].append(i)
        return adj

    def get_bond_lengths(self) -> list[tuple[int, int, float]]:
        results = []
        for i, j in self.bonds:
            dist = float(np.linalg.norm(self.atoms[i].position - self.atoms[j].position))
            results.append((i, j, dist))
        return results

    def get_angles(self) -> list[tuple[int, int, int, float]]:
        adj = self._adjacency()
        results = []
        for j, neighbors in adj.items():
            for ni, i in enumerate(neighbors):
                for k in neighbors[ni + 1 :]:
                    v1 = self.atoms[i].position - self.atoms[j].position
                    v2 = self.atoms[k].position - self.atoms[j].position
                    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
                    angle = float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))
                    results.append((i, j, k, angle))
        return results

    def get_dihedrals(self) -> list[tuple[int, int, int, int, float]]:
        adj = self._adjacency()
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
                    b1 = self.atoms[j].position - self.atoms[i].position
                    b2 = self.atoms[k].position - self.atoms[j].position
                    b3 = self.atoms[l].position - self.atoms[k].position
                    n1 = np.cross(b1, b2)
                    n2 = np.cross(b2, b3)
                    n1_norm = np.linalg.norm(n1) + 1e-10
                    n2_norm = np.linalg.norm(n2) + 1e-10
                    cos_d = np.dot(n1, n2) / (n1_norm * n2_norm)
                    angle = float(np.degrees(np.arccos(np.clip(cos_d, -1.0, 1.0))))
                    results.append((i, j, k, l, angle))
        return results

    def detect_bonds(self, tolerance: float = 1.3):
        self.bonds = []
        n = len(self.atoms)
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(self.atoms[i].position - self.atoms[j].position)
                max_bond = (
                    self.atoms[i].element.covalent_radius + self.atoms[j].element.covalent_radius
                ) * tolerance
                if dist < max_bond:
                    self.bonds.append((i, j))
