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
    "H":  Element("H",  "Hydrogen",   1,  (255, 255, 255), 0.31, 1.20),
    "He": Element("He", "Helium",     2,  (217, 255, 255), 0.28, 1.40),
    "Li": Element("Li", "Lithium",    3,  (204, 128, 255), 1.28, 1.82),
    "Be": Element("Be", "Beryllium",  4,  (194, 255,   0), 0.96, 1.53),
    "B":  Element("B",  "Boron",      5,  (255, 181, 181), 0.84, 1.92),
    "C":  Element("C",  "Carbon",     6,  (150, 150, 150), 0.76, 1.70),
    "N":  Element("N",  "Nitrogen",   7,  ( 48,  80, 248), 0.71, 1.55),
    "O":  Element("O",  "Oxygen",     8,  (255,  60,  60), 0.66, 1.52),
    "F":  Element("F",  "Fluorine",   9,  (144, 224,  80), 0.57, 1.47),
    "Ne": Element("Ne", "Neon",       10, (179, 227, 245), 0.58, 1.54),
    "Na": Element("Na", "Sodium",     11, (171,  92, 242), 1.66, 2.27),
    "Mg": Element("Mg", "Magnesium",  12, (138, 255,   0), 1.41, 1.73),
    "Al": Element("Al", "Aluminum",   13, (191, 166, 166), 1.21, 1.84),
    "Si": Element("Si", "Silicon",    14, (240, 200, 160), 1.11, 2.10),
    "P":  Element("P",  "Phosphorus", 15, (255, 128,   0), 1.07, 1.80),
    "S":  Element("S",  "Sulfur",     16, (255, 255,  48), 1.05, 1.80),
    "Cl": Element("Cl", "Chlorine",   17, ( 31, 240,  31), 1.02, 1.75),
    "Ar": Element("Ar", "Argon",      18, (128, 209, 227), 1.06, 1.88),
    "K":  Element("K",  "Potassium",  19, (143,  64, 212), 2.03, 2.75),
    "Ca": Element("Ca", "Calcium",    20, ( 61, 255,   0), 1.76, 2.31),
    "Ti": Element("Ti", "Titanium",   22, (191, 194, 199), 1.60, 2.11),
    "Fe": Element("Fe", "Iron",       26, (224, 102,  51), 1.32, 2.04),
    "Co": Element("Co", "Cobalt",     27, (240, 144, 160), 1.26, 2.00),
    "Ni": Element("Ni", "Nickel",     28, ( 80, 208,  80), 1.24, 1.97),
    "Cu": Element("Cu", "Copper",     29, (200, 128,  51), 1.32, 1.96),
    "Zn": Element("Zn", "Zinc",       30, (125, 128, 176), 1.22, 2.01),
    "Br": Element("Br", "Bromine",    35, (166,  41,  41), 1.20, 1.85),
    "I":  Element("I",  "Iodine",     53, (148,   0, 148), 1.39, 1.98),
}

DEFAULT_ELEMENT = Element("X", "Unknown", 0, (255, 20, 147), 1.00, 1.70)

ATOMIC_NUMBER_TO_SYMBOL: dict[int, str] = {
    e.atomic_number: e.symbol for e in ELEMENTS.values()
}


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

    def detect_bonds(self, tolerance: float = 1.3):
        self.bonds = []
        n = len(self.atoms)
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(self.atoms[i].position - self.atoms[j].position)
                max_bond = (
                    self.atoms[i].element.covalent_radius
                    + self.atoms[j].element.covalent_radius
                ) * tolerance
                if dist < max_bond:
                    self.bonds.append((i, j))
