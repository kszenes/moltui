# MolTUI

**Terminal-based 3D molecular viewer** for **XYZ**, **Zmat**, **Molden**, **Gaussian Cube** and **ORCA GBW** file formats.

<img width="480" height="480" alt="benzene" src="https://github.com/user-attachments/assets/c71de594-9dd3-4cb4-9754-e86dc663f730" />

## Installation

```bash
pip install moltui
```

## Usage

```bash
moltui <file>
```

## Features

### Visualize Orbitals

- The **rendering** of orbitals can be toggled via `o`.
- Molden and GBW files can contain **multiple molecular orbitals**. **Toggle the orbital **sidebar** with `m`. **Cycle** through MOs with `n`ext and `p`rev (or via `[` and `]` even when the sidebar is hidden)

<img width="1512" height="926" alt="image" src="https://github.com/user-attachments/assets/4c1743ba-aff0-4683-92a7-7ebfaa361258" />

### Analyze Geometry

- **Bond lengths, angles and dihedrals** can be viewed using the `g`eomtry key which opens a sidebar. Navigate between tabs via `<tab>`.
- The quantity is **highlighted in yellow** on the molecule
- **Sort** the quantity in ascending order via `s`.
- **Atom indices** can be toggled via `#`

<img width="1510" height="923" alt="image" src="https://github.com/user-attachments/assets/8a6dab9a-d377-4d16-bfe1-89c83d0763a1" />

### Export to PNG Format

The `e` **exports** the current **scene** to a **PNG**.

<img width="800" height="600" alt="benzene_hf 021" src="https://github.com/user-attachments/assets/2ca67320-9053-4b86-989f-b2abfaca8864" />

## Supported formats

### Structures

- **XYZ**
- **Gaussian Zmat**

### Structures and Orbitals

- **Gaussian Cube**
- **Molden**
- **Orca GBW** (requires `orca_2mkl` in `PATH`)

## Keybindings

### Navigation

| Key | Action |
|-----|--------|
| `h/j/k/l` or arrows | **Rotate** left/down/up/right |
| `,/.` | **Roll** clockwise/counter-clockwise |
| `J/K` or `+/-` | **Zoom** out/in |
| `t` | Toggle **pan/rotation** mode |
| `c` | **Center** view |
| `r` | **Reset** view |

### Display

| Key | Action |
|-----|--------|
| `o` | Toggle **orbital isosurfaces** |
| `i` | Toggle **dark/light** theme |
| `b` | Toggle **bonds** |
| `e` | **Export** PNG |
| `v` | Toggle CPK/licorice **style** |
| `#` | Toggle **atom numbers** |

### Panels

| Key | Action |
|-----|--------|
| `g` | **Geometry** panel (bonds, angles, dihedrals) |
| `m` | **MO** panel (molecular orbitals) |
| `V` | **Visual settings** panel (style, sizes, lighting) |
| `[`, `]` | **Previous/next** MO |
| `n/p` | **Navigate** panel entries |
| `Esc` | **Close** panel |

### Visual panel

| Key | Action |
|-----|--------|
| `n/p` | **Move** between controls |
| `Tab/Shift+Tab` | **Adjust value** (slider) or **switch option** (style) |

### General

| Key | Action |
|-----|--------|
| `q` | **Quit** |

