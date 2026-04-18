# MolTUI

**MolTUI** is a terminal molecular viewer for the **XYZ**, **Zmat**, **Molden**, **Gaussian Cube** and **ORCA GBW** file format designed for **quick inspection** of **geometries** and **orbitals** directly in the **terminal** using **Unicode** characters.
Ideal for **remote SSH sessions** and **lightweight analyses**.

<img width="480" height="480" alt="benzene" src="https://github.com/user-attachments/assets/c71de594-9dd3-4cb4-9754-e86dc663f730" />

## Installation

```bash
pip install moltui
# or
uv tool install moltui
```

## Usage

```bash
moltui <file>
```

## Features

### Visualize Orbitals

- The **rendering** of orbitals can be toggled via `o`.
- Molden and GBW files can contain **multiple molecular orbitals**. **Toggle** the orbital **sidebar** with `m`. **Cycle** through MOs with `n`ext and `p`rev (or via `[` and `]` even when the sidebar is hidden).

<img width="1512" height="926" alt="image" src="https://github.com/user-attachments/assets/4c1743ba-aff0-4683-92a7-7ebfaa361258" />

### Analyze Geometry

- **Bond lengths, angles and dihedrals** can be viewed using the `g`eomtry key which opens a sidebar. Navigate between tabs via `<tab>`.
- The quantity is **highlighted in yellow** on the molecule.
- **Sort** the quantity in ascending order via `s`.
- **Atom indices** can be toggled via `#`.

<img width="1510" height="923" alt="image" src="https://github.com/user-attachments/assets/8a6dab9a-d377-4d16-bfe1-89c83d0763a1" />

### Export to PNG Format

The `e` key **exports** the current **scene** to a **PNG**.

<img width="800" height="600" alt="benzene_hf 021" src="https://github.com/user-attachments/assets/2ca67320-9053-4b86-989f-b2abfaca8864" />

### Tune Visuals

The `V` key opens a sidebar where the **isovalue**, molecule **style** and **lighting** can be modified.

<img width="1229" height="877" alt="image" src="https://github.com/user-attachments/assets/177d4c5a-df88-4fb6-8fd7-5b51c0c467ef" />

The lower case `v` cycles between the **styles**.

| CPK | Licorice | VDW |
|--------|--------|--------|
| ![](https://github.com/user-attachments/assets/304b38f9-5fee-4bbc-89f6-062eb1fb0962) | ![](https://github.com/user-attachments/assets/b96f1780-90c0-49c2-8dd0-784602390f43) | ![](https://github.com/user-attachments/assets/d32b81b4-72b8-4ae5-9312-2ef7a73030bb) |

Toggle between **light** and **dark** mode with `i`.

| Light | Dark |
| --- | --- |
|![](https://github.com/user-attachments/assets/a82b5220-4eb4-4105-ada3-5ba1778f7dd6) | ![](https://github.com/user-attachments/assets/304b38f9-5fee-4bbc-89f6-062eb1fb0962) |


## Supported formats

- **Structures Only**: **XYZ**, Gaussian **ZMAT**.
- **Structures and Orbitals**: **Molden**, Gaussian **Cube**, Orca **GBW**¹.

¹ Requires `orca_2mkl` in `PATH`

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
| `v` | Toggle **style** (CPK, Licorice, VDW) |
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

## Limitations

- The content is rendered using braille Unicode characters and, therefore, the quality of rendering can depend on the font and terminal emulator. All figures in the repository have been generated using the JetBrains Mono Nerd Font in the Kitty terminal. 
- The Orca GBW file format is typically incompatible between versions. Therefore, the `orca_2mkl` should ideally be of the same version as the Orca version used to produce the GBW file. Newer version of Orca can try to recover earlier GBW files using the [rescue](https://orca-manual.mpi-muelheim.mpg.de/contents/quickstartguide/troubleshooting.html#using-old-orca-inputs) feature. 
