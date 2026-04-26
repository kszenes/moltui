# MolTUI

**MolTUI** is a terminal molecular viewer for the **XYZ**, **Zmat**, **Molden**, **Gaussian Cube** and **.fchk**, **Orca .gbw** and **.hess**, and **TrexIO** file format designed for **quick inspection** of **geometries**, **trajectories**, **orbitals** and **normal modes** directly in the **terminal** using **Unicode** characters.
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

**MolTUI** is organized in a series of **modes** that can be **cycled** using the `m` (forward) and `M` (backward) keys.
Each mode opens a respective **sidebar** which can be **toggled** on and off with `S`.
The modes consist of **molecular orbitals**, **normal modes** and **geometry** and availability depends on the file format that was opened.


### Visualize Orbitals

- The **rendering** of orbitals can be toggled via `o`.
- **Molden** and **.gbw** files can contain **multiple molecular orbitals**. **Cycle** through MOs with `n`ext and `p`rev.

<img width="1512" height="926" alt="image" src="https://github.com/user-attachments/assets/4c1743ba-aff0-4683-92a7-7ebfaa361258" />

### Analyze Geometry

- **Bond lengths, angles and dihedrals** can be viewed using the `g`eometry key which opens a sidebar. Navigate between tabs via `<tab>`.
- The quantity is **highlighted in yellow** on the molecule.
- **Sort** the quantity in increasing magnitude via `s`.
- **Atom indices** can be toggled via `#`.

<img width="1510" height="923" alt="image" src="https://github.com/user-attachments/assets/8a6dab9a-d377-4d16-bfe1-89c83d0763a1" />

### Animations

#### Trajectories

**Trajectories** can be provided in the **multi-XYZ** file format (essentially multiple XYZ files concatenated together).
**Toggle** a looping **animation** of the trajectory using `<space>` and **cycle** individual **frames** using `[` and `]`.
The **geometry** sidebar values **updates live** with the **current frame**'s geometry.

<img width="1200" height="733" alt="geom-opt" src="https://github.com/user-attachments/assets/b66d6cfb-c4ae-4c56-ba9c-3c26a4fcdf20" />

#### Normal Modes

**Normal modes** can be provided either via the **Molden** or Orca `.hess` file formats.
**Animation** playing can be **toggled** with `<space>`.

<img width="1920" height="1080" alt="normal-modes2" src="https://github.com/user-attachments/assets/c13326ff-e24e-46d6-8b24-818b18bac73c" />


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

| Format                | Geometry | Orbitals | Normal Modes |
|-----------------------|:--------:|:--------:|:------------:|
| **XYZ**               | ✓        |          |              |
| Gaussian **ZMAT**     | ✓        |          |              |
| Gaussian **Cube**     | ✓        | ✓        |              |
| **Molden**            | ✓        | ✓        | ✓            |
| Gaussian **.fchk**    | ✓        | ✓        | ✓            |
| Orca **.GBW**¹        | ✓        | ✓        |              |
| Orca **.hess**        | ✓        |          | ✓            |
| **TrexIO**²           | ✓        | ✓        |              |

¹ Requires `orca_2mkl` in `PATH`

² Requires installing `moltui[trexio]`

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
| `m` / `M` | Cycle sidebar **mode** forward/backward (Geometry, MO, Normal Modes) |
| `S` | Toggle current **sidebar** (hide/show) |
| `V` | Toggle **Visual settings** panel |
| `n/p` | **Next/previous** row in the active sidebar table |
| `d/u` | Half page **down/up** in datatable |
| `g/G` | **Top/bottom** of the datatable |

### Geometry panel

| Key | Action |
|-----|--------|
| `Tab/Shift+Tab` | Switch **Bonds/Angles/Dihedrals** tab |
| `s` | Toggle **sort by value** for active tab |

### Visual panel

| Key | Action |
|-----|--------|
| `n/p` | Move focus to **next/previous** control |
| `Tab/Shift+Tab` | **Increase/decrease** focused slider, or cycle style option |

### Animation and modes

| Key | Action |
|-----|--------|
| `Space` | Toggle **play/pause** trajectory or normal mode animation |
| `[` / `]` | Previous/next **animation step** (frame/phase) |

### General

| Key | Action |
|-----|--------|
| `q` | **Quit** |

## Known Issues

- Only up to `g`-shells are implemented as this is the highest orbital shell officially supported by the Molden format.
- The content is rendered using braille Unicode characters and, therefore, the quality of rendering can depend on the font and terminal emulator. All figures in the repository have been generated using the JetBrains Mono Nerd Font in the Kitty terminal. 
- The Orca GBW file format is typically incompatible between versions. Therefore, the `orca_2mkl` should ideally be of the same version as the Orca version used to produce the GBW file. Newer version of Orca can try to recover earlier GBW files using the [rescue](https://orca-manual.mpi-muelheim.mpg.de/contents/quickstartguide/troubleshooting.html#using-old-orca-inputs) feature. 
