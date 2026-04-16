# moltui

<img width="300" height="300" alt="benzene" src="https://github.com/user-attachments/assets/c71de594-9dd3-4cb4-9754-e86dc663f730" />

Terminal-based 3D molecular viewer.

## Installation

```bash
pip install moltui
```

```bash
pip install moltui
```

## Usage

```bash
moltui <file>
```


## Supported formats

### Structures
- XYZ
- Gaussian Zmat

### Structures and Orbitals
- Gaussian Cube
- Molden
- ORCA GBW (requires `orca_2mkl` in ``)

## Features

- 3D molecule visualization using Unicode braille characters
- Molecular orbital isosurfaces (positive/negative lobes)
- Geometry panel with bond lengths, angles, and dihedrals
- MO browser with energies, occupations, and symmetry labels
- Real-time rotation, zoom, and panning
- Dark and light themes

### Light and Dark Mode


## Keybindings

### Navigation

| Key | Action |
|-----|--------|
| `h/j/k/l` or arrows | Rotate left/down/up/right |
| `,/.` | Roll clockwise/counter-clockwise |
| `J/K` or `+/-` | Zoom out/in |
| `t` | Toggle pan/rotation mode |
| `c` | Center view |
| `r` | Reset view |

### Display

| Key | Action |
|-----|--------|
| `o` | Toggle orbital isosurfaces |
| `i` | Toggle dark/light background |
| `b` | Toggle bonds |
| `e` | Export PNG |
| `v` | Toggle CPK/licorice style |
| `#` | Toggle atom numbers |

### Panels

| Key | Action |
|-----|--------|
| `g` | Geometry panel (bonds, angles, dihedrals) |
| `m` | MO panel (molecular orbitals) |
| `V` | Visual settings panel (style, sizes, lighting) |
| `[`, `]` | Previous/next MO |
| `n/p` | Navigate panel entries |
| `Esc` | Close panel |

### Visual panel

| Key | Action |
|-----|--------|
| `n/p` | Move between controls |
| `Tab/Shift+Tab` | Adjust value (slider) or switch option (style) |

### General

| Key | Action |
|-----|--------|
| `q` | Quit |

## License

MIT
