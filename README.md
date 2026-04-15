# moltui

<img width="480" height="480" alt="benzene" src="https://github.com/user-attachments/assets/c71de594-9dd3-4cb4-9754-e86dc663f730" />

Terminal-based 3D molecular viewer with braille rendering.


## Features

- 3D molecule visualization using Unicode braille characters
- Atoms rendered with CPK coloring, bonds as cylinders
- Molecular orbital isosurfaces (positive/negative lobes)
- Geometry panel with bond lengths, angles, and dihedrals
- MO browser with energies, occupations, and symmetry labels
- Real-time rotation, zoom, and panning
- Dark and light themes

## Installation

```bash
pip install moltui
```

For Molden file support (requires [PySCF](https://pyscf.org)):

```bash
pip install moltui[molden]
```

## Usage

```bash
moltui <file>
```

Supported formats: `.xyz`, `.cube`, `.molden`

## Keybindings

| Key | Action |
|-----|--------|
| `h/j/k/l` | Rotate left/down/up/right |
| `J/K` or `+/-` | Zoom out/in |
| `,/.` | Roll clockwise/counter-clockwise |
| `t` | Toggle pan/rotation mode |
| `c` | Center view |
| `r` | Reset view |
| `b` | Toggle bonds |
| `o` | Toggle orbitals |
| `i` | Toggle dark/light background |
| `g` | Geometry panel |
| `m` | MO panel |
| `[/]` | Previous/next MO |
| `n/p` | Navigate panel rows |
| `q` | Quit |

## License

MIT
