import atexit
import base64
import fcntl
import os
import select
import struct
import sys
import tempfile
import termios
import tty
import zlib
from pathlib import Path

import numpy as np

from .elements import Molecule
from .image_renderer import render_scene, rotation_matrix
from .isosurface import IsosurfaceMesh, extract_isosurfaces
from .parsers import load_molecule, parse_cube_data
from .renderer import Renderer


def _get_terminal_pixel_size() -> tuple[int, int]:
    """Get terminal size in pixels via ioctl, fallback to estimate."""
    try:
        result = fcntl.ioctl(
            sys.stdout.fileno(), termios.TIOCGWINSZ, b"\0" * 8
        )
        rows, cols, xpixel, ypixel = struct.unpack("HHHH", result)
        if xpixel > 0 and ypixel > 0:
            return xpixel, ypixel
    except OSError:
        pass
    cols, rows = os.get_terminal_size()
    return cols * 8, rows * 16


_TEMP_PATH: str | None = None


def _get_temp_path() -> str:
    global _TEMP_PATH
    if _TEMP_PATH is None:
        fd, _TEMP_PATH = tempfile.mkstemp(prefix="moltui_")
        os.close(fd)
        atexit.register(lambda: os.unlink(_TEMP_PATH) if os.path.exists(_TEMP_PATH) else None)
    return _TEMP_PATH


def _kitty_send(
    pixels: np.ndarray, image_id: int = 1, cols: int = 0, rows: int = 0
) -> None:
    """Send an RGB pixel array to the terminal via kitty graphics protocol.

    Uses temp file + zlib compression for speed (no PNG encode, no base64 chunking).
    cols/rows: display size in terminal cells (kitty scales the image).
    """
    h, w = pixels.shape[:2]
    compressed = zlib.compress(pixels.tobytes(), level=1)

    path = _get_temp_path()
    with open(path, "wb") as f:
        f.write(compressed)

    path_b64 = base64.standard_b64encode(path.encode()).decode()
    placement = ""
    if cols > 0 and rows > 0:
        placement = f",c={cols},r={rows}"
    sys.stdout.write(
        f"\033_Ga=T,f=24,s={w},v={h},t=f,i={image_id},o=z,q=2{placement};{path_b64}\033\\"
    )
    sys.stdout.flush()


def _kitty_delete(image_id: int | None = None) -> None:
    """Delete kitty images."""
    if image_id is not None:
        sys.stdout.write(f"\033_Ga=d,d=I,i={image_id},q=2\033\\")
    else:
        sys.stdout.write("\033_Ga=d,d=A,q=2\033\\")
    sys.stdout.flush()


def _read_key() -> str:
    """Read a single keypress, handling escape sequences for arrow keys."""
    ch = sys.stdin.read(1)
    if ch == "\033":
        if select.select([sys.stdin], [], [], 0.05)[0]:
            ch2 = sys.stdin.read(1)
            if ch2 == "[":
                ch3 = sys.stdin.read(1)
                return {
                    "A": "up",
                    "B": "down",
                    "C": "right",
                    "D": "left",
                }.get(ch3, "")
        return "escape"
    return ch


class TextViewer:
    """Fallback viewer using text-based rendering (works over SSH)."""

    def __init__(
        self,
        molecule: Molecule,
        filepath: str = "",
    ):
        self.molecule = molecule
        self.filepath = filepath
        self.rot_x = 0.5
        self.rot_y = 0.0
        self.rot_z = 0.0
        mol_radius = molecule.radius()
        self.camera_distance = max(4.0, mol_radius * 3.0)
        self.show_bonds = True
        self.dark_bg = True

    def run(self) -> None:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            sys.stdout.write("\033[?25l")  # hide cursor
            sys.stdout.write("\033[2J")  # clear screen
            sys.stdout.flush()

            self._render()

            while True:
                key = _read_key()
                if not self._handle_key(key):
                    break
                while select.select([sys.stdin], [], [], 0)[0]:
                    key = _read_key()
                    if not self._handle_key(key):
                        return
        finally:
            sys.stdout.write("\033[?25h")  # show cursor
            sys.stdout.write("\033[2J\033[H")  # clear screen
            sys.stdout.flush()
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def _handle_key(self, key: str) -> bool:
        needs_render = True

        if key == "q":
            return False
        elif key in ("up", "k"):
            self.rot_x -= 0.1
        elif key in ("down", "j"):
            self.rot_x += 0.1
        elif key in ("left", "h"):
            self.rot_y -= 0.1
        elif key in ("right", "l"):
            self.rot_y += 0.1
        elif key == ",":
            self.rot_z += 0.1
        elif key == ".":
            self.rot_z -= 0.1
        elif key in ("+", "="):
            self.camera_distance = max(1.0, self.camera_distance - 0.5)
        elif key == "-":
            self.camera_distance += 0.5
        elif key == "r":
            self.rot_x = 0.5
            self.rot_y = 0.0
            self.rot_z = 0.0
            mol_radius = self.molecule.radius()
            self.camera_distance = max(4.0, mol_radius * 3.0)
        elif key == "b":
            self.show_bonds = not self.show_bonds
        elif key == "i":
            self.dark_bg = not self.dark_bg
        else:
            needs_render = False

        if needs_render:
            self._render()
        return True

    def _render(self) -> None:
        cols, rows = os.get_terminal_size()
        render_rows = rows - 1  # leave 1 row for status

        renderer = Renderer(cols, render_rows)
        rot = rotation_matrix(self.rot_x, self.rot_y, self.rot_z)

        mol = self.molecule
        if not self.show_bonds:
            mol = Molecule(atoms=mol.atoms, bonds=[])

        renderer.render_molecule(mol, rot, self.camera_distance)

        bg = "0;0;0" if self.dark_bg else "255;255;255"
        sys.stdout.write("\033[H")  # cursor home
        for y in range(render_rows):
            sys.stdout.write(f"\033[{y + 1};1H")
            for x in range(cols):
                bg_col = renderer.bg_buf[y][x]
                if bg_col is not None:
                    sys.stdout.write(
                        f"\033[48;2;{bg_col[0]};{bg_col[1]};{bg_col[2]}m \033[0m"
                    )
                else:
                    sys.stdout.write(f"\033[48;2;{bg}m \033[0m")

        # Status bar
        sys.stdout.write(f"\033[{rows};1H")
        sys.stdout.write("\033[7m")
        n = len(self.molecule.atoms)
        b = len(self.molecule.bonds)
        parts = [
            Path(self.filepath).name,
            f"{n} atoms, {b} bonds",
            "arrows rot", "+/- zoom", "b bonds", "i bg",
            "r reset", "q quit",
        ]
        status = " " + " | ".join(parts)
        sys.stdout.write(status[:cols].ljust(cols))
        sys.stdout.write("\033[0m")
        sys.stdout.flush()


class KittyViewer:
    def __init__(
        self,
        molecule: Molecule,
        filepath: str = "",
        isosurfaces: list[IsosurfaceMesh] | None = None,
        molden_data=None,
        current_mo: int = 0,
    ):
        self.molecule = molecule
        self.filepath = filepath
        self.isosurfaces = isosurfaces or []
        self.molden_data = molden_data
        self.current_mo = current_mo

        self.rot_x = 0.5
        self.rot_y = 0.0
        self.rot_z = 0.0
        mol_radius = molecule.radius()
        self.camera_distance = max(4.0, mol_radius * 3.0)
        self.show_bonds = True
        self.show_orbitals = True
        self.dark_bg = True

    def run(self) -> None:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            sys.stdout.write("\033[?25l")  # hide cursor
            sys.stdout.write("\033[2J")  # clear screen
            sys.stdout.flush()

            self._render()

            while True:
                key = _read_key()
                if not self._handle_key(key):
                    break
                # Debounce: drain any queued keys before rendering
                while select.select([sys.stdin], [], [], 0)[0]:
                    key = _read_key()
                    if not self._handle_key(key):
                        return
        finally:
            _kitty_delete()
            sys.stdout.write("\033[?25h")  # show cursor
            sys.stdout.write("\033[2J\033[H")  # clear screen
            sys.stdout.flush()
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def _handle_key(self, key: str) -> bool:
        """Handle a keypress. Returns False to quit."""
        needs_render = True

        if key == "q":
            return False
        elif key in ("up", "k"):
            self.rot_x -= 0.1
        elif key in ("down", "j"):
            self.rot_x += 0.1
        elif key in ("left", "h"):
            self.rot_y -= 0.1
        elif key in ("right", "l"):
            self.rot_y += 0.1
        elif key == ",":
            self.rot_z += 0.1
        elif key == ".":
            self.rot_z -= 0.1
        elif key in ("+", "="):
            self.camera_distance = max(1.0, self.camera_distance - 0.5)
        elif key == "-":
            self.camera_distance += 0.5
        elif key == "r":
            self.rot_x = 0.5
            self.rot_y = 0.0
            self.rot_z = 0.0
            mol_radius = self.molecule.radius()
            self.camera_distance = max(4.0, mol_radius * 3.0)
        elif key == "b":
            self.show_bonds = not self.show_bonds
        elif key == "i":
            self.dark_bg = not self.dark_bg
        elif key == "o":
            self.show_orbitals = not self.show_orbitals
        elif key == "]":
            self._next_mo()
        elif key == "[":
            self._prev_mo()
        else:
            needs_render = False

        if needs_render:
            self._render()
        return True

    MAX_RENDER_W = 800
    MAX_RENDER_H = 600

    def _render(self) -> None:
        cols, rows = os.get_terminal_size()
        px_w, px_h = _get_terminal_pixel_size()
        # Leave 1 row for status bar
        status_px = px_h // rows if rows > 0 else 16
        px_h -= status_px

        # Cap resolution — kitty scales the image to fit the terminal
        aspect = px_w / max(px_h, 1)
        if px_w > self.MAX_RENDER_W or px_h > self.MAX_RENDER_H:
            if aspect > self.MAX_RENDER_W / self.MAX_RENDER_H:
                px_w = self.MAX_RENDER_W
                px_h = int(px_w / aspect)
            else:
                px_h = self.MAX_RENDER_H
                px_w = int(px_h * aspect)

        bg = (0, 0, 0) if self.dark_bg else (255, 255, 255)
        rot = rotation_matrix(self.rot_x, self.rot_y, self.rot_z)

        mol = self.molecule
        if not self.show_bonds:
            mol = Molecule(atoms=mol.atoms, bonds=[])

        isos = self.isosurfaces if self.show_orbitals else None
        pixels = render_scene(
            px_w, px_h, mol, rot, self.camera_distance,
            bg_color=bg, isosurfaces=isos, ssaa=2,
        )

        # Send image at top-left, scaled to fill terminal
        sys.stdout.write("\033[H")  # cursor home
        _kitty_send(pixels, image_id=1, cols=cols, rows=rows - 1)

        # Draw status bar on last row
        sys.stdout.write(f"\033[{rows};1H")  # move to last line
        sys.stdout.write("\033[7m")  # reverse video
        status = self._status_text()
        sys.stdout.write(status[:cols].ljust(cols))
        sys.stdout.write("\033[0m")  # reset
        sys.stdout.flush()

    def _status_text(self) -> str:
        n = len(self.molecule.atoms)
        b = len(self.molecule.bonds)
        parts = [
            Path(self.filepath).name,
            f"{n} atoms, {b} bonds",
        ]
        if self.molden_data is not None:
            md = self.molden_data
            energy = md.mo_energies[self.current_mo]
            occ = md.mo_occupations[self.current_mo]
            sym = (
                md.mo_symmetries[self.current_mo]
                if self.current_mo < len(md.mo_symmetries)
                else ""
            )
            homo_label = ""
            if self.current_mo == md.homo_idx:
                homo_label = " HOMO"
            elif self.current_mo == md.homo_idx + 1:
                homo_label = " LUMO"
            parts.append(
                f"MO {self.current_mo + 1}/{md.n_mos} {sym}{homo_label} E={energy:.4f} occ={occ:.1f}"
            )
            parts.append("[/] MO")
        parts += ["arrows rot", "+/- zoom", "b bonds", "i bg"]
        if self.isosurfaces:
            parts.append("o orb")
        parts += ["r reset", "q quit"]
        return " " + " | ".join(parts)

    def _next_mo(self) -> None:
        if self.molden_data is None:
            return
        if self.current_mo < self.molden_data.n_mos - 1:
            self.current_mo += 1
            self._switch_mo()

    def _prev_mo(self) -> None:
        if self.molden_data is None:
            return
        if self.current_mo > 0:
            self.current_mo -= 1
            self._switch_mo()

    def _switch_mo(self) -> None:
        from .molden import evaluate_mo

        cube_data = evaluate_mo(self.molden_data, self.current_mo)
        self.isosurfaces = extract_isosurfaces(cube_data)


def run():
    args = sys.argv[1:]
    text_mode = "--text" in args
    if text_mode:
        args.remove("--text")

    if len(args) < 1:
        print("Usage: moltui [--text] <file.xyz|file.cube|file.molden>")
        sys.exit(1)

    filepath = args[0]
    suffix = Path(filepath).suffix.lower()
    isosurfaces: list[IsosurfaceMesh] = []
    molden_data = None
    current_mo = 0

    if suffix == ".cube":
        cube_data = parse_cube_data(filepath)
        molecule = cube_data.molecule
        isosurfaces = extract_isosurfaces(cube_data)
    elif suffix == ".molden":
        from .molden import evaluate_mo, load_molden_data

        print("Loading molden file...")
        molden_data = load_molden_data(filepath)
        molecule = molden_data.molecule
        current_mo = molden_data.homo_idx
        print(f"Evaluating MO {current_mo + 1}/{molden_data.n_mos} (HOMO)...")
        cube_data = evaluate_mo(molden_data, current_mo)
        isosurfaces = extract_isosurfaces(cube_data)
    else:
        molecule = load_molecule(filepath)

    if text_mode:
        viewer = TextViewer(
            molecule=molecule,
            filepath=filepath,
        )
    else:
        viewer = KittyViewer(
            molecule=molecule,
            filepath=filepath,
            isosurfaces=isosurfaces,
            molden_data=molden_data,
            current_mo=current_mo,
        )
    viewer.run()
