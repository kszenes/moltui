import os
import select
import signal
import sys
import termios
import tty
from pathlib import Path

import numpy as np

from .elements import Molecule
from .image_renderer import render_scene, rotation_matrix
from .isosurface import IsosurfaceMesh, extract_isosurfaces
from .parsers import CubeData, load_molecule, parse_cube_data


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
        isosurfaces: list[IsosurfaceMesh] | None = None,
        molden_data=None,
        current_mo: int = 0,
        cube_data: CubeData | None = None,
    ):
        self.molecule = molecule
        self.filepath = filepath
        self.isosurfaces = isosurfaces or []
        self.molden_data = molden_data
        self.current_mo = current_mo
        self.cube_data = cube_data
        self.rot_x = 0.5
        self.rot_y = 0.0
        self.rot_z = 0.0
        mol_radius = molecule.radius()
        self.camera_distance = max(4.0, mol_radius * 3.0)
        self.show_bonds = True
        self.show_orbitals = True
        self.dark_bg = True
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.pan_mode = False

    def run(self) -> None:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        self._resize_pending = False
        old_sigwinch = signal.getsignal(signal.SIGWINCH)

        def _on_resize(signum, frame):
            self._resize_pending = True

        try:
            signal.signal(signal.SIGWINCH, _on_resize)
            tty.setcbreak(fd)
            sys.stdout.write("\033[?25l")  # hide cursor
            sys.stdout.write("\033[2J")  # clear screen
            sys.stdout.flush()

            self._render()

            while True:
                ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                if self._resize_pending:
                    self._resize_pending = False
                    self._render()
                if not ready:
                    continue
                key = _read_key()
                if not self._handle_key(key):
                    break
                while select.select([sys.stdin], [], [], 0)[0]:
                    key = _read_key()
                    if not self._handle_key(key):
                        return
        finally:
            signal.signal(signal.SIGWINCH, old_sigwinch)
            sys.stdout.write("\033[?25h")  # show cursor
            sys.stdout.write("\033[2J\033[H")  # clear screen
            sys.stdout.flush()
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def _clamp_pan(self) -> None:
        max_pan = self.molecule.radius() * 0.5
        self.pan_x = max(-max_pan, min(max_pan, self.pan_x))
        self.pan_y = max(-max_pan, min(max_pan, self.pan_y))

    def _handle_key(self, key: str) -> bool:
        needs_render = True
        pan_step = self.camera_distance * 0.05

        if key == "q":
            return False
        elif key == "t":
            self.pan_mode = not self.pan_mode
        elif key in ("up", "k"):
            if self.pan_mode:
                self.pan_y -= pan_step
                self._clamp_pan()
            else:
                self.rot_x -= 0.1
        elif key in ("down", "j"):
            if self.pan_mode:
                self.pan_y += pan_step
                self._clamp_pan()
            else:
                self.rot_x += 0.1
        elif key in ("left", "h"):
            if self.pan_mode:
                self.pan_x += pan_step
                self._clamp_pan()
            else:
                self.rot_y -= 0.1
        elif key in ("right", "l"):
            if self.pan_mode:
                self.pan_x -= pan_step
                self._clamp_pan()
            else:
                self.rot_y += 0.1
        elif key == ",":
            self.rot_z += 0.1
        elif key == ".":
            self.rot_z -= 0.1
        elif key in ("+", "=", "K"):
            self.camera_distance = max(1.0, self.camera_distance - 0.5)
        elif key in ("-", "J"):
            self.camera_distance += 0.5
        elif key == "r":
            self.rot_x = 0.5
            self.rot_y = 0.0
            self.rot_z = 0.0
            self.pan_x = 0.0
            self.pan_y = 0.0
            self.pan_mode = False
            mol_radius = self.molecule.radius()
            self.camera_distance = max(4.0, mol_radius * 3.0)
        elif key == "c":
            self.pan_x = 0.0
            self.pan_y = 0.0
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

    # Braille dot positions: each cell is 2 wide × 4 tall
    # Bit layout for Unicode braille (U+2800 + bits):
    #   col0: rows 0-2 = bits 0,1,2; row 3 = bit 6
    #   col1: rows 0-2 = bits 3,4,5; row 3 = bit 7
    _BRAILLE_MAP = np.array([
        [0x01, 0x08],
        [0x02, 0x10],
        [0x04, 0x20],
        [0x40, 0x80],
    ], dtype=np.uint8)

    def _render(self) -> None:
        cols, rows = os.get_terminal_size()
        display_rows = rows - 2  # leave 1 row for title bar + 1 for status
        # Each terminal cell = 2×4 pixels
        px_w = cols * 2
        px_h = display_rows * 4

        bg = (0, 0, 0) if self.dark_bg else (255, 255, 255)
        rot = rotation_matrix(self.rot_x, self.rot_y, self.rot_z)

        mol = self.molecule
        if not self.show_bonds:
            mol = Molecule(atoms=mol.atoms, bonds=[])

        isos = self.isosurfaces if self.show_orbitals else None
        pixels = render_scene(
            px_w, px_h, mol, rot, self.camera_distance,
            bg_color=bg, isosurfaces=isos, ssaa=1,
            pan=(self.pan_x, self.pan_y),
        )

        bg_arr = np.array(bg, dtype=np.uint8)
        # Reshape to (display_rows, 4, cols, 2, 3) for block processing
        blocks = pixels.reshape(display_rows, 4, cols, 2, 3)
        # Determine which pixels differ from background -> "on" dots
        is_on = np.any(blocks != bg_arr, axis=4)  # (display_rows, 4, cols, 2)

        # Compute braille codepoints: sum dot bits where pixel is on
        # _BRAILLE_MAP is (4, 2), broadcast with is_on (display_rows, 4, cols, 2)
        braille_bits = np.where(is_on, self._BRAILLE_MAP[None, :, None, :], 0)
        # Sum over the 4 rows and 2 cols of each block -> (display_rows, cols)
        codepoints = 0x2800 + braille_bits.sum(axis=(1, 3)).astype(np.uint32)

        # Average foreground color per block (only from "on" pixels)
        on_count = is_on.sum(axis=(1, 3))  # (display_rows, cols)
        # Sum colors of on-pixels: expand is_on to broadcast with color
        on_mask = is_on[:, :, :, :, None]  # (display_rows, 4, cols, 2, 1)
        color_sum = (blocks * on_mask).sum(axis=(1, 3))  # (display_rows, cols, 3)
        safe_count = np.maximum(on_count, 1)[:, :, None]
        avg_fg = (color_sum / safe_count).astype(np.uint8)  # (display_rows, cols, 3)

        # Title bar (row 1)
        buf = ["\033[1;1H\033[7m"]
        title_parts = [
            Path(self.filepath).name,
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
            title_parts.append(
                f"MO {self.current_mo + 1}/{md.n_mos} {sym}{homo_label} E={energy:.4f} occ={occ:.1f}"
            )
        title = " " + " | ".join(title_parts)
        buf.append(title[:cols].ljust(cols))
        buf.append("\033[0m")

        # Braille rendering area (rows 2 to rows-1)
        for row in range(display_rows):
            buf.append(f"\033[{row + 2};1H")
            buf.append(f"\033[48;2;{bg[0]};{bg[1]};{bg[2]}m")
            prev_fg = None
            for x in range(cols):
                cp = int(codepoints[row, x])
                if cp == 0x2800:
                    if prev_fg is not None:
                        prev_fg = None
                    buf.append(" ")
                else:
                    fg = (int(avg_fg[row, x, 0]), int(avg_fg[row, x, 1]), int(avg_fg[row, x, 2]))
                    if fg != prev_fg:
                        buf.append(f"\033[38;2;{fg[0]};{fg[1]};{fg[2]}m")
                        prev_fg = fg
                    buf.append(chr(cp))
        buf.append("\033[0m")
        sys.stdout.write("".join(buf))

        # Keybinding bar (last row)
        sys.stdout.write(f"\033[{rows};1H\033[7m")
        mode = "PAN" if self.pan_mode else "ROT"
        key_parts = [f"[{mode}] t toggle", "+/- zoom", "b bonds", "i bg"]
        if self.isosurfaces:
            key_parts.append("o orb")
        if self.molden_data is not None:
            key_parts.append("[/] MO")
        key_parts += ["r reset", "q quit"]
        status = " " + " | ".join(key_parts)
        sys.stdout.write(status[:cols].ljust(cols))
        sys.stdout.write("\033[0m")
        sys.stdout.flush()


def run():
    args = sys.argv[1:]

    if len(args) < 1:
        print("Usage: moltui <file.xyz|file.cube|file.molden>")
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

    viewer = TextViewer(
        molecule=molecule,
        filepath=filepath,
        isosurfaces=isosurfaces,
        molden_data=molden_data,
        current_mo=current_mo,
    )
    viewer.run()
