from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rich.segment import Segment
from rich.style import Style
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.events import Key
from textual.strip import Strip
from textual.widget import Widget
from textual.widgets import DataTable, Footer, Header, RadioSet, TabbedContent

from .elements import Molecule
from .geometry_panel import GeometryPanel
from .image_renderer import render_scene, rotation_matrix
from .isosurface import IsosurfaceMesh, extract_isosurfaces
from .mo_panel import MOPanel
from .normal_mode_panel import NormalModePanel
from .parsers import (
    CubeData,
    load_molecule,
    parse_cube_data,
    parse_orca_hess_data,
    parse_xyz_trajectory,
)
from .visual_panel import Slider, VisualPanel

# Braille dot positions: each cell is 2 wide x 4 tall
# Bit layout for Unicode braille (U+2800 + bits):
#   col0: rows 0-2 = bits 0,1,2; row 3 = bit 6
#   col1: rows 0-2 = bits 3,4,5; row 3 = bit 7
_BRAILLE_MAP = np.array(
    [
        [0x01, 0x08],
        [0x02, 0x10],
        [0x04, 0x20],
        [0x40, 0x80],
    ],
    dtype=np.uint8,
)
_ZERO_MODE_FREQ_TOL_CM1 = 10.0
_VIEW_GEOMETRY = "geometry"
_VIEW_MO = "mo"
_VIEW_NORMAL = "normal"


def _compute_mo_isosurfaces(
    molden_data, mo_idx: int, isovalue: float = 0.05
) -> list[IsosurfaceMesh]:
    from .molden import evaluate_mo

    cube_data = evaluate_mo(molden_data, mo_idx)
    return extract_isosurfaces(cube_data, isovalue=isovalue)


@dataclass
class TrajectoryData:
    frames: np.ndarray  # (n_frames, n_atoms, 3)
    frame_index: int = 0


@dataclass
class NormalModeData:
    equilibrium_coords: np.ndarray  # (n_atoms, 3) in Angstrom
    mode_vectors: np.ndarray  # (n_modes, n_atoms, 3) in Angstrom
    frequencies: np.ndarray | None = None  # (n_modes,)
    mode_index: int = 0
    phase: float = 0.0
    phase_step: float = 0.30
    amplitude: float = 1.0


class MoleculeView(Widget):
    """Braille-based 3D molecule renderer."""

    can_focus = True

    def __init__(self) -> None:
        super().__init__()
        self.molecule: Molecule | None = None
        self.isosurfaces: list[IsosurfaceMesh] = []
        self.rot_matrix = rotation_matrix(-0.2, -0.5, 0.0)
        self.camera_distance = 4.0
        self.show_bonds = True
        self.show_orbitals = True
        self.dark_bg = True
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.pan_mode = False
        self.highlighted_atoms: set[int] = set()
        self.show_atom_numbers = False
        self.licorice = False
        self.vdw = False
        self.atom_scale = 0.35
        self.bond_radius = 0.08
        self.ambient = 0.50
        self.diffuse = 0.60
        self.specular = 0.40
        self.shininess = 32.0
        self._cached_strips: list[Strip] = []
        self._cached_size: tuple[int, int] = (0, 0)

    def set_molecule(
        self, molecule: Molecule, isosurfaces: list[IsosurfaceMesh] | None = None
    ) -> None:
        self.molecule = molecule
        self.isosurfaces = isosurfaces or []
        mol_radius = molecule.radius()
        self.camera_distance = max(4.0, mol_radius * 3.0)
        self._invalidate_cache()

    def _clamp_pan(self) -> None:
        if self.molecule is None:
            return
        max_pan = self.molecule.radius() * 0.5
        self.pan_x = max(-max_pan, min(max_pan, self.pan_x))
        self.pan_y = max(-max_pan, min(max_pan, self.pan_y))

    def _invalidate_cache(self) -> None:
        self._cached_size = (0, 0)
        self.refresh()

    def render_line(self, y: int) -> Strip:
        w, h = self.size.width, self.size.height
        if (w, h) != self._cached_size:
            self._rebuild(w, h)
        if 0 <= y < len(self._cached_strips):
            return self._cached_strips[y]
        return Strip.blank(w)

    def _rebuild(self, cols: int, rows: int) -> None:
        self._cached_size = (cols, rows)

        if self.molecule is None or cols == 0 or rows == 0:
            self._cached_strips = [Strip.blank(cols) for _ in range(rows)]
            return

        px_w = cols * 2
        px_h = rows * 4

        bg = (0, 0, 0) if self.dark_bg else (255, 255, 255)
        rot = self.rot_matrix

        mol = self.molecule
        if not self.show_bonds:
            mol = Molecule(atoms=mol.atoms, bonds=[])

        isos = self.isosurfaces if self.show_orbitals else None
        hl = self.highlighted_atoms if self.highlighted_atoms else None
        pixels, hit = render_scene(
            px_w,
            px_h,
            mol,
            rot,
            self.camera_distance,
            bg_color=bg,
            isosurfaces=isos,
            ssaa=1,
            pan=(self.pan_x, self.pan_y),
            highlighted_atoms=hl,
            licorice=self.licorice,
            vdw=self.vdw,
            ambient=self.ambient,
            diffuse=self.diffuse,
            specular=self.specular,
            shininess=self.shininess,
            atom_scale=self.atom_scale,
            bond_radius=self.bond_radius,
        )

        blocks = pixels.reshape(rows, 4, cols, 2, 3)
        is_on = hit.reshape(rows, 4, cols, 2)

        on_count = is_on.sum(axis=(1, 3))
        on_mask = is_on[:, :, :, :, None]
        color_sum = (blocks * on_mask).sum(axis=(1, 3))
        safe_count = np.maximum(on_count, 1)[:, :, None]
        avg_fg = (color_sum / safe_count).astype(np.uint8)

        if self.dark_bg:
            braille_bits = np.where(is_on, _BRAILLE_MAP[None, :, None, :], 0)
        else:
            # Light mode: invert braille so atom color becomes cell background
            braille_bits = np.where(~is_on, _BRAILLE_MAP[None, :, None, :], 0)
        codepoints = 0x2800 + braille_bits.sum(axis=(1, 3)).astype(np.uint32)

        any_hit = on_count > 0
        bg_style = Style(bgcolor=f"rgb({bg[0]},{bg[1]},{bg[2]})")

        # Compute atom number label positions if enabled
        label_cells: dict[tuple[int, int], tuple[str, Style]] = {}
        if self.show_atom_numbers and self.molecule is not None:
            fov = 1.5
            scale = min(px_w, px_h) / 2
            centroid = self.molecule.center()
            label_style = Style(
                color="rgb(255,255,0)" if self.dark_bg else "rgb(0,0,180)",
                bgcolor=f"rgb({bg[0]},{bg[1]},{bg[2]})",
                bold=True,
            )
            for idx, atom in enumerate(self.molecule.atoms):
                pos = rot @ (atom.position - centroid)
                pos[0] += self.pan_x
                pos[1] += self.pan_y
                pos[2] += self.camera_distance
                if pos[2] <= 0.1:
                    continue
                sx = px_w / 2 + pos[0] * fov / pos[2] * scale
                sy = px_h / 2 - pos[1] * fov / pos[2] * scale
                # Convert pixel coords to terminal cell coords
                cell_col = int(sx / 2)
                cell_row = int(sy / 4) - 1  # place label above atom
                label = str(idx + 1)
                for ci, ch in enumerate(label):
                    c = cell_col + ci
                    if 0 <= cell_row < rows and 0 <= c < cols:
                        label_cells[(cell_row, c)] = (ch, label_style)

        strips = []
        for row in range(rows):
            segments = []
            prev_style = None
            run_chars: list[str] = []
            for x in range(cols):
                if (row, x) in label_cells:
                    ch, style = label_cells[(row, x)]
                elif self.dark_bg:
                    cp = int(codepoints[row, x])
                    if cp == 0x2800:
                        style = bg_style
                        ch = " "
                    else:
                        fg = avg_fg[row, x]
                        style = Style(
                            color=f"rgb({int(fg[0])},{int(fg[1])},{int(fg[2])})",
                            bgcolor=f"rgb({bg[0]},{bg[1]},{bg[2]})",
                        )
                        ch = chr(cp)
                else:
                    cp = int(codepoints[row, x])
                    if not any_hit[row, x]:
                        style = bg_style
                        ch = " "
                    else:
                        fc = avg_fg[row, x]
                        if cp == 0x2800:
                            # Fully covered — solid atom color
                            style = Style(bgcolor=f"rgb({int(fc[0])},{int(fc[1])},{int(fc[2])})")
                            ch = " "
                        else:
                            # Edge — white dots for gaps, atom color bg
                            style = Style(
                                color=f"rgb({bg[0]},{bg[1]},{bg[2]})",
                                bgcolor=f"rgb({int(fc[0])},{int(fc[1])},{int(fc[2])})",
                            )
                            ch = chr(cp)

                if style == prev_style:
                    run_chars.append(ch)
                else:
                    if run_chars and prev_style is not None:
                        segments.append(Segment("".join(run_chars), prev_style))
                    run_chars = [ch]
                    prev_style = style
            if run_chars and prev_style is not None:
                segments.append(Segment("".join(run_chars), prev_style))
            strips.append(Strip(segments, cols))

        self._cached_strips = strips


class MoltuiApp(App):
    CSS = """
    Screen {
        layout: vertical;
    }
    #main-content {
        height: 1fr;
    }
    MoleculeView {
        width: 1fr;
    }
    """

    BINDINGS = [
        Binding("k,up", "rotate_up", "Rot up", show=False),
        Binding("j,down", "rotate_down", "Rot down", show=False),
        Binding("h,left", "rotate_left", "Rot left", show=False),
        Binding("l,right", "rotate_right", "Rot right", show=False),
        Binding("comma", "rotate_cw", ",/. roll", show=False),
        Binding("full_stop", "rotate_ccw", show=False),
        Binding("K", "zoom_in", "J/K zoom", show=False),
        Binding("J", "zoom_out", show=False),
        Binding("plus,equal_sign", "zoom_in", show=False),
        Binding("minus", "zoom_out", show=False),
        Binding("t", "toggle_mode", "Pan/Rot"),
        Binding("c", "center", "Center", show=False),
        Binding("r", "reset_view", "Reset"),
        Binding("b", "toggle_bonds", "Bonds"),
        Binding("i", "toggle_bg", "Bg"),
        Binding("v", "toggle_style", "Style"),
        Binding("number_sign", "toggle_atom_numbers", "#Nums"),
        Binding("o", "toggle_orbitals", "Orbitals"),
        Binding("V", "toggle_visual", "Visual"),
        Binding("m", "cycle_view_mode_next", "Mode"),
        Binding("M", "cycle_view_mode_prev", "Mode-", show=False),
        Binding("n", "panel_next", "Next", priority=True),
        Binding("p", "panel_prev", "Prev", priority=True),
        Binding("S", "toggle_sidebar", "Sidebar"),
        Binding("space", "toggle_playback", "Play"),
        Binding("right_square_bracket", "next_animation_step", "Frame]", show=False),
        Binding("left_square_bracket", "prev_animation_step", "[Frame", show=False),
        Binding("e", "export_png", "Export"),
        Binding("q", "quit", "Quit"),
        Binding("tab", "tab_forward", show=False, priority=True),
        Binding("shift+tab", "tab_backward", show=False, priority=True),
    ]

    def __init__(
        self,
        molecule: Molecule,
        filepath: str = "",
        isosurfaces: list[IsosurfaceMesh] | None = None,
        molden_data=None,
        current_mo: int = 0,
        trajectory_data: TrajectoryData | None = None,
        normal_mode_data: NormalModeData | None = None,
    ):
        super().__init__()
        self.molecule = molecule
        self.filepath = filepath
        self._isosurfaces = isosurfaces or []
        self.molden_data = molden_data
        self.current_mo = current_mo
        self._cube_data: CubeData | None = None
        self.isovalue: float = 0.05
        self._mo_switch_timer = None
        self._mo_pending = False
        self._mo_switch_task: asyncio.Task[None] | None = None
        self.trajectory_data = trajectory_data
        self.normal_mode_data = normal_mode_data
        if self.normal_mode_data is not None and self.normal_mode_data.mode_vectors.shape[0] > 0:
            self.normal_mode_data.mode_index = self._first_vibrational_mode_index()
        self._view_mode = _VIEW_GEOMETRY
        self._panel_hidden = False
        self._playback_timer = None
        self._is_playing = False
        self._playback_interval_sec = 0.08
        self.title = self._title_text()

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main-content"):
            yield MoleculeView()
            yield GeometryPanel()
            yield MOPanel()
            yield NormalModePanel()
            yield VisualPanel()
        yield Footer()

    def on_mount(self) -> None:
        view = self.query_one(MoleculeView)
        view.set_molecule(self.molecule, self._isosurfaces)
        panel = self.query_one(GeometryPanel)
        panel.set_molecule(self.molecule)
        if self.molden_data is not None and self.molden_data.n_mos > 0:
            md = self.molden_data
            mo_panel = self.query_one(MOPanel)
            mo_panel.set_mo_data(
                energies=md.mo_energies.tolist(),
                occupations=md.mo_occupations.tolist(),
                symmetries=md.mo_symmetries,
                spins=md.mo_spins,
                current_mo=self.current_mo,
            )
        if self.normal_mode_data is not None:
            mode_panel = self.query_one(NormalModePanel)
            mode_panel.set_mode_data(
                mode_count=self.normal_mode_data.mode_vectors.shape[0],
                frequencies=(
                    self.normal_mode_data.frequencies.tolist()
                    if self.normal_mode_data.frequencies is not None
                    else None
                ),
                current_mode=self.normal_mode_data.mode_index,
            )
        initial_mode = self._available_view_modes()[0]
        self._set_view_mode(initial_mode, reveal_panel=True)

    def _panel_is_open(self) -> bool:
        return (
            self.query_one(GeometryPanel).has_class("visible")
            or self.query_one(MOPanel).has_class("visible")
            or self.query_one(NormalModePanel).has_class("visible")
            or self.query_one(VisualPanel).has_class("visible")
        )

    def _available_view_modes(self) -> list[str]:
        modes: list[str] = []
        if self.molden_data is not None and self.molden_data.n_mos > 0:
            modes.append(_VIEW_MO)
        if self.normal_mode_data is not None:
            modes.append(_VIEW_NORMAL)
        modes.append(_VIEW_GEOMETRY)
        return modes

    def check_action(self, action: str, _parameters: tuple[object, ...]) -> bool | None:
        if action in ("toggle_mo_panel", "next_mo", "prev_mo"):
            has_mo = self.molden_data is not None and self.molden_data.n_mos > 0
            if not has_mo and not self._isosurfaces:
                return False
        if action in ("next_mo", "prev_mo", "toggle_orbitals") and self._view_mode != _VIEW_MO:
            return False
        if action == "toggle_normal_mode_panel":
            if self.normal_mode_data is None:
                return False
        if action in ("toggle_playback", "next_animation_step", "prev_animation_step"):
            if self._view_mode == _VIEW_NORMAL:
                if (
                    self.normal_mode_data is None
                    or self.normal_mode_data.mode_vectors.shape[0] == 0
                ):
                    return False
            elif self._view_mode == _VIEW_GEOMETRY:
                if self.trajectory_data is None or self.trajectory_data.frames.shape[0] <= 1:
                    return False
            else:
                return False
        if action in ("panel_next", "panel_prev"):
            if not self._panel_is_open():
                return False
        if action in ("tab_forward", "tab_backward"):
            geom = self.query_one(GeometryPanel)
            if not geom.has_class("visible"):
                return False
        if action in ("cycle_view_mode_next", "cycle_view_mode_prev"):
            if len(self._available_view_modes()) <= 1 and not self._panel_hidden:
                return False
        return True

    def _title_text(self) -> str:
        parts = [Path(self.filepath).name]
        if self._view_mode == _VIEW_GEOMETRY and self.trajectory_data is not None:
            frame_idx = self.trajectory_data.frame_index + 1
            n_frames = self.trajectory_data.frames.shape[0]
            parts.append(f"Frame {frame_idx}/{n_frames}")
        if self._view_mode == _VIEW_NORMAL and self.normal_mode_data is not None:
            mode_idx = self.normal_mode_data.mode_index + 1
            n_modes = self.normal_mode_data.mode_vectors.shape[0]
            mode_text = f"Mode {mode_idx}/{n_modes}"
            if (
                self.normal_mode_data.frequencies is not None
                and self.normal_mode_data.mode_index < len(self.normal_mode_data.frequencies)
            ):
                freq = self.normal_mode_data.frequencies[self.normal_mode_data.mode_index]
                mode_text += f" {freq:.2f} cm^-1"
            parts.append(mode_text)
        if self._view_mode == _VIEW_NORMAL and self._is_playing:
            parts.append("PLAY")
        if (
            self._view_mode == _VIEW_MO
            and self.molden_data is not None
            and self.molden_data.n_mos > 0
        ):
            md = self.molden_data
            energy = md.mo_energies[self.current_mo]
            occ = md.mo_occupations[self.current_mo]
            sym = (
                md.mo_symmetries[self.current_mo] if self.current_mo < len(md.mo_symmetries) else ""
            )
            spin_symbol = {"Alpha": "\u03b1", "Beta": "\u03b2"}
            spin = ""
            if len(set(md.mo_spins)) > 1 and self.current_mo < len(md.mo_spins):
                s = md.mo_spins[self.current_mo]
                spin = f" {spin_symbol.get(s, s)}"
            mo_str = f"MO {self.current_mo + 1}/{md.n_mos}{spin}"
            sym_str = f" {sym}" if len(set(md.mo_symmetries)) > 1 else ""
            parts.append(f"{mo_str}{sym_str} E={energy:.5f} occ={occ:.5f}")
        return " | ".join(parts)

    def _update_title(self) -> None:
        self.title = self._title_text()

    def _has_animation(self) -> bool:
        if self.trajectory_data is not None and self.trajectory_data.frames.shape[0] > 1:
            return True
        if self.normal_mode_data is not None and self.normal_mode_data.mode_vectors.shape[0] > 0:
            return True
        return False

    def _is_linear_molecule(self) -> bool:
        coords = np.array([atom.position for atom in self.molecule.atoms], dtype=np.float64)
        if coords.shape[0] <= 2:
            return True
        centered = coords - coords.mean(axis=0)
        ref_idx = None
        for i in range(centered.shape[0]):
            if np.linalg.norm(centered[i]) > 1e-8:
                ref_idx = i
                break
        if ref_idx is None:
            return True
        ref = centered[ref_idx]
        for i in range(centered.shape[0]):
            if i == ref_idx:
                continue
            if np.linalg.norm(np.cross(ref, centered[i])) > 1e-6:
                return False
        return True

    def _first_vibrational_mode_index(self) -> int:
        if self.normal_mode_data is None:
            return 0
        n_atoms = len(self.molecule.atoms)
        if n_atoms <= 1:
            return 0
        n_modes = self.normal_mode_data.mode_vectors.shape[0]
        if n_modes <= 1:
            return 0

        expected_zero_modes = 5 if self._is_linear_molecule() else 6
        freqs = self.normal_mode_data.frequencies

        # If frequencies are available, only skip rigid-body modes when the file
        # actually appears to include them at the beginning.
        if freqs is not None and len(freqs) >= expected_zero_modes:
            leading = np.asarray(freqs[:expected_zero_modes], dtype=np.float64)
            if np.all(np.abs(leading) < _ZERO_MODE_FREQ_TOL_CM1):
                return min(expected_zero_modes, n_modes - 1)

        # Many Molden writers store only vibrational modes (already trimmed).
        return 0

    def _apply_active_animation_geometry(self) -> None:
        if self.trajectory_data is not None:
            coords = self.trajectory_data.frames[self.trajectory_data.frame_index]
        elif self.normal_mode_data is not None:
            mode = self.normal_mode_data.mode_vectors[self.normal_mode_data.mode_index]
            disp = self.normal_mode_data.amplitude * np.sin(self.normal_mode_data.phase) * mode
            coords = self.normal_mode_data.equilibrium_coords + disp
        else:
            return

        for i, atom in enumerate(self.molecule.atoms):
            atom.position = coords[i].copy()
        view = self.query_one(MoleculeView)
        view._invalidate_cache()
        if self._view_mode == _VIEW_GEOMETRY:
            self.query_one(GeometryPanel).refresh_measurements()
        self._update_title()

    def _reset_normal_mode_geometry(self) -> None:
        if self.normal_mode_data is None:
            return
        self.normal_mode_data.phase = 0.0
        for i, atom in enumerate(self.molecule.atoms):
            atom.position = self.normal_mode_data.equilibrium_coords[i].copy()
        view = self.query_one(MoleculeView)
        view._invalidate_cache()
        self._update_title()

    def _animation_tick(self) -> None:
        if self.trajectory_data is not None:
            n_frames = self.trajectory_data.frames.shape[0]
            if n_frames > 1:
                self.trajectory_data.frame_index = (self.trajectory_data.frame_index + 1) % n_frames
        elif self.normal_mode_data is not None:
            self.normal_mode_data.phase += self.normal_mode_data.phase_step
        self._apply_active_animation_geometry()

    def _start_playback(self) -> None:
        if not self._has_animation():
            return
        if self._playback_timer is not None:
            return
        self._playback_timer = self.set_interval(self._playback_interval_sec, self._animation_tick)
        self._is_playing = True
        self._update_title()

    def _stop_playback(self) -> None:
        if self._playback_timer is None:
            return
        self._playback_timer.stop()
        self._playback_timer = None
        self._is_playing = False
        self._update_title()

    def action_rotate_up(self) -> None:
        view = self.query_one(MoleculeView)
        if view.pan_mode:
            view.pan_y -= view.camera_distance * 0.05
            view._clamp_pan()
        else:
            view.rot_matrix = rotation_matrix(-0.1, 0, 0) @ view.rot_matrix
        view._invalidate_cache()

    def action_rotate_down(self) -> None:
        view = self.query_one(MoleculeView)
        if view.pan_mode:
            view.pan_y += view.camera_distance * 0.05
            view._clamp_pan()
        else:
            view.rot_matrix = rotation_matrix(0.1, 0, 0) @ view.rot_matrix
        view._invalidate_cache()

    def action_rotate_left(self) -> None:
        view = self.query_one(MoleculeView)
        if view.pan_mode:
            view.pan_x += view.camera_distance * 0.05
            view._clamp_pan()
        else:
            view.rot_matrix = rotation_matrix(0, 0.1, 0) @ view.rot_matrix
        view._invalidate_cache()

    def action_rotate_right(self) -> None:
        view = self.query_one(MoleculeView)
        if view.pan_mode:
            view.pan_x -= view.camera_distance * 0.05
            view._clamp_pan()
        else:
            view.rot_matrix = rotation_matrix(0, -0.1, 0) @ view.rot_matrix
        view._invalidate_cache()

    def action_rotate_cw(self) -> None:
        view = self.query_one(MoleculeView)
        view.rot_matrix = rotation_matrix(0, 0, 0.1) @ view.rot_matrix
        view._invalidate_cache()

    def action_rotate_ccw(self) -> None:
        view = self.query_one(MoleculeView)
        view.rot_matrix = rotation_matrix(0, 0, -0.1) @ view.rot_matrix
        view._invalidate_cache()

    def action_zoom_in(self) -> None:
        view = self.query_one(MoleculeView)
        view.camera_distance = max(1.0, view.camera_distance - 0.5)
        view._invalidate_cache()

    def action_zoom_out(self) -> None:
        view = self.query_one(MoleculeView)
        view.camera_distance += 0.5
        view._invalidate_cache()

    def action_toggle_mode(self) -> None:
        view = self.query_one(MoleculeView)
        view.pan_mode = not view.pan_mode
        mode = "PAN" if view.pan_mode else "ROT"
        self.notify(f"Mode: {mode}", timeout=1)

    def action_center(self) -> None:
        view = self.query_one(MoleculeView)
        view.pan_x = 0.0
        view.pan_y = 0.0
        view._invalidate_cache()

    def action_toggle_style(self) -> None:
        view = self.query_one(MoleculeView)
        if not view.licorice and not view.vdw:
            # CPK → Licorice
            view.licorice = True
            view.vdw = False
            view.bond_radius = 0.15
        elif view.licorice:
            # Licorice → VDW
            view.licorice = False
            view.vdw = True
            view.bond_radius = 0.08
        else:
            # VDW → CPK
            view.licorice = False
            view.vdw = False
            view.bond_radius = 0.08
        vis = self.query_one(VisualPanel)
        if vis.has_class("visible"):
            vis.set_state(
                licorice=view.licorice,
                vdw=view.vdw,
                ambient=view.ambient,
                diffuse=view.diffuse,
                specular=view.specular,
                shininess=view.shininess,
                atom_scale=view.atom_scale,
                bond_radius=view.bond_radius,
                isovalue=self.isovalue,
                has_isosurfaces=bool(self._isosurfaces),
            )
        view._invalidate_cache()

    def action_toggle_atom_numbers(self) -> None:
        view = self.query_one(MoleculeView)
        view.show_atom_numbers = not view.show_atom_numbers
        view._invalidate_cache()

    def action_toggle_bonds(self) -> None:
        view = self.query_one(MoleculeView)
        view.show_bonds = not view.show_bonds
        view._invalidate_cache()

    def action_toggle_bg(self) -> None:
        view = self.query_one(MoleculeView)
        view.dark_bg = not view.dark_bg
        self.theme = "textual-dark" if view.dark_bg else "textual-light"
        view._invalidate_cache()

    def action_toggle_orbitals(self) -> None:
        view = self.query_one(MoleculeView)
        if not view.show_orbitals and self.normal_mode_data is not None and self._is_playing:
            self.notify("Stop normal-mode playback before showing orbitals", timeout=2)
            return
        view.show_orbitals = not view.show_orbitals
        view._invalidate_cache()

    def action_reset_view(self) -> None:
        view = self.query_one(MoleculeView)
        view.rot_matrix = rotation_matrix(-0.2, -0.5, 0.0)
        view.pan_x = 0.0
        view.pan_y = 0.0
        view.pan_mode = False
        view.highlighted_atoms = set()
        if view.molecule:
            view.camera_distance = max(4.0, view.molecule.radius() * 3.0)
        view._invalidate_cache()

    def action_toggle_playback(self) -> None:
        if self._is_playing:
            self._stop_playback()
            self.notify("Playback paused", timeout=1)
        else:
            self._start_playback()
            self.notify("Playback started", timeout=1)

    def action_next_animation_step(self) -> None:
        if self.trajectory_data is not None:
            n_frames = self.trajectory_data.frames.shape[0]
            if n_frames > 1:
                self.trajectory_data.frame_index = (self.trajectory_data.frame_index + 1) % n_frames
        elif self.normal_mode_data is not None:
            self.normal_mode_data.phase += self.normal_mode_data.phase_step
        self._apply_active_animation_geometry()

    def action_prev_animation_step(self) -> None:
        if self.trajectory_data is not None:
            n_frames = self.trajectory_data.frames.shape[0]
            if n_frames > 1:
                self.trajectory_data.frame_index = (self.trajectory_data.frame_index - 1) % n_frames
        elif self.normal_mode_data is not None:
            self.normal_mode_data.phase -= self.normal_mode_data.phase_step
        self._apply_active_animation_geometry()

    def action_tab_forward(self) -> None:
        """Handle Tab without enabling global focus cycling."""
        geom = self.query_one(GeometryPanel)
        if geom.has_class("visible"):
            geom.action_next_tab()

    def action_tab_backward(self) -> None:
        """Handle Shift+Tab without enabling global focus cycling."""
        geom = self.query_one(GeometryPanel)
        if geom.has_class("visible"):
            geom.action_prev_tab()

    def action_export_png(self) -> None:
        view = self.query_one(MoleculeView)
        if view.molecule is None:
            return
        self.notify("Exporting PNG...", timeout=2)
        asyncio.create_task(self._export_png_async())

    async def _export_png_async(self) -> None:
        view = self.query_one(MoleculeView)
        if view.molecule is None:
            return

        mol = view.molecule
        if not view.show_bonds:
            mol = Molecule(atoms=mol.atoms, bonds=[])
        isos = self._isosurfaces if view.show_orbitals else None

        export_w, export_h = 1600, 1200
        bg = (0, 0, 0) if view.dark_bg else (255, 255, 255)

        pixels, _ = await asyncio.to_thread(
            render_scene,
            export_w,
            export_h,
            mol,
            view.rot_matrix,
            view.camera_distance,
            bg_color=bg,
            isosurfaces=isos,
            ssaa=2,
            pan=(view.pan_x, view.pan_y),
            licorice=view.licorice,
            vdw=view.vdw,
            ambient=0.31,
            diffuse=0.72,
            specular=0.42,
            shininess=96.0,
            atom_scale=view.atom_scale,
            bond_radius=view.bond_radius,
        )

        try:
            from PIL import Image
        except ImportError:
            self.notify("Pillow required: pip install Pillow", timeout=3)
            return

        stem = Path(self.filepath).stem
        if self.molden_data is not None:
            mo_label = f".{self.current_mo + 1:03d}"
        else:
            mo_label = ""
        out_path = Path(self.filepath).parent / f"{stem}{mo_label}.png"
        img = Image.fromarray(pixels)
        await asyncio.to_thread(img.save, str(out_path))
        self.notify(f"Saved {out_path}", timeout=3)

    def on_unmount(self) -> None:
        self._stop_playback()

    def _active_panel_table(self) -> DataTable | None:
        """Return the DataTable in the currently visible panel's active tab."""
        geom = self.query_one(GeometryPanel)
        if geom.has_class("visible"):
            tabs = geom.query_one(TabbedContent)
            pane = tabs.get_pane(tabs.active)
            for dt in pane.query(DataTable):
                return dt
        mo = self.query_one(MOPanel)
        if mo.has_class("visible"):
            for dt in mo.query(DataTable):
                return dt
        mode_panel = self.query_one(NormalModePanel)
        if mode_panel.has_class("visible"):
            for dt in mode_panel.query(DataTable):
                return dt
        return None

    def _emit_active_panel_selection(self, dt: DataTable) -> None:
        geom = self.query_one(GeometryPanel)
        if geom.has_class("visible"):
            geom._emit_current_highlight(dt)
            return
        mo = self.query_one(MOPanel)
        if mo.has_class("visible"):
            if dt.row_count == 0:
                return
            rk = list(dt.rows.keys())[dt.cursor_row]
            if rk.value is not None:
                self._set_current_mo(int(rk.value))
            return
        mode_panel = self.query_one(NormalModePanel)
        if mode_panel.has_class("visible"):
            if dt.row_count == 0:
                return
            rk = list(dt.rows.keys())[dt.cursor_row]
            if rk.value is not None and self.normal_mode_data is not None:
                self.normal_mode_data.mode_index = int(rk.value)
                self.normal_mode_data.phase = 0.0
                self._apply_active_animation_geometry()

    def action_panel_next(self) -> None:
        if self.query_one(VisualPanel).has_class("visible"):
            self.screen.focus_next()
            return
        dt = self._active_panel_table()
        if dt is not None and dt.row_count > 0:
            target = min(dt.row_count - 1, dt.cursor_row + 1)
            dt.move_cursor(row=target, scroll=True)
            self._emit_active_panel_selection(dt)

    def action_panel_prev(self) -> None:
        if self.query_one(VisualPanel).has_class("visible"):
            self.screen.focus_previous()
            return
        dt = self._active_panel_table()
        if dt is not None and dt.row_count > 0:
            target = max(0, dt.cursor_row - 1)
            dt.move_cursor(row=target, scroll=True)
            self._emit_active_panel_selection(dt)

    def on_key(self, event: Key) -> None:
        # Ensure n/p panel navigation works while DataTable has focus.
        if event.key not in ("n", "p"):
            return
        if self.query_one(VisualPanel).has_class("visible"):
            return
        if not self._panel_is_open():
            return
        if event.key == "n":
            self.action_panel_next()
        else:
            self.action_panel_prev()
        event.stop()

    def action_close_panel(self) -> None:
        geom = self.query_one(GeometryPanel)
        mo = self.query_one(MOPanel)
        mode_panel = self.query_one(NormalModePanel)
        vis = self.query_one(VisualPanel)
        if (
            geom.has_class("visible")
            or mo.has_class("visible")
            or mode_panel.has_class("visible")
            or vis.has_class("visible")
        ):
            self._close_panels()
            self._panel_hidden = True
            view = self.query_one(MoleculeView)
            view._invalidate_cache()
            view.focus()

    def action_toggle_sidebar(self) -> None:
        if self._panel_is_open():
            self.action_close_panel()
            return
        self._set_view_mode(self._view_mode, reveal_panel=True)

    def _close_panels(self) -> None:
        """Close all sidebar panels and reset their state."""
        view = self.query_one(MoleculeView)
        geom = self.query_one(GeometryPanel)
        mo = self.query_one(MOPanel)
        mode_panel = self.query_one(NormalModePanel)
        vis = self.query_one(VisualPanel)
        if geom.has_class("visible"):
            geom.remove_class("visible")
            view.highlighted_atoms = set()
        if mo.has_class("visible"):
            mo.remove_class("visible")
        if mode_panel.has_class("visible"):
            mode_panel.remove_class("visible")
        if vis.has_class("visible"):
            vis.remove_class("visible")

    def action_cycle_view_mode_next(self) -> None:
        modes = self._available_view_modes()
        if len(modes) <= 1:
            if self._panel_hidden and modes:
                self._set_view_mode(modes[0], reveal_panel=True)
            return
        idx = modes.index(self._view_mode) if self._view_mode in modes else 0
        self._set_view_mode(modes[(idx + 1) % len(modes)], reveal_panel=True)

    def action_cycle_view_mode_prev(self) -> None:
        modes = self._available_view_modes()
        if len(modes) <= 1:
            if self._panel_hidden and modes:
                self._set_view_mode(modes[0], reveal_panel=True)
            return
        idx = modes.index(self._view_mode) if self._view_mode in modes else 0
        self._set_view_mode(modes[(idx - 1) % len(modes)], reveal_panel=True)

    def _focus_panel_table(self, panel_widget, emit_fn) -> None:
        for dt in panel_widget.query(DataTable):
            dt.focus()
            emit_fn(dt)
            break

    def _set_view_mode(self, mode: str, *, reveal_panel: bool = True) -> None:
        if mode not in self._available_view_modes():
            return
        view = self.query_one(MoleculeView)
        self._close_panels()
        if reveal_panel:
            self._panel_hidden = False
        self._view_mode = mode

        if mode == _VIEW_GEOMETRY:
            if self._is_playing:
                self._stop_playback()
            if self.normal_mode_data is not None:
                self._reset_normal_mode_geometry()
            panel = self.query_one(GeometryPanel)
            if not self._panel_hidden:
                panel.add_class("visible")
            view.show_orbitals = False
            if not self._panel_hidden:
                self._focus_panel_table(panel, panel._emit_current_highlight)
        elif mode == _VIEW_MO:
            if self._is_playing:
                self._stop_playback()
            if self.normal_mode_data is not None:
                self._reset_normal_mode_geometry()
            mo_panel = self.query_one(MOPanel)
            if not self._panel_hidden:
                mo_panel.add_class("visible")
            view.show_orbitals = True
            mo_panel.select_mo(self.current_mo, center=True)
            if not self._panel_hidden:
                self._focus_panel_table(mo_panel, mo_panel.emit_current_highlight)
        elif mode == _VIEW_NORMAL:
            nm = self.normal_mode_data
            if nm is None:
                return
            mode_panel = self.query_one(NormalModePanel)
            if not self._panel_hidden:
                mode_panel.add_class("visible")
            view.show_orbitals = False
            mode_panel.select_mode(nm.mode_index, center=True)
            if not self._panel_hidden:
                self._focus_panel_table(mode_panel, mode_panel.emit_current_highlight)
            if not self._is_playing:
                self._start_playback()

        view._invalidate_cache()
        self._update_title()

    def action_toggle_mo_panel(self) -> None:
        if self.molden_data is None or self.molden_data.n_mos == 0:
            self.notify("No MO data (molden files only)", timeout=2)
            return
        self._set_view_mode(_VIEW_MO)

    def action_toggle_normal_mode_panel(self) -> None:
        if self.normal_mode_data is None:
            self.notify("No normal mode data", timeout=2)
            return
        self._set_view_mode(_VIEW_NORMAL)

    def action_toggle_visual(self) -> None:
        vis = self.query_one(VisualPanel)
        was_visible = vis.has_class("visible")
        self._close_panels()
        view = self.query_one(MoleculeView)
        if not was_visible:
            has_isosurfaces = bool(self._isosurfaces)
            vis.set_state(
                licorice=view.licorice,
                vdw=view.vdw,
                ambient=view.ambient,
                diffuse=view.diffuse,
                specular=view.specular,
                shininess=view.shininess,
                atom_scale=view.atom_scale,
                bond_radius=view.bond_radius,
                isovalue=self.isovalue,
                has_isosurfaces=has_isosurfaces,
            )
            vis.add_class("visible")
            if has_isosurfaces:
                focus_target = vis.query_one("#slider-isovalue", Slider)
            else:
                focus_target = vis.query_one(RadioSet)
            self.call_after_refresh(self.set_focus, focus_target)
        else:
            view.focus()
        view._invalidate_cache()

    def on_visual_panel_style_changed(self, event: VisualPanel.StyleChanged) -> None:
        view = self.query_one(MoleculeView)
        view.licorice = event.licorice
        view.vdw = event.vdw
        # Switch bond_radius to a sensible default for the style
        view.bond_radius = 0.15 if event.licorice else 0.08
        vis = self.query_one(VisualPanel)
        if vis.has_class("visible"):
            vis.set_state(
                licorice=view.licorice,
                vdw=view.vdw,
                ambient=view.ambient,
                diffuse=view.diffuse,
                specular=view.specular,
                shininess=view.shininess,
                atom_scale=view.atom_scale,
                bond_radius=view.bond_radius,
                isovalue=self.isovalue,
                has_isosurfaces=bool(self._isosurfaces),
            )
        view._invalidate_cache()

    def on_visual_panel_isovalue_changed(self, event: VisualPanel.IsovalueChanged) -> None:
        self.isovalue = event.isovalue
        if self._cube_data is not None:
            self._isosurfaces = extract_isosurfaces(self._cube_data, isovalue=self.isovalue)
            view = self.query_one(MoleculeView)
            view.isosurfaces = self._isosurfaces
            view._invalidate_cache()
        elif self.molden_data is not None and self.molden_data.n_mos > 0:
            self._switch_mo()

    def on_visual_panel_lighting_changed(self, event: VisualPanel.LightingChanged) -> None:
        view = self.query_one(MoleculeView)
        view.ambient = event.ambient
        view.diffuse = event.diffuse
        view.specular = event.specular
        view.shininess = event.shininess
        view._invalidate_cache()

    def on_visual_panel_size_changed(self, event: VisualPanel.SizeChanged) -> None:
        view = self.query_one(MoleculeView)
        view.atom_scale = event.atom_scale
        view.bond_radius = event.bond_radius
        view._invalidate_cache()

    def on_mopanel_moselected(self, event: MOPanel.MOSelected) -> None:
        self._set_current_mo(event.mo_index)

    def on_normalmodepanel_modeselected(self, event: NormalModePanel.ModeSelected) -> None:
        if self.normal_mode_data is None:
            return
        self.normal_mode_data.mode_index = event.mode_index
        self.normal_mode_data.phase = 0.0
        self._apply_active_animation_geometry()

    def on_geometry_panel_highlight_atoms(self, event: GeometryPanel.HighlightAtoms) -> None:
        view = self.query_one(MoleculeView)
        view.highlighted_atoms = set(event.atom_indices)
        view._invalidate_cache()

    def _set_current_mo(self, mo_idx: int) -> None:
        """Single entry point for all MO changes."""
        if self.molden_data is None or self.molden_data.n_mos == 0:
            return
        mo_idx = max(0, min(mo_idx, self.molden_data.n_mos - 1))
        if mo_idx == self.current_mo:
            return
        self.current_mo = mo_idx
        self._debounced_switch_mo()
        mo_panel = self.query_one(MOPanel)
        mo_panel.select_mo(mo_idx)

    def action_next_mo(self) -> None:
        mo_panel = self.query_one(MOPanel)
        next_mo = mo_panel.adjacent_mo(self.current_mo, 1)
        if next_mo is not None:
            self._set_current_mo(next_mo)

    def action_prev_mo(self) -> None:
        mo_panel = self.query_one(MOPanel)
        prev_mo = mo_panel.adjacent_mo(self.current_mo, -1)
        if prev_mo is not None:
            self._set_current_mo(prev_mo)

    def _debounced_switch_mo(self) -> None:
        self._update_title()
        if self._mo_switch_timer is not None or self._mo_switch_task is not None:
            # Cooling down or already computing: remember there is a newer request.
            self._mo_pending = True
            return
        self._switch_mo()

    def _mo_cooldown_done(self) -> None:
        self._mo_switch_timer = None
        if self._mo_pending:
            self._mo_pending = False
            self._switch_mo()

    def _switch_mo(self) -> None:
        if self.molden_data is None or self.molden_data.n_mos == 0:
            return
        self._mo_switch_task = asyncio.create_task(self._switch_mo_async(self.current_mo))

    async def _switch_mo_async(self, target_mo: int) -> None:
        try:
            if self.molden_data is None:
                return
            isosurfaces = await asyncio.to_thread(
                _compute_mo_isosurfaces, self.molden_data, target_mo, self.isovalue
            )
            # If user moved again while this was computing, skip stale frame.
            if target_mo != self.current_mo:
                return
            self._isosurfaces = isosurfaces
            view = self.query_one(MoleculeView)
            view.isosurfaces = self._isosurfaces
            view._invalidate_cache()
            self._update_title()
        finally:
            self._mo_switch_task = None
            self._mo_switch_timer = self.set_timer(0.3, self._mo_cooldown_done)


def _detect_filetype(filepath: str) -> str:
    """Detect file type from content, falling back to extension."""
    path = Path(filepath)
    suffix = path.suffix.lower()
    name_lower = path.name.lower()
    if suffix == ".gbw":
        return "gbw"
    if suffix == ".hess":
        return "hess"
    if suffix in (".zmat", ".zmatrix"):
        return "zmat"
    if suffix == ".molden" or name_lower.endswith(".molden.input"):
        return "molden"
    with open(filepath) as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            if "[molden format]" in stripped.lower():
                return "molden"
            if stripped.lower() == "$orca_hessian_file":
                return "hess"
            try:
                int(stripped)
                return "xyz"
            except ValueError:
                pass
            return "cube"
    return suffix.lstrip(".")


def _convert_gbw_to_molden(gbw_path: str | Path) -> Path:
    """Convert an ORCA .gbw file to a temporary .molden file using orca_2mkl."""
    import shutil
    import subprocess
    import tempfile

    if shutil.which("orca_2mkl") is None:
        raise RuntimeError(
            "orca_2mkl not found on PATH. "
            "Install ORCA or add orca_2mkl to your PATH to open .gbw files."
        )

    gbw = Path(gbw_path).resolve()
    stem = gbw.stem

    tmpdir = tempfile.mkdtemp(prefix="moltui_gbw_")
    tmp_gbw = Path(tmpdir) / gbw.name
    shutil.copy2(gbw, tmp_gbw)

    try:
        result = subprocess.run(
            ["orca_2mkl", stem, "-molden"],
            cwd=tmpdir,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise RuntimeError(f"Failed to run orca_2mkl: {exc}") from exc

    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"orca_2mkl failed (exit {result.returncode}):\n{stderr}")

    molden_file = Path(tmpdir) / f"{stem}.molden.input"
    if not molden_file.exists():
        raise RuntimeError(f"orca_2mkl did not produce expected output file: {molden_file.name}")

    return molden_file


def run():
    import argparse

    parser = argparse.ArgumentParser(
        prog="moltui",
        description="Terminal-based 3D molecular viewer",
    )
    parser.add_argument(
        "file", help="molecular structure file (XYZ, Cube, Molden, ORCA .hess, or ORCA .gbw)"
    )
    parsed = parser.parse_args()

    filepath = parsed.file
    filetype = _detect_filetype(filepath)
    isosurfaces: list[IsosurfaceMesh] = []
    molden_data = None
    current_mo = 0
    trajectory_data: TrajectoryData | None = None
    normal_mode_data: NormalModeData | None = None
    gbw_tmpdir: Path | None = None

    if filetype == "gbw":
        print(f"Converting {Path(filepath).name} to Molden via orca_2mkl...")
        try:
            molden_file = _convert_gbw_to_molden(filepath)
        except RuntimeError as exc:
            import sys

            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
        gbw_tmpdir = molden_file.parent
        filepath = str(molden_file)
        filetype = "molden"

    try:
        cube_data_for_app: CubeData | None = None
        if filetype == "cube":
            cube_data = parse_cube_data(filepath)
            molecule = cube_data.molecule
            isosurfaces = extract_isosurfaces(cube_data)
            cube_data_for_app = cube_data
        elif filetype == "molden":
            from .molden import evaluate_mo, load_molden_data

            molden_data = load_molden_data(filepath)
            molecule = molden_data.molecule
            if molden_data.normal_modes is not None:
                eq_coords = np.array([atom.position.copy() for atom in molecule.atoms])
                normal_mode_data = NormalModeData(
                    equilibrium_coords=eq_coords,
                    mode_vectors=molden_data.normal_modes,
                    frequencies=molden_data.mode_frequencies,
                )
            if molden_data.n_mos > 0:
                current_mo = molden_data.homo_idx
                cube_data = evaluate_mo(molden_data, current_mo)
                isosurfaces = extract_isosurfaces(cube_data)
        elif filetype == "hess":
            hess_data = parse_orca_hess_data(filepath)
            molecule = hess_data.molecule
            if hess_data.normal_modes is not None:
                eq_coords = np.array([atom.position.copy() for atom in molecule.atoms])
                normal_mode_data = NormalModeData(
                    equilibrium_coords=eq_coords,
                    mode_vectors=hess_data.normal_modes,
                    frequencies=hess_data.frequencies,
                )
        elif filetype == "xyz":
            traj = parse_xyz_trajectory(filepath)
            molecule = traj.molecule
            trajectory_data = TrajectoryData(frames=traj.frames)
        else:
            molecule = load_molecule(filepath)

        app = MoltuiApp(
            molecule=molecule,
            filepath=parsed.file,
            isosurfaces=isosurfaces,
            molden_data=molden_data,
            current_mo=current_mo,
            trajectory_data=trajectory_data,
            normal_mode_data=normal_mode_data,
        )
        app._cube_data = cube_data_for_app
        app.run()
    except ValueError as exc:
        import sys

        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        if gbw_tmpdir is not None:
            import shutil

            shutil.rmtree(gbw_tmpdir, ignore_errors=True)
