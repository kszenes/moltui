from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from .molden import OrbitalData

import numpy as np
from rich.segment import Segment
from rich.style import Style
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.css.query import NoMatches
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
from .visual_panel import Slider, Toggle, VisualPanel

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


def _filter_rigid_body_modes(
    mode_vectors: np.ndarray,
    frequencies: np.ndarray | None,
    threshold_cm1: float = _ZERO_MODE_FREQ_TOL_CM1,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Drop translational/rotational modes (|freq| below threshold)."""
    if frequencies is None or len(frequencies) == 0:
        return mode_vectors, frequencies
    keep = np.abs(np.asarray(frequencies, dtype=np.float64)) >= threshold_cm1
    if keep.all():
        return mode_vectors, frequencies
    return mode_vectors[keep], frequencies[keep]


_VIEW_GEOMETRY = "geometry"
_VIEW_MO = "mo"
_VIEW_NORMAL = "normal"
_PANEL_NAV_DEBOUNCE_SEC = 0.06


class ExportRenderKwargs(TypedDict):
    ssaa: int
    pan: tuple[float, float]
    licorice: bool
    vdw: bool
    ambient: float
    diffuse: float
    specular: float
    shininess: float
    atom_scale: float
    bond_radius: float
    cell_dash: tuple[int, int] | None
    cell_line_width: int


def _export_render_kwargs(view: "MoleculeView") -> ExportRenderKwargs:
    """Build export-time renderer settings from current view state."""
    return {
        "ssaa": 2,
        "pan": (view.pan_x, view.pan_y),
        "licorice": view.licorice,
        "vdw": view.vdw,
        "ambient": view.ambient,
        "diffuse": view.diffuse,
        "specular": view.specular,
        "shininess": view.shininess,
        "atom_scale": view.atom_scale,
        "bond_radius": view.bond_radius,
        "cell_dash": (5, 3),
        "cell_line_width": 2,
    }


def _build_view_render_scene(
    view: "MoleculeView",
) -> tuple[Molecule, list[IsosurfaceMesh] | None, bool]:
    """Build the molecule/isosurface scene shared by live rendering and export."""
    if view.molecule is None:
        raise ValueError("view must have a molecule before rendering")

    mol = view.molecule
    if mol.lattice is not None and view.show_replication:
        mol = mol.with_bonded_periodic_images()
    elif mol.lattice is not None:
        mol = mol.with_in_cell_bonds()

    cell_lattice = mol.lattice if view.show_cell else None
    if not view.show_bonds:
        mol = Molecule(atoms=mol.atoms, bonds=[], lattice=cell_lattice, pbc=mol.pbc)
    elif cell_lattice is not mol.lattice:
        mol = Molecule(atoms=mol.atoms, bonds=mol.bonds, lattice=cell_lattice, pbc=mol.pbc)

    isos = view.isosurfaces if view.show_orbitals else None
    return mol, isos, view.show_cell


def _compute_mo_isosurfaces(
    orbital_data,
    mo_idx: int,
    isovalue: float = 0.05,
    grid_shape: tuple[int, int, int] = (60, 60, 60),
) -> list[IsosurfaceMesh]:
    from .molden import evaluate_mo

    cube_data = evaluate_mo(orbital_data, mo_idx, grid_shape=grid_shape)
    return extract_isosurfaces(cube_data, isovalue=isovalue)


def _compute_parent_indices(base: Molecule, augmented: Molecule) -> list[int]:
    """Map each augmented atom back to its originating in-cell atom index."""
    if base.lattice is None:
        return list(range(len(augmented.atoms)))
    inv = np.linalg.inv(base.lattice)
    ref = np.array([a.position @ inv for a in base.atoms], dtype=np.float64)
    ref_mod = ref - np.floor(ref + 1e-6)
    parents: list[int] = []
    for atom in augmented.atoms:
        frac = atom.position @ inv
        frac_mod = frac - np.floor(frac + 1e-6)
        delta = ref_mod - frac_mod
        delta -= np.round(delta)
        d2 = np.einsum("ij,ij->i", delta, delta)
        parents.append(int(np.argmin(d2)))
    return parents


def _position_key(position: np.ndarray) -> tuple[int, int, int]:
    """Stable key for matching rendered atoms back to selected display atoms."""
    return tuple(np.round(np.asarray(position, dtype=np.float64) * 1000).astype(int).tolist())


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


@dataclass
class DisplayGeometry:
    molecule: Molecule
    parent_indices: list[int] | None = None

    def parent_index(self, atom_index: int) -> int:
        if self.parent_indices is None or not (0 <= atom_index < len(self.parent_indices)):
            return atom_index
        return self.parent_indices[atom_index]


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
        self.show_cell = True
        self.show_replication = True
        self.show_orbitals = True
        self.dark_bg = True
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.pan_mode = False
        self.highlighted_atoms: set[int] = set()
        self.highlighted_display_positions: set[tuple[int, int, int]] = set()
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
        scene_radius = self._scene_radius()
        self.camera_distance = max(4.0, scene_radius * 3.0)
        self._invalidate_cache()

    def _scene_radius(self) -> float:
        if self.molecule is None:
            return 1.0
        radius = self.molecule.radius()
        if self.molecule.lattice is None:
            return radius
        lattice = self.molecule.lattice
        center = 0.5 * (lattice[0] + lattice[1] + lattice[2])
        corners = np.array(
            [
                s0 * lattice[0] + s1 * lattice[1] + s2 * lattice[2]
                for s0 in (0, 1)
                for s1 in (0, 1)
                for s2 in (0, 1)
            ],
            dtype=np.float64,
        )
        cell_radius = float(np.linalg.norm(corners - center, axis=1).max())
        return max(radius, cell_radius)

    def _clamp_pan(self) -> None:
        if self.molecule is None:
            return
        max_pan = self._scene_radius() * 0.5
        self.pan_x = max(-max_pan, min(max_pan, self.pan_x))
        self.pan_y = max(-max_pan, min(max_pan, self.pan_y))

    def _invalidate_cache(self) -> None:
        self._cached_size = (0, 0)
        self.refresh()

    def _compute_parent_indices(self, augmented: Molecule) -> list[int]:
        """For each atom in `augmented`, return the index of the in-cell parent."""
        if self.molecule is None:
            return list(range(len(augmented.atoms)))
        return _compute_parent_indices(self.molecule, augmented)

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

        mol, isos, show_cell = _build_view_render_scene(self)
        hl: set[int] | None = None
        if self.highlighted_display_positions:
            hl_set = {
                idx
                for idx, atom in enumerate(mol.atoms)
                if _position_key(atom.position) in self.highlighted_display_positions
            }
            hl = hl_set or None
        elif self.highlighted_atoms:
            hl = self.highlighted_atoms
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
            show_cell=show_cell,
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
            # Match the renderer's centroid: cell center when periodic, else mean.
            if mol.lattice is not None:
                centroid = 0.5 * (mol.lattice[0] + mol.lattice[1] + mol.lattice[2])
            else:
                centroid = mol.center()
            label_style = Style(
                color="rgb(255,255,0)" if self.dark_bg else "rgb(0,0,180)",
                bgcolor=f"rgb({bg[0]},{bg[1]},{bg[2]})",
                bold=True,
            )
            # Map each rendered atom (originals + ghosts) to its parent index
            # so ghost atoms inherit their parent's number.
            parent_lookup = self._compute_parent_indices(mol)
            for idx, atom in enumerate(mol.atoms):
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
                label_idx = parent_lookup[idx] if idx < len(parent_lookup) else idx
                label = str(label_idx + 1)
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
        Binding("K,plus,equals_sign", "zoom_in", "J/K zoom", show=False),
        Binding("J,minus", "zoom_out", show=False),
        Binding("t", "toggle_mode", "Pan/Rot"),
        Binding("c", "center", "Center", show=False),
        Binding("r", "reset_view", "Reset"),
        Binding("B", "toggle_bonds", "Bonds"),
        Binding("b", "toggle_replication", "Ghosts"),
        Binding("i", "toggle_bg", "Bg"),
        Binding("v", "toggle_style", "Style"),
        Binding("number_sign", "toggle_atom_numbers", "#Nums"),
        Binding("o", "toggle_orbitals", "Orbitals"),
        Binding("V", "toggle_visual", "Visual"),
        Binding("m", "cycle_view_mode_next", "Mode"),
        Binding("M", "cycle_view_mode_prev", "Mode-", show=False),
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
        orbital_data=None,
        current_mo: int = 0,
        trajectory_data: TrajectoryData | None = None,
        normal_mode_data: NormalModeData | None = None,
    ):
        super().__init__()
        self.molecule = molecule
        self.filepath = filepath
        self._isosurfaces = isosurfaces or []
        self._display_geometry: DisplayGeometry | None = None
        self.orbital_data = orbital_data
        self.current_mo = current_mo
        self._cube_data: CubeData | None = None
        self.isovalue: float = 0.05
        self._mo_switch_timer = None
        self._mo_pending = False
        self._mo_switch_task: asyncio.Task[None] | None = None
        self.trajectory_data = trajectory_data
        self.normal_mode_data = normal_mode_data
        self._startup_toast: str | None = None
        self._view_mode = _VIEW_GEOMETRY
        self._panel_hidden = False
        self._playback_timer = None
        self._is_playing = False
        self._playback_interval_sec = 1.0 / 12.0
        self._last_panel_nav_at = 0.0
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

    def _build_display_geometry(self, *, replicate: bool | None = None) -> DisplayGeometry:
        """Build the molecule shown in the geometry sidebar."""
        mol = self.molecule
        if mol.lattice is None:
            return DisplayGeometry(molecule=mol)

        if replicate is None:
            view = self._query_molecule_view()
            replicate = bool(view and view.show_replication)

        if replicate:
            augmented = mol.with_bonded_periodic_images()
            return DisplayGeometry(
                molecule=augmented,
                parent_indices=_compute_parent_indices(mol, augmented),
            )

        return DisplayGeometry(molecule=mol.with_in_cell_bonds())

    def on_mount(self) -> None:
        view = self.query_one(MoleculeView)
        view.set_molecule(self.molecule, self._isosurfaces)
        panel = self.query_one(GeometryPanel)
        self._display_geometry = self._build_display_geometry()
        panel.set_molecule(self._display_geometry.molecule, self._display_geometry.parent_indices)
        if self._has_orbital_mos():
            md = self.orbital_data
            assert md is not None
            mo_panel = self.query_one(MOPanel)
            mo_panel.set_mo_data(
                energies=md.mo_energies.tolist(),
                occupations=md.mo_occupations.tolist(),
                symmetries=md.mo_symmetries,
                spins=md.mo_spins,
                current_mo=self.current_mo,
                has_energies=md.has_mo_energies,
                has_occupations=md.has_mo_occupations,
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
        if self._has_cube_mo() and not self._has_orbital_mos():
            self._set_view_mode(_VIEW_MO, reveal_panel=False)
        else:
            initial_mode = self._available_view_modes()[0]
            self._set_view_mode(initial_mode, reveal_panel=True)
        if self._has_trajectory:
            self._start_playback()
        if self._startup_toast is not None:
            self.notify(self._startup_toast, severity="warning", timeout=5)

    def _panel_is_open(self) -> bool:
        return (
            self.query_one(GeometryPanel).has_class("visible")
            or self.query_one(MOPanel).has_class("visible")
            or self.query_one(NormalModePanel).has_class("visible")
            or self.query_one(VisualPanel).has_class("visible")
        )

    def _has_orbital_mos(self) -> bool:
        return self.orbital_data is not None and self.orbital_data.n_mos > 0

    def _has_cube_mo(self) -> bool:
        return (
            self.orbital_data is None
            and self._cube_data is not None
            and Path(self.filepath).suffix.lower() == ".cube"
        )

    def _available_view_modes(self) -> list[str]:
        modes: list[str] = []
        if self._has_orbital_mos() or self._has_cube_mo():
            modes.append(_VIEW_MO)
        if self.normal_mode_data is not None:
            modes.append(_VIEW_NORMAL)
        modes.append(_VIEW_GEOMETRY)
        return modes

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        _ = parameters
        if action == "toggle_mo_panel":
            if not self._has_orbital_mos() and not self._has_cube_mo():
                return False
        if action in ("next_mo", "prev_mo"):
            if not self._has_orbital_mos():
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
            if len(self._available_view_modes()) <= 1 and self._mode_panel_active():
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
            and self.orbital_data is not None
            and self.orbital_data.n_mos > 0
        ):
            md = self.orbital_data
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
            detail_parts: list[str] = []
            if md.has_mo_energies:
                detail_parts.append(f"E={md.mo_energies[self.current_mo]:.5f}")
            if md.has_mo_occupations:
                detail_parts.append(f"occ={md.mo_occupations[self.current_mo]:.5f}")
            detail_str = " ".join(detail_parts)
            title = f"{mo_str}{sym_str}"
            if detail_str:
                title += f" {detail_str}"
            parts.append(title)
        return " | ".join(parts)

    def _update_title(self) -> None:
        self.title = self._title_text()

    def _query_molecule_view(self) -> MoleculeView | None:
        """Return MoleculeView when mounted, otherwise None during teardown."""
        try:
            return self.query_one(MoleculeView)
        except NoMatches:
            return None

    @property
    def _has_normal_modes(self) -> bool:
        return self.normal_mode_data is not None

    @property
    def _has_trajectory(self) -> bool:
        return self.trajectory_data is not None and self.trajectory_data.frames.shape[0] > 1

    def _sync_visual_panel(self, view: MoleculeView) -> None:
        vis = self.query_one(VisualPanel)
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
            has_normal_modes=self._has_normal_modes,
            has_trajectory=self._has_trajectory,
            vibrational_phase_step=self.normal_mode_data.phase_step
            if self.normal_mode_data is not None
            else 0.0,
            trajectory_fps=1.0 / self._playback_interval_sec,
            has_lattice=view.molecule is not None and view.molecule.lattice is not None,
            show_cell=view.show_cell,
        )

    def _has_animation(self) -> bool:
        if self._view_mode == _VIEW_GEOMETRY:
            return self._has_trajectory
        if self._view_mode == _VIEW_NORMAL:
            return self._has_normal_modes
        return False

    def _apply_active_animation_geometry(self) -> None:
        recompute_bonds = False
        if self._view_mode == _VIEW_GEOMETRY and self.trajectory_data is not None:
            coords = self.trajectory_data.frames[self.trajectory_data.frame_index]
            recompute_bonds = True
        elif self._view_mode == _VIEW_NORMAL and self.normal_mode_data is not None:
            mode = self.normal_mode_data.mode_vectors[self.normal_mode_data.mode_index]
            disp = self.normal_mode_data.amplitude * np.sin(self.normal_mode_data.phase) * mode
            coords = self.normal_mode_data.equilibrium_coords + disp
        else:
            return

        for i, atom in enumerate(self.molecule.atoms):
            atom.position = coords[i].copy()
        if recompute_bonds:
            self.molecule.detect_bonds_auto()
        view = self._query_molecule_view()
        if view is None:
            # Timer callbacks can race with app teardown in tests/CI.
            self._stop_playback()
            return
        view._invalidate_cache()
        if self._view_mode == _VIEW_GEOMETRY:
            try:
                self._refresh_geometry_panel()
            except NoMatches:
                self._stop_playback()
                return
        self._update_title()

    def _reset_normal_mode_geometry(self) -> None:
        if self.normal_mode_data is None:
            return
        self.normal_mode_data.phase = 0.0
        for i, atom in enumerate(self.molecule.atoms):
            atom.position = self.normal_mode_data.equilibrium_coords[i].copy()
        view = self._query_molecule_view()
        if view is None:
            return
        if self.molecule.lattice is not None:
            self._refresh_geometry_panel()
        view._invalidate_cache()
        self._update_title()

    def _apply_orbital_geometry(self) -> None:
        if self.orbital_data is None or not hasattr(self.orbital_data, "molecule"):
            return
        orbital_molecule = self.orbital_data.molecule
        if len(orbital_molecule.atoms) != len(self.molecule.atoms):
            return
        for atom, orbital_atom in zip(self.molecule.atoms, orbital_molecule.atoms, strict=True):
            atom.position = orbital_atom.position.copy()
        self.molecule.detect_bonds_auto()
        view = self._query_molecule_view()
        if view is None:
            return
        view._invalidate_cache()
        self._update_title()

    def _animation_tick(self) -> None:
        if self._view_mode == _VIEW_GEOMETRY and self.trajectory_data is not None:
            n_frames = self.trajectory_data.frames.shape[0]
            if n_frames > 1:
                self.trajectory_data.frame_index = (self.trajectory_data.frame_index + 1) % n_frames
        elif self._view_mode == _VIEW_NORMAL and self.normal_mode_data is not None:
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
            self._sync_visual_panel(view)
        view._invalidate_cache()

    def action_toggle_atom_numbers(self) -> None:
        view = self.query_one(MoleculeView)
        view.show_atom_numbers = not view.show_atom_numbers
        view._invalidate_cache()

    def action_toggle_bonds(self) -> None:
        view = self.query_one(MoleculeView)
        view.show_bonds = not view.show_bonds
        view._invalidate_cache()

    def action_toggle_replication(self) -> None:
        if self.molecule is None or self.molecule.lattice is None:
            return
        view = self.query_one(MoleculeView)
        view.show_replication = not view.show_replication
        view.highlighted_display_positions = set()
        view._invalidate_cache()
        self._refresh_geometry_panel()

    def _refresh_geometry_panel(self) -> None:
        try:
            panel = self.query_one(GeometryPanel)
        except NoMatches:
            return
        self._display_geometry = self._build_display_geometry()
        panel.set_molecule(self._display_geometry.molecule, self._display_geometry.parent_indices)
        if panel.has_class("visible"):
            tabs = panel.query_one(TabbedContent)
            panel._emit_current_highlight(panel._table_for_tab(tabs.active))

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
        view.highlighted_display_positions = set()
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

        mol, isos, show_cell = _build_view_render_scene(view)

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
            show_cell=show_cell,
            **_export_render_kwargs(view),
        )

        try:
            from PIL import Image
        except ImportError:
            self.notify("Pillow required: pip install Pillow", timeout=3)
            return

        stem = Path(self.filepath).stem
        if self.orbital_data is not None:
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

    def _half_page_rows(self, dt: DataTable) -> int:
        viewport_rows = max(1, dt.size.height - 1)
        return max(1, viewport_rows // 2)

    def _move_active_table_by(self, delta_rows: int) -> bool:
        dt = self._active_panel_table()
        if dt is None or dt.row_count == 0:
            return False
        target = max(0, min(dt.row_count - 1, dt.cursor_row + delta_rows))
        dt.move_cursor(row=target, scroll=True)
        self._emit_active_panel_selection(dt)
        return True

    def _jump_active_table_to(self, row: int) -> bool:
        dt = self._active_panel_table()
        if dt is None or dt.row_count == 0:
            return False
        target = max(0, min(dt.row_count - 1, row))
        dt.move_cursor(row=target, scroll=True)
        self._emit_active_panel_selection(dt)
        return True

    def action_panel_next(self) -> None:
        if self.query_one(VisualPanel).has_class("visible"):
            self.screen.focus_next()
            return
        self._move_active_table_by(1)

    def action_panel_prev(self) -> None:
        if self.query_one(VisualPanel).has_class("visible"):
            self.screen.focus_previous()
            return
        self._move_active_table_by(-1)

    def on_key(self, event: Key) -> None:
        # Ensure table navigation keys work while DataTable has focus.
        if event.key not in ("n", "p", "d", "u", "ctrl+d", "ctrl+u", "g", "G"):
            return
        if not self._panel_is_open():
            return
        if event.key in ("n", "p"):
            now = time.monotonic()
            if now - self._last_panel_nav_at < _PANEL_NAV_DEBOUNCE_SEC:
                event.stop()
                return
            self._last_panel_nav_at = now
            if event.key == "n":
                self.action_panel_next()
            else:
                self.action_panel_prev()
            event.stop()
            return
        if self.query_one(VisualPanel).has_class("visible"):
            return
        if event.key in ("d", "ctrl+d"):
            dt = self._active_panel_table()
            if dt is not None:
                self._move_active_table_by(self._half_page_rows(dt))
                event.stop()
            return
        if event.key in ("u", "ctrl+u"):
            dt = self._active_panel_table()
            if dt is not None:
                self._move_active_table_by(-self._half_page_rows(dt))
                event.stop()
            return
        if event.key == "g":
            if self._jump_active_table_to(0):
                event.stop()
            return
        if event.key == "G":
            dt = self._active_panel_table()
            if dt is not None and self._jump_active_table_to(dt.row_count - 1):
                event.stop()
            return

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
            self._sidebar_was_visual = vis.has_class("visible")
            self._close_panels()
            self._panel_hidden = True
            view = self.query_one(MoleculeView)
            view._invalidate_cache()
            view.focus()

    def action_toggle_sidebar(self) -> None:
        if self._panel_is_open():
            self.action_close_panel()
            return
        if getattr(self, "_sidebar_was_visual", False):
            self._sidebar_was_visual = False
            self.action_toggle_visual()
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
            view.highlighted_display_positions = set()
        if mo.has_class("visible"):
            mo.remove_class("visible")
        if mode_panel.has_class("visible"):
            mode_panel.remove_class("visible")
        if vis.has_class("visible"):
            vis.remove_class("visible")

    def _mode_panel_active(self) -> bool:
        """True when the panel for the current view mode is visible.

        The visual panel is a separate sidebar and is not counted as the
        "mode panel"; cycling should still surface the mode panel even if
        the visual panel happens to be open.
        """
        if self._panel_hidden:
            return False
        try:
            if self._view_mode == _VIEW_GEOMETRY:
                return self.query_one(GeometryPanel).has_class("visible")
            if self._view_mode == _VIEW_MO:
                return self.query_one(MOPanel).has_class("visible")
            if self._view_mode == _VIEW_NORMAL:
                return self.query_one(NormalModePanel).has_class("visible")
        except NoMatches:
            return False
        return False

    def action_cycle_view_mode_next(self) -> None:
        modes = self._available_view_modes()
        if not modes:
            return
        if len(modes) <= 1:
            if not self._mode_panel_active():
                self._set_view_mode(modes[0], reveal_panel=True)
            return
        idx = modes.index(self._view_mode) if self._view_mode in modes else 0
        self._set_view_mode(modes[(idx + 1) % len(modes)], reveal_panel=True)

    def action_cycle_view_mode_prev(self) -> None:
        modes = self._available_view_modes()
        if not modes:
            return
        if len(modes) <= 1:
            if not self._mode_panel_active():
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
            if self.trajectory_data is not None:
                self._apply_active_animation_geometry()
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
            self._apply_orbital_geometry()
            view.show_orbitals = True
            if self._has_orbital_mos():
                mo_panel = self.query_one(MOPanel)
                if not self._panel_hidden:
                    mo_panel.add_class("visible")
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
        if not self._has_orbital_mos() and not self._has_cube_mo():
            self.notify("No MO data (molden files only)", timeout=2)
            return
        self._set_view_mode(_VIEW_MO, reveal_panel=self._has_orbital_mos())

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
            self._sync_visual_panel(view)
            vis.add_class("visible")
            focus_target = next(
                (
                    widget
                    for widget in vis.query("*")
                    if isinstance(widget, (Toggle, Slider)) and widget.display
                ),
                vis.query_one(RadioSet),
            )
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
            self._sync_visual_panel(view)
        view._invalidate_cache()

    def on_visual_panel_isovalue_changed(self, event: VisualPanel.IsovalueChanged) -> None:
        self.isovalue = event.isovalue
        if self._cube_data is not None:
            self._isosurfaces = extract_isosurfaces(self._cube_data, isovalue=self.isovalue)
            view = self.query_one(MoleculeView)
            view.isosurfaces = self._isosurfaces
            view._invalidate_cache()
        elif self.orbital_data is not None and self.orbital_data.n_mos > 0:
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

    def on_visual_panel_vibration_speed_changed(
        self, event: VisualPanel.VibrationSpeedChanged
    ) -> None:
        if self.normal_mode_data is None:
            return
        self.normal_mode_data.phase_step = event.phase_step

    def on_visual_panel_trajectory_speed_changed(
        self, event: VisualPanel.TrajectorySpeedChanged
    ) -> None:
        self._playback_interval_sec = 1.0 / event.trajectory_fps
        if self._is_playing:
            self._stop_playback()
            self._start_playback()

    def on_visual_panel_cell_changed(self, event: VisualPanel.CellChanged) -> None:
        view = self.query_one(MoleculeView)
        view.show_cell = event.show_cell
        view.highlighted_display_positions = set()
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
        view = self._query_molecule_view()
        if view is None:
            return
        display = self._display_geometry
        positions: set[tuple[int, int, int]] = set()
        if display is not None:
            for idx in event.atom_indices:
                if 0 <= idx < len(display.molecule.atoms):
                    positions.add(_position_key(display.molecule.atoms[idx].position))
        view.highlighted_display_positions = positions
        view.highlighted_atoms = set(event.atom_indices)
        view._invalidate_cache()

    def _set_current_mo(self, mo_idx: int) -> None:
        """Single entry point for all MO changes."""
        if self.orbital_data is None or self.orbital_data.n_mos == 0:
            return
        mo_idx = max(0, min(mo_idx, self.orbital_data.n_mos - 1))
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
        if self.orbital_data is None or self.orbital_data.n_mos == 0:
            return
        self._mo_switch_task = asyncio.create_task(self._switch_mo_async(self.current_mo))

    async def _switch_mo_async(self, target_mo: int) -> None:
        try:
            if self.orbital_data is None:
                return
            isosurfaces = await asyncio.to_thread(
                _compute_mo_isosurfaces, self.orbital_data, target_mo, self.isovalue
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
    from .qc_inputs import (
        QC_INPUT_AMBIGUOUS_SUFFIXES,
        detect_qc_input_by_extension,
        sniff_qc_input,
    )

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
    if suffix in (".fchk", ".fch"):
        return "fchk"
    if suffix in (".xyz", ".extxyz"):
        return "xyz"
    if suffix == ".cif":
        return "cif"
    if path.name.lower() in ("poscar", "contcar") or suffix in (".vasp", ".poscar"):
        return "poscar"
    if suffix == ".xsf":
        return "xsf"
    if suffix in (".h5", ".hdf5") and path.is_file():
        return "trexio"
    if suffix == ".trexio" and (path.is_file() or path.is_dir()):
        return "trexio"
    qc_by_ext = detect_qc_input_by_extension(path)
    if qc_by_ext is not None:
        return qc_by_ext
    if path.is_dir():
        from .trexio_support import is_readable_trexio_text_directory

        if is_readable_trexio_text_directory(path):
            return "trexio"
        raise ValueError(
            f"{path.resolve()!s} is a directory and is not a readable TREXIO text archive. "
        )

    sniffed_qc: str | None = None
    try:
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
                break
        if suffix in QC_INPUT_AMBIGUOUS_SUFFIXES or suffix == "":
            sniffed_qc = sniff_qc_input(path)
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"Could not read {filepath!s}: {exc}") from exc

    if sniffed_qc is not None:
        return sniffed_qc
    if suffix in (".cube", ".cub"):
        return "cube"
    raise ValueError(
        f"Unsupported file format: {filepath!s}. "
        "Supported formats: .xyz, .extxyz, .cif, POSCAR/CONTCAR/.vasp, .xsf, "
        ".cube, .molden, .fchk, .hess, .zmat, .gbw, .h5/.hdf5/.trexio "
        "(TREXIO), and QC inputs from "
        "Orca, Molcas, Q-Chem, Gaussian, NWChem, Turbomole, Molpro, MRCC, "
        "CFOUR, Psi4, GAMESS, and Jaguar."
    )


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


def _cli_homo_mo_isosurfaces(orbital_data: OrbitalData) -> tuple[list[IsosurfaceMesh], int]:
    """Evaluate the HOMO on a grid and mesh isosurfaces for CLI startup."""
    from .molden import evaluate_mo

    current_mo = orbital_data.homo_idx
    cube_data = evaluate_mo(orbital_data, current_mo)
    return extract_isosurfaces(cube_data), current_mo


def _prepare_trexio_cli_session(
    filepath: str | Path,
) -> tuple[Molecule, OrbitalData | None, list[IsosurfaceMesh], int, str | None]:
    """Load a TREXIO path for the CLI: MO data when present, else geometry only."""
    from .trexio_molden import load_trexio_orbital_data
    from .trexio_support import load_molecule_from_trexio

    orbital_data = load_trexio_orbital_data(filepath)
    toast: str | None = None
    if orbital_data is not None and orbital_data.n_mos > 0:
        missing = []
        if not orbital_data.has_mo_energies:
            missing.append("energies")
        if not orbital_data.has_mo_occupations:
            missing.append("occupations")
        if missing:
            toast = f"TREXIO file missing MO {', '.join(missing)}"
        surfs, current_mo = _cli_homo_mo_isosurfaces(orbital_data)
        return orbital_data.molecule, orbital_data, surfs, current_mo, toast
    mol = load_molecule_from_trexio(filepath)
    return mol, None, [], 0, toast


def run():
    import argparse

    parser = argparse.ArgumentParser(
        prog="moltui",
        description="Terminal-based 3D molecular viewer",
    )
    parser.add_argument(
        "file",
        help=(
            "molecular structure file (XYZ, Cube, Molden, Gaussian .fchk, "
            "ORCA .hess, ORCA .gbw, "
            'or TREXIO .h5/.hdf5/.trexio; optional: pip install "moltui[trexio]")'
        ),
    )
    parsed = parser.parse_args()

    filepath = parsed.file
    try:
        filetype = _detect_filetype(filepath)
    except ValueError as exc:
        import sys

        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    isosurfaces: list[IsosurfaceMesh] = []
    orbital_data = None
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
        trexio_toast: str | None = None
        cube_data_for_app: CubeData | None = None
        if filetype == "cube":
            cube_data = parse_cube_data(filepath)
            molecule = cube_data.molecule
            isosurfaces = extract_isosurfaces(cube_data)
            cube_data_for_app = cube_data
        elif filetype == "molden":
            from .molden import load_molden_data

            orbital_data = load_molden_data(filepath)
            molecule = orbital_data.molecule
            if orbital_data.normal_modes is not None:
                eq_coords = np.array([atom.position.copy() for atom in molecule.atoms])
                vib_modes, vib_freqs = _filter_rigid_body_modes(
                    orbital_data.normal_modes, orbital_data.mode_frequencies
                )
                normal_mode_data = NormalModeData(
                    equilibrium_coords=eq_coords,
                    mode_vectors=vib_modes,
                    frequencies=vib_freqs,
                )
            if orbital_data.n_mos > 0:
                isosurfaces, current_mo = _cli_homo_mo_isosurfaces(orbital_data)
        elif filetype == "fchk":
            from .fchk import load_fchk_data, parse_fchk_atoms, parse_fchk_trajectory

            trajectory = None
            try:
                trajectory = parse_fchk_trajectory(filepath)
            except ValueError as traj_exc:
                if "does not contain" not in str(traj_exc):
                    raise
            else:
                trajectory_data = TrajectoryData(frames=trajectory.frames)

            try:
                orbital_data = load_fchk_data(filepath)
            except ValueError as orbital_exc:
                if "missing required" not in str(orbital_exc):
                    raise
                # Some fchk files contain only geometries/trajectories and no
                # basis/MO blocks. Still open them in geometry mode.
                molecule = (
                    trajectory.molecule if trajectory is not None else parse_fchk_atoms(filepath)
                )
            else:
                molecule = orbital_data.molecule
                if trajectory is not None:
                    # Show the first trajectory frame initially, matching XYZ
                    # trajectory behavior, while retaining MO/normal-mode data
                    # when the fchk also contains it.
                    molecule = trajectory.molecule
                if orbital_data.normal_modes is not None:
                    eq_coords = np.array(
                        [atom.position.copy() for atom in orbital_data.molecule.atoms]
                    )
                    normal_mode_data = NormalModeData(
                        equilibrium_coords=eq_coords,
                        mode_vectors=orbital_data.normal_modes,
                        frequencies=orbital_data.mode_frequencies,
                    )
                if orbital_data.n_mos > 0:
                    isosurfaces, current_mo = _cli_homo_mo_isosurfaces(orbital_data)
        elif filetype == "hess":
            hess_data = parse_orca_hess_data(filepath)
            molecule = hess_data.molecule
            if hess_data.normal_modes is not None:
                eq_coords = np.array([atom.position.copy() for atom in molecule.atoms])
                vib_modes, vib_freqs = _filter_rigid_body_modes(
                    hess_data.normal_modes, hess_data.frequencies
                )
                normal_mode_data = NormalModeData(
                    equilibrium_coords=eq_coords,
                    mode_vectors=vib_modes,
                    frequencies=vib_freqs,
                )
        elif filetype == "xyz":
            traj = parse_xyz_trajectory(filepath)
            molecule = traj.molecule
            trajectory_data = TrajectoryData(frames=traj.frames)
        elif filetype == "trexio":
            molecule, orbital_data, isosurfaces, current_mo, trexio_toast = (
                _prepare_trexio_cli_session(filepath)
            )
        elif filetype == "cif":
            import warnings as _warnings

            from .parsers import CIFParseWarning, parse_cif

            with _warnings.catch_warnings(record=True) as caught:
                _warnings.simplefilter("always", CIFParseWarning)
                molecule = parse_cif(filepath)
            cif_warnings = [
                str(w.message) for w in caught if issubclass(w.category, CIFParseWarning)
            ]
            if cif_warnings:
                trexio_toast = "; ".join(cif_warnings)
        else:
            molecule = load_molecule(filepath)

        app = MoltuiApp(
            molecule=molecule,
            filepath=parsed.file,
            isosurfaces=isosurfaces,
            orbital_data=orbital_data,
            current_mo=current_mo,
            trajectory_data=trajectory_data,
            normal_mode_data=normal_mode_data,
        )
        app._cube_data = cube_data_for_app
        if filetype in ("trexio", "cif"):
            app._startup_toast = trexio_toast
        app.run()
    except ValueError as exc:
        import sys

        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        if gbw_tmpdir is not None:
            import shutil

            shutil.rmtree(gbw_tmpdir, ignore_errors=True)
