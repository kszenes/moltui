from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
from rich.segment import Segment
from rich.style import Style
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.strip import Strip
from textual.widget import Widget
from textual.widgets import DataTable, Footer, Header, TabbedContent

from .elements import Molecule
from .geometry_panel import GeometryPanel
from .image_renderer import render_scene, rotation_matrix
from .isosurface import IsosurfaceMesh, extract_isosurfaces
from .mo_panel import MOPanel
from .parsers import load_molecule, parse_cube_data
from .visual_panel import VisualPanel

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


def _compute_mo_isosurfaces(molden_data, mo_idx: int) -> list[IsosurfaceMesh]:
    from .molden import evaluate_mo

    cube_data = evaluate_mo(molden_data, mo_idx)
    return extract_isosurfaces(cube_data)


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
        Binding("v", "toggle_style", "Style"),
        Binding("i", "toggle_bg", "Bg"),
        Binding("o", "toggle_orbitals", "Orbitals"),
        Binding("escape", "close_panel", "Close panel", show=False),
        Binding("q", "quit", "Quit"),
        Binding("g", "toggle_geometry", "Geom"),
        Binding("m", "toggle_mo_panel", "MOs"),
        Binding("right_square_bracket", "next_mo", "MO]", show=False),
        Binding("left_square_bracket", "prev_mo", "[MO", show=False),
        Binding("e", "export_png", "Export"),
        Binding("number_sign", "toggle_atom_numbers", "#Nums"),
        Binding("n", "panel_next", "Next"),
        Binding("p", "panel_prev", "Prev"),
        Binding("V", "toggle_visual", "Visual"),
    ]

    def __init__(
        self,
        molecule: Molecule,
        filepath: str = "",
        isosurfaces: list[IsosurfaceMesh] | None = None,
        molden_data=None,
        current_mo: int = 0,
    ):
        super().__init__()
        self.molecule = molecule
        self.filepath = filepath
        self._isosurfaces = isosurfaces or []
        self.molden_data = molden_data
        self.current_mo = current_mo
        self._mo_switch_timer = None
        self._mo_pending = False
        self._mo_switch_task: asyncio.Task[None] | None = None
        self.title = self._title_text()

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main-content"):
            yield MoleculeView()
            yield GeometryPanel()
            yield MOPanel()
            yield VisualPanel()
        yield Footer()

    def on_mount(self) -> None:
        view = self.query_one(MoleculeView)
        view.set_molecule(self.molecule, self._isosurfaces)
        panel = self.query_one(GeometryPanel)
        panel.set_molecule(self.molecule)
        if self.molden_data is not None:
            md = self.molden_data
            mo_panel = self.query_one(MOPanel)
            mo_panel.set_mo_data(
                energies=md.mo_energies.tolist(),
                occupations=md.mo_occupations.tolist(),
                symmetries=md.mo_symmetries,
                spins=md.mo_spins,
                current_mo=self.current_mo,
            )
        view.focus()

    def _panel_is_open(self) -> bool:
        return (
            self.query_one(GeometryPanel).has_class("visible")
            or self.query_one(MOPanel).has_class("visible")
            or self.query_one(VisualPanel).has_class("visible")
        )

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        if action in ("toggle_orbitals", "toggle_mo_panel", "next_mo", "prev_mo"):
            if self.molden_data is None and not self._isosurfaces:
                return False
        if action in ("panel_next", "panel_prev"):
            if not self._panel_is_open():
                return False
        return True

    def _title_text(self) -> str:
        parts = [Path(self.filepath).name]
        if self.molden_data is not None:
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
            view.rot_matrix = rotation_matrix(0, -0.1, 0) @ view.rot_matrix
        view._invalidate_cache()

    def action_rotate_right(self) -> None:
        view = self.query_one(MoleculeView)
        if view.pan_mode:
            view.pan_x -= view.camera_distance * 0.05
            view._clamp_pan()
        else:
            view.rot_matrix = rotation_matrix(0, 0.1, 0) @ view.rot_matrix
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
        view.licorice = not view.licorice
        view.bond_radius = 0.15 if view.licorice else 0.08
        vis = self.query_one(VisualPanel)
        if vis.has_class("visible"):
            vis.set_state(
                licorice=view.licorice,
                ambient=view.ambient,
                diffuse=view.diffuse,
                specular=view.specular,
                shininess=view.shininess,
                atom_scale=view.atom_scale,
                bond_radius=view.bond_radius,
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
        return None

    def action_panel_next(self) -> None:
        if self.query_one(VisualPanel).has_class("visible"):
            self.screen.focus_next()
            return
        dt = self._active_panel_table()
        if dt is not None:
            dt.action_cursor_down()

    def action_panel_prev(self) -> None:
        if self.query_one(VisualPanel).has_class("visible"):
            self.screen.focus_previous()
            return
        dt = self._active_panel_table()
        if dt is not None:
            dt.action_cursor_up()

    def action_close_panel(self) -> None:
        geom = self.query_one(GeometryPanel)
        mo = self.query_one(MOPanel)
        vis = self.query_one(VisualPanel)
        if geom.has_class("visible") or mo.has_class("visible") or vis.has_class("visible"):
            self._close_panels()
            view = self.query_one(MoleculeView)
            view._invalidate_cache()
            view.focus()

    def _close_panels(self) -> None:
        """Close all sidebar panels and reset their state."""
        view = self.query_one(MoleculeView)
        geom = self.query_one(GeometryPanel)
        mo = self.query_one(MOPanel)
        vis = self.query_one(VisualPanel)
        if geom.has_class("visible"):
            geom.remove_class("visible")
            view.highlighted_atoms = set()
        if mo.has_class("visible"):
            mo.remove_class("visible")
        if vis.has_class("visible"):
            vis.remove_class("visible")

    def action_toggle_geometry(self) -> None:
        panel = self.query_one(GeometryPanel)
        was_visible = panel.has_class("visible")
        self._close_panels()
        view = self.query_one(MoleculeView)
        if not was_visible:
            panel.add_class("visible")
            view.show_orbitals = False
            for dt in panel.query(DataTable):
                dt.focus()
                panel._emit_current_highlight(dt)
                break
        else:
            view.focus()
        view._invalidate_cache()

    def action_toggle_mo_panel(self) -> None:
        if self.molden_data is None:
            self.notify("No MO data (molden files only)", timeout=2)
            return
        mo_panel = self.query_one(MOPanel)
        was_visible = mo_panel.has_class("visible")
        self._close_panels()
        view = self.query_one(MoleculeView)
        if not was_visible:
            mo_panel.add_class("visible")
            view.show_orbitals = True
            mo_panel.select_mo(self.current_mo, center=True)
            for dt in mo_panel.query(DataTable):
                dt.focus()
                mo_panel.emit_current_highlight(dt)
                break
        else:
            view.focus()
        view._invalidate_cache()

    def action_toggle_visual(self) -> None:
        vis = self.query_one(VisualPanel)
        was_visible = vis.has_class("visible")
        self._close_panels()
        view = self.query_one(MoleculeView)
        if not was_visible:
            vis.set_state(
                licorice=view.licorice,
                ambient=view.ambient,
                diffuse=view.diffuse,
                specular=view.specular,
                shininess=view.shininess,
                atom_scale=view.atom_scale,
                bond_radius=view.bond_radius,
            )
            vis.add_class("visible")
            for child in vis.query("*"):
                if child.can_focus:
                    child.focus()
                    break
        else:
            view.focus()
        view._invalidate_cache()

    def on_visual_panel_style_changed(self, event: VisualPanel.StyleChanged) -> None:
        view = self.query_one(MoleculeView)
        view.licorice = event.licorice
        # Switch bond_radius to a sensible default for the style
        view.bond_radius = 0.15 if event.licorice else 0.08
        vis = self.query_one(VisualPanel)
        if vis.has_class("visible"):
            vis.set_state(
                licorice=view.licorice,
                ambient=view.ambient,
                diffuse=view.diffuse,
                specular=view.specular,
                shininess=view.shininess,
                atom_scale=view.atom_scale,
                bond_radius=view.bond_radius,
            )
        view._invalidate_cache()

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

    def on_geometry_panel_highlight_atoms(self, event: GeometryPanel.HighlightAtoms) -> None:
        view = self.query_one(MoleculeView)
        view.highlighted_atoms = set(event.atom_indices)
        view._invalidate_cache()

    def _set_current_mo(self, mo_idx: int) -> None:
        """Single entry point for all MO changes."""
        if self.molden_data is None:
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
        self._mo_switch_task = asyncio.create_task(self._switch_mo_async(self.current_mo))

    async def _switch_mo_async(self, target_mo: int) -> None:
        try:
            if self.molden_data is None:
                return
            isosurfaces = await asyncio.to_thread(
                _compute_mo_isosurfaces, self.molden_data, target_mo
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
    suffix = Path(filepath).suffix.lower()
    if suffix == ".gbw":
        return "gbw"
    with open(filepath) as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            if "[Molden Format]" in stripped:
                return "molden"
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
    parser.add_argument("file", help="molecular structure file (XYZ, Cube, Molden, or ORCA .gbw)")
    parsed = parser.parse_args()

    filepath = parsed.file
    filetype = _detect_filetype(filepath)
    isosurfaces: list[IsosurfaceMesh] = []
    molden_data = None
    current_mo = 0
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
        if filetype == "cube":
            cube_data = parse_cube_data(filepath)
            molecule = cube_data.molecule
            isosurfaces = extract_isosurfaces(cube_data)
        elif filetype == "molden":
            from .molden import evaluate_mo, load_molden_data

            molden_data = load_molden_data(filepath)
            molecule = molden_data.molecule
            current_mo = molden_data.homo_idx
            cube_data = evaluate_mo(molden_data, current_mo)
            isosurfaces = extract_isosurfaces(cube_data)
        else:
            molecule = load_molecule(filepath)

        app = MoltuiApp(
            molecule=molecule,
            filepath=parsed.file,
            isosurfaces=isosurfaces,
            molden_data=molden_data,
            current_mo=current_mo,
        )
        app.run()
    finally:
        if gbw_tmpdir is not None:
            import shutil

            shutil.rmtree(gbw_tmpdir, ignore_errors=True)
