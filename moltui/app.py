import sys
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


class MoleculeView(Widget):
    """Braille-based 3D molecule renderer."""

    can_focus = True

    def __init__(self) -> None:
        super().__init__()
        self.molecule: Molecule | None = None
        self.isosurfaces: list[IsosurfaceMesh] = []
        self.rot_x = 0.5
        self.rot_y = 0.0
        self.rot_z = 0.0
        self.camera_distance = 4.0
        self.show_bonds = True
        self.show_orbitals = True
        self.dark_bg = True
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.pan_mode = False
        self.highlighted_atoms: set[int] = set()
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
        rot = rotation_matrix(self.rot_x, self.rot_y, self.rot_z)

        mol = self.molecule
        if not self.show_bonds:
            mol = Molecule(atoms=mol.atoms, bonds=[])

        isos = self.isosurfaces if self.show_orbitals else None
        hl = self.highlighted_atoms if self.highlighted_atoms else None
        pixels = render_scene(
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
        )

        bg_arr = np.array(bg, dtype=np.uint8)
        blocks = pixels.reshape(rows, 4, cols, 2, 3)
        is_on = np.any(blocks != bg_arr, axis=4)

        braille_bits = np.where(is_on, _BRAILLE_MAP[None, :, None, :], 0)
        codepoints = 0x2800 + braille_bits.sum(axis=(1, 3)).astype(np.uint32)

        on_count = is_on.sum(axis=(1, 3))
        on_mask = is_on[:, :, :, :, None]
        color_sum = (blocks * on_mask).sum(axis=(1, 3))
        safe_count = np.maximum(on_count, 1)[:, :, None]
        avg_fg = (color_sum / safe_count).astype(np.uint8)

        bg_style = Style(bgcolor=f"rgb({bg[0]},{bg[1]},{bg[2]})")

        strips = []
        for row in range(rows):
            segments = []
            prev_style = None
            run_chars: list[str] = []
            for x in range(cols):
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
        Binding("o", "toggle_orbitals", "Orbitals"),
        Binding("escape", "close_panel", "Close panel", show=False),
        Binding("q", "quit", "Quit"),
        Binding("g", "toggle_geometry", "Geom"),
        Binding("m", "toggle_mo_panel", "MOs"),
        Binding("right_square_bracket", "next_mo", "MO]", show=False),
        Binding("left_square_bracket", "prev_mo", "[MO", show=False),
        Binding("n", "panel_next", "Next"),
        Binding("p", "panel_prev", "Prev"),
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
        self.title = self._title_text()

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main-content"):
            yield MoleculeView()
            yield GeometryPanel()
            yield MOPanel()
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
                homo_idx=md.homo_idx,
                current_mo=self.current_mo,
            )
        view.focus()

    def _panel_is_open(self) -> bool:
        return self.query_one(GeometryPanel).has_class("visible") or self.query_one(
            MOPanel
        ).has_class("visible")

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
            homo_label = ""
            if self.current_mo == md.homo_idx:
                homo_label = " HOMO"
            elif self.current_mo == md.homo_idx + 1:
                homo_label = " LUMO"
            mo_str = f"MO {self.current_mo + 1}/{md.n_mos}"
            parts.append(f"{mo_str} {sym}{homo_label} E={energy:.4f} occ={occ:.1f}")
        return " | ".join(parts)

    def _update_title(self) -> None:
        self.title = self._title_text()

    def action_rotate_up(self) -> None:
        view = self.query_one(MoleculeView)
        if view.pan_mode:
            view.pan_y -= view.camera_distance * 0.05
            view._clamp_pan()
        else:
            view.rot_x -= 0.1
        view._invalidate_cache()

    def action_rotate_down(self) -> None:
        view = self.query_one(MoleculeView)
        if view.pan_mode:
            view.pan_y += view.camera_distance * 0.05
            view._clamp_pan()
        else:
            view.rot_x += 0.1
        view._invalidate_cache()

    def action_rotate_left(self) -> None:
        view = self.query_one(MoleculeView)
        if view.pan_mode:
            view.pan_x += view.camera_distance * 0.05
            view._clamp_pan()
        else:
            view.rot_y -= 0.1
        view._invalidate_cache()

    def action_rotate_right(self) -> None:
        view = self.query_one(MoleculeView)
        if view.pan_mode:
            view.pan_x -= view.camera_distance * 0.05
            view._clamp_pan()
        else:
            view.rot_y += 0.1
        view._invalidate_cache()

    def action_rotate_cw(self) -> None:
        view = self.query_one(MoleculeView)
        view.rot_z += 0.1
        view._invalidate_cache()

    def action_rotate_ccw(self) -> None:
        view = self.query_one(MoleculeView)
        view.rot_z -= 0.1
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
        view.rot_x = 0.5
        view.rot_y = 0.0
        view.rot_z = 0.0
        view.pan_x = 0.0
        view.pan_y = 0.0
        view.pan_mode = False
        view.highlighted_atoms = set()
        if view.molecule:
            view.camera_distance = max(4.0, view.molecule.radius() * 3.0)
        view._invalidate_cache()

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
        dt = self._active_panel_table()
        if dt is not None:
            dt.action_cursor_down()

    def action_panel_prev(self) -> None:
        dt = self._active_panel_table()
        if dt is not None:
            dt.action_cursor_up()

    def action_close_panel(self) -> None:
        geom = self.query_one(GeometryPanel)
        mo = self.query_one(MOPanel)
        if geom.has_class("visible") or mo.has_class("visible"):
            self._close_panels()
            view = self.query_one(MoleculeView)
            view._invalidate_cache()
            view.focus()

    def _close_panels(self) -> None:
        """Close all sidebar panels and reset their state."""
        view = self.query_one(MoleculeView)
        geom = self.query_one(GeometryPanel)
        mo = self.query_one(MOPanel)
        if geom.has_class("visible"):
            geom.remove_class("visible")
            view.highlighted_atoms = set()
        if mo.has_class("visible"):
            mo.remove_class("visible")

    def action_toggle_geometry(self) -> None:
        panel = self.query_one(GeometryPanel)
        was_visible = panel.has_class("visible")
        self._close_panels()
        view = self.query_one(MoleculeView)
        if not was_visible:
            panel.add_class("visible")
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
            mo_panel.select_mo(self.current_mo)
            for dt in mo_panel.query(DataTable):
                dt.focus()
                mo_panel.emit_current_highlight(dt)
                break
        else:
            view.focus()
        view._invalidate_cache()

    def on_mopanel_moselected(self, event: MOPanel.MOSelected) -> None:
        if self.molden_data is None:
            return
        mo_idx = event.mo_index
        if mo_idx == self.current_mo:
            return
        self.current_mo = mo_idx
        self._switch_mo()
        mo_panel = self.query_one(MOPanel)
        mo_panel._current_mo = mo_idx

    def on_geometry_panel_highlight_atoms(self, event: GeometryPanel.HighlightAtoms) -> None:
        view = self.query_one(MoleculeView)
        view.highlighted_atoms = set(event.atom_indices)
        view._invalidate_cache()

    def action_next_mo(self) -> None:
        if self.molden_data is None:
            return
        if self.current_mo < self.molden_data.n_mos - 1:
            self.current_mo += 1
            self._switch_mo()

    def action_prev_mo(self) -> None:
        if self.molden_data is None:
            return
        if self.current_mo > 0:
            self.current_mo -= 1
            self._switch_mo()

    def _switch_mo(self) -> None:
        from .molden import evaluate_mo

        cube_data = evaluate_mo(self.molden_data, self.current_mo)
        self._isosurfaces = extract_isosurfaces(cube_data)
        view = self.query_one(MoleculeView)
        view.isosurfaces = self._isosurfaces
        view._invalidate_cache()
        self._update_title()


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

    app = MoltuiApp(
        molecule=molecule,
        filepath=filepath,
        isosurfaces=isosurfaces,
        molden_data=molden_data,
        current_mo=current_mo,
    )
    app.run()
