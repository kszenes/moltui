from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.message import Message
from textual.widget import Widget
from textual.widgets import DataTable, TabbedContent, TabPane

from .elements import Molecule


def _parse_row_key(value: str) -> tuple[int, ...]:
    """Parse a geometry-panel row key into atom indices."""
    atom_part = value.split("#", 1)[0]
    return tuple(int(x) for x in atom_part.split("-"))


class GeometryPanel(Widget):
    BINDINGS = [
        Binding("tab", "next_tab", "Tab", show=True),
        Binding("shift+tab", "prev_tab", "Prev tab", show=False),
        Binding("s", "toggle_sort", "Sort", show=True),
    ]
    DEFAULT_CSS = """
    GeometryPanel {
        dock: right;
        width: 45;
        display: none;
        border-left: solid $accent;
    }
    GeometryPanel.visible {
        display: block;
    }
    GeometryPanel DataTable {
        height: 1fr;
    }
    """

    class HighlightAtoms(Message):
        def __init__(self, atom_indices: tuple[int, ...]) -> None:
            super().__init__()
            self.atom_indices = atom_indices

    def __init__(self) -> None:
        super().__init__()
        self._molecule: Molecule | None = None
        self._parent_indices: list[int] | None = None
        self._populating = False
        self._sort_ascending: dict[str, bool] = {
            "tab-bonds": False,
            "tab-angles": False,
            "tab-dihedrals": False,
        }

    def set_molecule(self, molecule: Molecule, parent_indices: list[int] | None = None) -> None:
        selected_row_keys: dict[str, str | None] | None = None
        if self.is_mounted and self._molecule is not None:
            tab_ids = ("tab-bonds", "tab-angles", "tab-dihedrals")
            selected_row_keys = {
                tab_id: self._current_row_key_value(self._table_for_tab(tab_id))
                for tab_id in tab_ids
            }
        self._molecule = molecule
        self._parent_indices = parent_indices
        if self.is_mounted:
            self._populate_tables(selected_row_keys)

    def refresh_measurements(self) -> None:
        """Recompute displayed geometry values while preserving table selection."""
        if self._molecule is None or not self.is_mounted:
            return
        tab_ids = ("tab-bonds", "tab-angles", "tab-dihedrals")
        selected_row_keys = {
            tab_id: self._current_row_key_value(self._table_for_tab(tab_id)) for tab_id in tab_ids
        }
        self._populate_tables(selected_row_keys)
        tabs = self.query_one(TabbedContent)
        self._emit_current_highlight(self._table_for_tab(tabs.active))

    def on_mount(self) -> None:
        if self._molecule is not None:
            self._populate_tables()

    def action_next_tab(self) -> None:
        tabs = self.query_one(TabbedContent)
        tab_ids = ["tab-bonds", "tab-angles", "tab-dihedrals"]
        current = tabs.active
        idx = tab_ids.index(current) if current in tab_ids else 0
        tabs.active = tab_ids[(idx + 1) % len(tab_ids)]

    def action_prev_tab(self) -> None:
        tabs = self.query_one(TabbedContent)
        tab_ids = ["tab-bonds", "tab-angles", "tab-dihedrals"]
        current = tabs.active
        idx = tab_ids.index(current) if current in tab_ids else 0
        tabs.active = tab_ids[(idx - 1) % len(tab_ids)]

    def action_toggle_sort(self) -> None:
        if self._molecule is None:
            return
        tabs = self.query_one(TabbedContent)
        active = tabs.active
        selected_row_key = self._current_row_key_value(self._table_for_tab(active))
        asc = self._sort_ascending.get(active, False)
        self._sort_ascending[active] = not asc
        self._populate_tables({active: selected_row_key} if selected_row_key is not None else None)
        table = self._table_for_tab(active)
        table.focus()
        self._emit_current_highlight(table)

    def compose(self) -> ComposeResult:
        with TabbedContent("Bonds", "Angles", "Dihedrals"):
            with TabPane("Bonds", id="tab-bonds"):
                yield DataTable(id="bonds-table", cursor_type="row")
            with TabPane("Angles", id="tab-angles"):
                yield DataTable(id="angles-table", cursor_type="row")
            with TabPane("Dihedrals", id="tab-dihedrals"):
                yield DataTable(id="dihedrals-table", cursor_type="row")

    def _atom_label(self, idx: int) -> str:
        if self._molecule is None:
            return str(idx + 1)
        width = len(str(len(self._molecule.atoms)))
        if self._parent_indices is not None and 0 <= idx < len(self._parent_indices):
            label_idx = self._parent_indices[idx]
        else:
            label_idx = idx
        return f"{label_idx + 1:>{width}}:{self._molecule.atoms[idx].element.symbol}"

    def _table_for_tab(self, tab_id: str) -> DataTable:
        table_id = {
            "tab-bonds": "#bonds-table",
            "tab-angles": "#angles-table",
            "tab-dihedrals": "#dihedrals-table",
        }.get(tab_id, "#bonds-table")
        return self.query_one(table_id, DataTable)

    def _current_row_key_value(self, dt: DataTable) -> str | None:
        if dt.row_count == 0:
            return None
        row_keys = list(dt.rows.keys())
        if not row_keys:
            return None
        row = max(0, min(dt.cursor_row, len(row_keys) - 1))
        return row_keys[row].value

    def _restore_cursor(self, dt: DataTable, row_key: str | None) -> None:
        if row_key is None or dt.row_count == 0:
            return
        row_keys = list(dt.rows.keys())
        for row, key in enumerate(row_keys):
            if key.value == row_key:
                dt.move_cursor(row=row, scroll=False)
                return

    def _populate_tables(self, selected_row_keys: dict[str, str | None] | None = None) -> None:
        if self._molecule is None:
            return
        self._populating = True
        selected_row_keys = selected_row_keys or {}

        # Bonds
        bonds = self._molecule.get_bond_lengths()
        table = self.query_one("#bonds-table", DataTable)
        table.clear(columns=True)
        table.add_columns("Atom 1", "Atom 2", "Length (\u00c5)")
        if self._sort_ascending["tab-bonds"]:
            bonds.sort(key=lambda x: x[2])
        for row_idx, (i, j, dist) in enumerate(bonds):
            table.add_row(
                self._atom_label(i),
                self._atom_label(j),
                f"{dist:.4f}",
                key=f"{i}-{j}#{row_idx}",
            )
        self._restore_cursor(table, selected_row_keys.get("tab-bonds"))

        # Angles
        angles = self._molecule.get_angles()
        if self._sort_ascending["tab-angles"]:
            angles.sort(key=lambda x: x[3])
        table = self.query_one("#angles-table", DataTable)
        table.clear(columns=True)
        table.add_columns("Atom 1", "Vertex", "Atom 3", "Angle (\u00b0)")
        for row_idx, (i, j, k, angle) in enumerate(angles):
            table.add_row(
                self._atom_label(i),
                self._atom_label(j),
                self._atom_label(k),
                f"{angle:.3f}",
                key=f"{i}-{j}-{k}#{row_idx}",
            )
        self._restore_cursor(table, selected_row_keys.get("tab-angles"))

        # Dihedrals
        dihedrals = self._molecule.get_dihedrals()
        if self._sort_ascending["tab-dihedrals"]:
            dihedrals.sort(key=lambda x: x[4])
        table = self.query_one("#dihedrals-table", DataTable)
        table.clear(columns=True)
        table.add_columns("Atom 1", "Atom 2", "Atom 3", "Atom 4", "Angle (\u00b0)")
        for row_idx, (i, j, k, l, angle) in enumerate(dihedrals):
            table.add_row(
                self._atom_label(i),
                self._atom_label(j),
                self._atom_label(k),
                self._atom_label(l),
                f"{angle:.3f}",
                key=f"{i}-{j}-{k}-{l}#{row_idx}",
            )
        self._restore_cursor(table, selected_row_keys.get("tab-dihedrals"))

        self._populating = False

    def _emit_current_highlight(self, dt: DataTable) -> None:
        """Emit highlight for the currently selected row in the given table."""
        if not self.has_class("visible"):
            return
        if dt.row_count == 0:
            self.post_message(self.HighlightAtoms(()))
            return
        rk = list(dt.rows.keys())[dt.cursor_row]
        if rk.value is not None:
            self.post_message(self.HighlightAtoms(_parse_row_key(rk.value)))

    def on_tabbed_content_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        """Focus the DataTable in the newly active tab and highlight its row."""
        for dt in event.pane.query(DataTable):
            dt.focus()
            self._emit_current_highlight(dt)
            break

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if self._populating or not self.has_class("visible"):
            return
        # During table repopulation (e.g. after sort toggle), Textual can emit
        # highlight events for non-active tables. Only the active tab should
        # control molecule highlighting.
        tabs = self.query_one(TabbedContent)
        active_table = self._table_for_tab(tabs.active)
        if event.data_table is not active_table:
            return
        if event.row_key is None or event.row_key.value is None:
            return
        self.post_message(self.HighlightAtoms(_parse_row_key(event.row_key.value)))
