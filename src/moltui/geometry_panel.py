from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.message import Message
from textual.widget import Widget
from textual.widgets import DataTable, TabbedContent, TabPane

from .elements import Molecule


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
        self._populating = False
        self._sort_ascending: dict[str, bool] = {
            "tab-bonds": False,
            "tab-angles": False,
            "tab-dihedrals": False,
        }

    def set_molecule(self, molecule: Molecule) -> None:
        self._molecule = molecule
        if self.is_mounted:
            self._populate_tables()

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
        asc = self._sort_ascending.get(active, False)
        self._sort_ascending[active] = not asc
        self._populate_tables()

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
        return f"{idx + 1}:{self._molecule.atoms[idx].element.symbol}"

    def _populate_tables(self) -> None:
        if self._molecule is None:
            return
        self._populating = True

        # Bonds
        bonds = self._molecule.get_bond_lengths()
        if self._sort_ascending["tab-bonds"]:
            bonds.sort(key=lambda x: x[2])
        table = self.query_one("#bonds-table", DataTable)
        table.clear(columns=True)
        table.add_columns("Atom 1", "Atom 2", "Length (\u00c5)")
        for i, j, dist in bonds:
            table.add_row(
                self._atom_label(i),
                self._atom_label(j),
                f"{dist:.4f}",
                key=f"{i}-{j}",
            )

        # Angles
        angles = self._molecule.get_angles()
        if self._sort_ascending["tab-angles"]:
            angles.sort(key=lambda x: x[3])
        table = self.query_one("#angles-table", DataTable)
        table.clear(columns=True)
        table.add_columns("Atom 1", "Vertex", "Atom 3", "Angle (\u00b0)")
        for i, j, k, angle in angles:
            table.add_row(
                self._atom_label(i),
                self._atom_label(j),
                self._atom_label(k),
                f"{angle:.3f}",
                key=f"{i}-{j}-{k}",
            )

        # Dihedrals
        dihedrals = self._molecule.get_dihedrals()
        if self._sort_ascending["tab-dihedrals"]:
            dihedrals.sort(key=lambda x: x[4])
        table = self.query_one("#dihedrals-table", DataTable)
        table.clear(columns=True)
        table.add_columns("Atom 1", "Atom 2", "Atom 3", "Atom 4", "Angle (\u00b0)")
        for i, j, k, l, angle in dihedrals:
            table.add_row(
                self._atom_label(i),
                self._atom_label(j),
                self._atom_label(k),
                self._atom_label(l),
                f"{angle:.3f}",
                key=f"{i}-{j}-{k}-{l}",
            )

        self._populating = False

    def _emit_current_highlight(self, dt: DataTable) -> None:
        """Emit highlight for the currently selected row in the given table."""
        if not self.has_class("visible") or dt.row_count == 0:
            return
        rk = list(dt.rows.keys())[dt.cursor_row]
        if rk.value is not None:
            indices = tuple(int(x) for x in rk.value.split("-"))
            self.post_message(self.HighlightAtoms(indices))

    def on_tabbed_content_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        """Focus the DataTable in the newly active tab and highlight its row."""
        for dt in event.pane.query(DataTable):
            dt.focus()
            self._emit_current_highlight(dt)
            break

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if self._populating or not self.has_class("visible"):
            return
        if event.row_key is None or event.row_key.value is None:
            return
        indices = tuple(int(x) for x in event.row_key.value.split("-"))
        self.post_message(self.HighlightAtoms(indices))
