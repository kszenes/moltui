from __future__ import annotations

from textual.app import ComposeResult
from textual.message import Message
from textual.widget import Widget
from textual.widgets import DataTable


class MOPanel(Widget):
    DEFAULT_CSS = """
    MOPanel {
        width: 45;
        display: none;
        border-left: solid $accent;
    }
    MOPanel.visible {
        display: block;
    }
    MOPanel DataTable {
        height: 1fr;
    }
    """

    class MOSelected(Message):
        def __init__(self, mo_index: int) -> None:
            super().__init__()
            self.mo_index = mo_index

    def __init__(self) -> None:
        super().__init__()
        self._mo_data: list[tuple[int, str, float, float, str]] = []
        self._populating = False
        self._current_mo: int = 0

    def set_mo_data(
        self,
        energies: list[float],
        occupations: list[float],
        symmetries: list[str],
        homo_idx: int,
        current_mo: int,
    ) -> None:
        self._current_mo = current_mo
        self._mo_data = []
        for i in range(len(energies)):
            sym = symmetries[i] if i < len(symmetries) else ""
            label = ""
            if i == homo_idx:
                label = "HOMO"
            elif i == homo_idx + 1:
                label = "LUMO"
            self._mo_data.append((i, sym, energies[i], occupations[i], label))
        # Always sort by energy
        self._mo_data.sort(key=lambda x: x[2])
        if self.is_mounted:
            self._populate_table()

    def on_mount(self) -> None:
        if self._mo_data:
            self._populate_table()

    def compose(self) -> ComposeResult:
        yield DataTable(id="mo-table", cursor_type="row")

    def _populate_table(self) -> None:
        self._populating = True
        table = self.query_one("#mo-table", DataTable)
        table.clear(columns=True)
        table.add_columns("MO", "Sym", "Energy", "Occ", "")

        current_row = 0
        for idx, (mo_idx, sym, energy, occ, label) in enumerate(self._mo_data):
            table.add_row(
                str(mo_idx + 1),
                sym,
                f"{energy:.4f}",
                f"{occ:.1f}",
                label,
                key=str(mo_idx),
            )
            if mo_idx == self._current_mo:
                current_row = idx

        if table.row_count > 0:
            table.move_cursor(row=current_row)

        self._populating = False

    def emit_current_highlight(self, dt: DataTable) -> None:
        if not self.has_class("visible") or dt.row_count == 0:
            return
        rk = list(dt.rows.keys())[dt.cursor_row]
        if rk.value is not None:
            self.post_message(self.MOSelected(int(rk.value)))

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if self._populating or not self.has_class("visible"):
            return
        if event.row_key is None or event.row_key.value is None:
            return
        self.post_message(self.MOSelected(int(event.row_key.value)))
