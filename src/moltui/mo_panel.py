from __future__ import annotations

from textual.app import ComposeResult
from textual.message import Message
from textual.widget import Widget
from textual.widgets import DataTable


class MOPanel(Widget):
    DEFAULT_CSS = """
    MOPanel {
        dock: right;
        width: auto;
        display: none;
        border-left: solid $accent;
    }
    MOPanel.visible {
        display: block;
    }
    MOPanel DataTable {
        width: auto;
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
        spins: list[str],
        current_mo: int,
    ) -> None:
        self._current_mo = current_mo
        self._mo_data = []
        for i in range(len(energies)):
            sym = symmetries[i] if i < len(symmetries) else ""
            spin = spins[i] if i < len(spins) else ""
            self._mo_data.append((i, sym, energies[i], occupations[i], spin))
        # Sort by occupancy descending, then energy ascending within each group
        self._mo_data.sort(key=lambda x: (-x[3], x[2]))
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

        syms = {s for _, s, _, _, _ in self._mo_data}
        show_sym = len(syms) > 1
        spins = {s for _, _, _, _, s in self._mo_data}
        show_spin = len(spins) > 1

        columns = ["MO"]
        if show_sym:
            columns.append("Sym")
        columns.extend(["Energy", "Occ"])
        table.add_columns(*columns)

        spin_symbol = {"Alpha": "\u03b1", "Beta": "\u03b2"}

        max_mo_num = max((mo_idx + 1 for mo_idx, *_ in self._mo_data), default=1)
        mo_width = len(str(max_mo_num))
        if show_spin:
            mo_width += 2  # space + spin symbol

        current_row = 0
        for idx, (mo_idx, sym, energy, occ, spin) in enumerate(self._mo_data):
            mo_label = str(mo_idx + 1)
            if show_spin:
                mo_label += f" {spin_symbol.get(spin, spin)}"
            mo_label = mo_label.rjust(mo_width)
            row = [mo_label]
            if show_sym:
                row.append(sym)
            row.append(f"{energy:>10.5f}")
            row.append(f"{occ:.5f}")
            table.add_row(*row, key=str(mo_idx))
            if mo_idx == self._current_mo:
                current_row = idx

        if table.row_count > 0:
            table.move_cursor(row=current_row, scroll=False)
            # Center after layout is computed; size can be 0 during initial populate.
            self.call_after_refresh(self._center_row, table, current_row)

        self._populating = False

    def _center_row(self, table: DataTable, row: int) -> None:
        if table.row_count == 0:
            return
        viewport_rows = max(1, table.size.height - 1)  # account for header row
        center_target = max(0, row - viewport_rows // 2)
        table.scroll_to(y=center_target, animate=False, immediate=True)

    def adjacent_mo(self, mo_idx: int, delta: int) -> int | None:
        """Return the MO index of the next/prev entry in table order."""
        for row, (mi, *_) in enumerate(self._mo_data):
            if mi == mo_idx:
                target = row + delta
                if 0 <= target < len(self._mo_data):
                    return self._mo_data[target][0]
                return None
        return None

    def select_mo(self, mo_idx: int, *, center: bool = False) -> None:
        """Move the cursor to the given MO index."""
        self._current_mo = mo_idx
        table = self.query_one("#mo-table", DataTable)
        for row, (mi, *_) in enumerate(self._mo_data):
            if mi == mo_idx:
                self._populating = True
                table.move_cursor(row=row, scroll=not center)
                self._populating = False
                if center:
                    self.call_after_refresh(self._center_row, table, row)
                return

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
