from __future__ import annotations

from textual.message import Message

from .selection_table_panel import SelectionTablePanel


class MOPanel(SelectionTablePanel):
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
        super().__init__(table_id="mo-table")
        self._mo_data: list[tuple[int, str, float, float, str]] = []
        self._current_mo: int = 0
        self._has_energies: bool = True
        self._has_occupations: bool = True

    def set_mo_data(
        self,
        energies: list[float],
        occupations: list[float],
        symmetries: list[str],
        spins: list[str],
        current_mo: int,
        has_energies: bool = True,
        has_occupations: bool = True,
    ) -> None:
        self._current_mo = current_mo
        self._has_energies = has_energies
        self._has_occupations = has_occupations
        self._mo_data = []
        for i in range(len(energies)):
            sym = symmetries[i] if i < len(symmetries) else ""
            spin = spins[i] if i < len(spins) else ""
            self._mo_data.append((i, sym, energies[i], occupations[i], spin))
        # Sort by occupancy descending, then energy ascending within each group
        self._mo_data.sort(key=lambda x: (-x[3], x[2]))
        if self.is_mounted:
            self._populate_table()

    def _has_rows(self) -> bool:
        return bool(self._mo_data)

    def _populate_table(self) -> None:
        def _populate() -> None:
            table = self._table()
            table.clear(columns=True)

            syms = {s for _, s, _, _, _ in self._mo_data}
            show_sym = len(syms) > 1
            spins = {s for _, _, _, _, s in self._mo_data}
            show_spin = len(spins) > 1

            columns = ["MO"]
            if show_sym:
                columns.append("Irrep")
            if self._has_energies:
                columns.append("Energy")
            if self._has_occupations:
                columns.append("Occ")
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
                if self._has_energies:
                    row.append(f"{energy:>10.5f}")
                if self._has_occupations:
                    row.append(f"{occ:.5f}")
                table.add_row(*row, key=str(mo_idx))
                if mo_idx == self._current_mo:
                    current_row = idx

            if table.row_count > 0:
                self._set_cursor_row(current_row, center=True)

        self._with_populating(_populate)

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
        self.select_row_key(mo_idx, center=center)

    def _selection_message(self, row_key: int) -> Message:
        return self.MOSelected(row_key)
