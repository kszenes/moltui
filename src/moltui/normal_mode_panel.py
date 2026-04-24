from __future__ import annotations

from textual.message import Message

from .selection_table_panel import SelectionTablePanel


class NormalModePanel(SelectionTablePanel):
    DEFAULT_CSS = """
    NormalModePanel {
        dock: right;
        width: auto;
        display: none;
        border-left: solid $accent;
    }
    NormalModePanel.visible {
        display: block;
    }
    NormalModePanel DataTable {
        width: auto;
        height: 1fr;
    }
    """

    class ModeSelected(Message):
        def __init__(self, mode_index: int) -> None:
            super().__init__()
            self.mode_index = mode_index

    def __init__(self) -> None:
        super().__init__(table_id="normal-mode-table")
        self._mode_data: list[tuple[int, float | None, str | None]] = []
        self._current_mode: int = 0

    def set_mode_data(
        self,
        mode_count: int,
        frequencies: list[float] | None,
        current_mode: int,
        symmetries: list[str] | None = None,
    ) -> None:
        self._current_mode = current_mode
        self._mode_data = []
        for i in range(mode_count):
            freq = frequencies[i] if frequencies is not None and i < len(frequencies) else None
            sym = symmetries[i] if symmetries is not None and i < len(symmetries) else None
            self._mode_data.append((i, freq, sym))
        if self.is_mounted:
            self._populate_table()

    def _has_rows(self) -> bool:
        return bool(self._mode_data)

    def _populate_table(self) -> None:
        def _populate() -> None:
            table = self._table()
            table.clear(columns=True)
            has_irreps = any(sym is not None for _, _, sym in self._mode_data)
            if has_irreps:
                table.add_columns("Mode", "Irrep", "Frequency (cm^-1)")
            else:
                table.add_columns("Mode", "Frequency (cm^-1)")

            current_row = 0
            for idx, (mode_idx, freq, irrep) in enumerate(self._mode_data):
                freq_text = f"{freq:>10.2f}" if freq is not None else "-"
                if has_irreps:
                    table.add_row(str(mode_idx + 1), irrep or "-", freq_text, key=str(mode_idx))
                else:
                    table.add_row(str(mode_idx + 1), freq_text, key=str(mode_idx))
                if mode_idx == self._current_mode:
                    current_row = idx

            if table.row_count > 0:
                self._set_cursor_row(current_row, center=True)

        self._with_populating(_populate)

    def select_mode(self, mode_idx: int, *, center: bool = False) -> None:
        self._current_mode = mode_idx
        self.select_row_key(mode_idx, center=center)

    def _selection_message(self, row_key: int) -> Message:
        return self.ModeSelected(row_key)
