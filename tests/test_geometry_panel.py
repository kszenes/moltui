#!/usr/bin/env python3
"""Tests for geometry_panel.py highlight routing behavior."""

from __future__ import annotations

from types import SimpleNamespace

from textual.widgets import TabbedContent

from moltui.geometry_panel import GeometryPanel


def test_row_highlight_ignores_non_active_table(monkeypatch) -> None:
    """Ignore row-highlight events from hidden/inactive geometry tables."""
    panel = GeometryPanel()
    panel._populating = False

    active_table = object()
    inactive_table = object()
    tabs = SimpleNamespace(active="tab-bonds")
    posted: list[GeometryPanel.HighlightAtoms] = []

    monkeypatch.setattr(panel, "has_class", lambda class_name: class_name == "visible")
    monkeypatch.setattr(panel, "_table_for_tab", lambda _tab_id: active_table)
    monkeypatch.setattr(
        panel,
        "query_one",
        lambda selector, *_args, **kwargs: tabs if selector is TabbedContent else None,
    )
    monkeypatch.setattr(panel, "post_message", lambda message: posted.append(message))

    # Simulate an angle-row highlight firing while Bonds tab is active.
    event = SimpleNamespace(
        data_table=inactive_table,
        row_key=SimpleNamespace(value="1-0-2"),
    )
    panel.on_data_table_row_highlighted(event)

    assert posted == []


def test_row_highlight_emits_for_active_table(monkeypatch) -> None:
    panel = GeometryPanel()
    panel._populating = False

    active_table = object()
    tabs = SimpleNamespace(active="tab-bonds")
    posted: list[GeometryPanel.HighlightAtoms] = []

    monkeypatch.setattr(panel, "has_class", lambda class_name: class_name == "visible")
    monkeypatch.setattr(panel, "_table_for_tab", lambda _tab_id: active_table)
    monkeypatch.setattr(
        panel,
        "query_one",
        lambda selector, *_args, **kwargs: tabs if selector is TabbedContent else None,
    )
    monkeypatch.setattr(panel, "post_message", lambda message: posted.append(message))

    event = SimpleNamespace(
        data_table=active_table,
        row_key=SimpleNamespace(value="0-2"),
    )
    panel.on_data_table_row_highlighted(event)

    assert len(posted) == 1
    assert posted[0].atom_indices == (0, 2)
