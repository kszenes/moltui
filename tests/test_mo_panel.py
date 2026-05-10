from __future__ import annotations

from moltui.mo_panel import MOPanel, mo_display_order


def test_mo_panel_sorts_rows_but_displays_source_mo_numbers() -> None:
    panel = MOPanel()
    # Source indices 2 and 3 are intentionally out of display order: source MO 4
    # has larger occupation than source MO 3, so it appears earlier in the panel.
    energies = [-1.0, -0.5, -0.8, -0.9]
    occupations = [2.0, 0.0, 1.5, 2.0]

    panel.set_mo_data(
        energies=energies,
        occupations=occupations,
        symmetries=["A"] * 4,
        spins=["Alpha"] * 4,
        current_mo=3,
    )

    assert [source_idx for source_idx, *_ in panel._mo_data] == [0, 3, 2, 1]
    assert mo_display_order(energies, occupations) == [0, 3, 2, 1]
    assert panel.display_number_for_mo(3) == 4
    assert panel.display_number_for_mo(2) == 3
