from types import SimpleNamespace

from moltui.app import _compute_export_isosurfaces, _export_render_kwargs


def test_export_render_kwargs_use_view_lighting_and_geometry_settings():
    view = SimpleNamespace(
        pan_x=1.25,
        pan_y=-0.75,
        licorice=True,
        vdw=False,
        ambient=0.46,
        diffuse=0.58,
        specular=0.21,
        shininess=44.0,
        atom_scale=0.42,
        bond_radius=0.11,
    )

    kwargs = _export_render_kwargs(view)

    assert kwargs["ssaa"] == 2
    assert kwargs["pan"] == (1.25, -0.75)
    assert kwargs["licorice"] is True
    assert kwargs["vdw"] is False
    assert kwargs["ambient"] == 0.46
    assert kwargs["diffuse"] == 0.58
    assert kwargs["specular"] == 0.21
    assert kwargs["shininess"] == 44.0
    assert kwargs["atom_scale"] == 0.42
    assert kwargs["bond_radius"] == 0.11


def test_compute_export_isosurfaces_returns_none_when_orbitals_hidden():
    out = _compute_export_isosurfaces(
        show_orbitals=False,
        current_isosurfaces=[],
        cube_data=None,
        molden_data=None,
        current_mo=0,
        isovalue=0.05,
    )
    assert out is None


def test_compute_export_isosurfaces_uses_hq_mo_mesh(monkeypatch):
    import moltui.app as app_mod

    called = {}

    def fake_compute_mo_isosurfaces(molden_data, mo_idx, isovalue, grid_shape):
        called["args"] = (molden_data, mo_idx, isovalue, grid_shape)
        return ["hq-mo"]

    monkeypatch.setattr(app_mod, "_compute_mo_isosurfaces", fake_compute_mo_isosurfaces)
    md = SimpleNamespace(n_mos=12)
    out = _compute_export_isosurfaces(
        show_orbitals=True,
        current_isosurfaces=["preview"],
        cube_data=None,
        molden_data=md,
        current_mo=4,
        isovalue=0.07,
    )
    assert out == ["hq-mo"]
    assert called["args"] == (md, 4, 0.07, (96, 96, 96))


def test_compute_export_isosurfaces_uses_hq_cube_upsample(monkeypatch):
    import moltui.app as app_mod

    called = {}

    def fake_extract(cube_data, isovalue=0.05, step=1, upsample=1):
        called["args"] = (cube_data, isovalue, step, upsample)
        return ["hq-cube"]

    monkeypatch.setattr(app_mod, "extract_isosurfaces", fake_extract)
    cube_data = object()
    out = _compute_export_isosurfaces(
        show_orbitals=True,
        current_isosurfaces=["preview"],
        cube_data=cube_data,
        molden_data=None,
        current_mo=0,
        isovalue=0.04,
    )
    assert out == ["hq-cube"]
    assert called["args"] == (cube_data, 0.04, 1, 2)


def test_compute_export_isosurfaces_falls_back_to_current_when_no_sources():
    current = ["preview"]
    out = _compute_export_isosurfaces(
        show_orbitals=True,
        current_isosurfaces=current,
        cube_data=None,
        molden_data=SimpleNamespace(n_mos=0),
        current_mo=0,
        isovalue=0.05,
    )
    assert out == current
