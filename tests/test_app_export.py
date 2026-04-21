from types import SimpleNamespace

from moltui.app import _export_render_kwargs


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
