from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Label, RadioButton, RadioSet, Static


class _NavRadioSet(RadioSet):
    """RadioSet where tab/shift+tab cycle and activate options."""

    BINDINGS = [
        Binding("tab", "next_and_toggle", "Next", show=False),
        Binding("shift+tab", "prev_and_toggle", "Prev", show=False),
    ]

    def action_next_and_toggle(self) -> None:
        self.action_next_button()
        self.action_toggle_button()

    def action_prev_and_toggle(self) -> None:
        self.action_previous_button()
        self.action_toggle_button()


class Toggle(Static, can_focus=True):
    """Minimal single-line checkbox in the visual-panel style."""

    BINDINGS = [
        Binding("tab", "flip", "Toggle", show=False),
        Binding("shift+tab", "flip", "Toggle", show=False),
        Binding("space", "flip", "Toggle", show=False),
        Binding("enter", "flip", "Toggle", show=False),
    ]

    DEFAULT_CSS = """
    Toggle {
        height: 1;
        width: 1fr;
        padding: 0 1;
    }
    Toggle:focus {
        background: $accent 30%;
        text-style: bold;
    }
    """

    def __init__(self, label: str, value: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.label = label
        self.value = value

    class Changed(Message):
        def __init__(self, toggle: "Toggle", value: bool) -> None:
            super().__init__()
            self.toggle = toggle
            self.value = value

    def render(self) -> str:
        prefix = "▸ " if self.has_focus else "  "
        mark = "✓" if self.value else " "
        return f"{prefix}{self.label}: [{mark}]"

    def action_flip(self) -> None:
        self.value = not self.value
        self.refresh()
        self.post_message(self.Changed(self, self.value))


class Slider(Static, can_focus=True):
    """Minimal horizontal slider. Tab/Shift+Tab adjust value."""

    BINDINGS = [
        Binding("tab", "increase", "Increase", show=False),
        Binding("shift+tab", "decrease", "Decrease", show=False),
    ]

    DEFAULT_CSS = """
    Slider {
        height: 1;
        width: 1fr;
        padding: 0 1;
    }
    Slider:focus {
        background: $accent 30%;
        text-style: bold;
    }
    """

    def __init__(
        self,
        label: str,
        value: float = 0.5,
        min_val: float = 0.0,
        max_val: float = 1.0,
        step: float = 0.05,
        decimals: int = 2,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.label = label
        self.value = value
        self.min_val = min_val
        self.max_val = max_val
        self.step = step
        self.decimals = decimals

    class Changed(Message):
        def __init__(self, slider: "Slider", value: float) -> None:
            super().__init__()
            self.slider = slider
            self.value = value

    def render(self) -> str:
        bar_width = 10
        frac = (self.value - self.min_val) / max(self.max_val - self.min_val, 1e-9)
        filled = int(frac * bar_width)
        bar = "\u2588" * filled + "\u2591" * (bar_width - filled)
        prefix = "\u25b8 " if self.has_focus else "  "
        arrows = " \u25c0\u25b6" if self.has_focus else ""
        return f"{prefix}{self.label}: {self.value:.{self.decimals}f} [{bar}]{arrows}"

    def _adjust(self, delta: float) -> None:
        new = max(self.min_val, min(self.max_val, self.value + delta))
        if new != self.value:
            self.value = new
            self.refresh()
            self.post_message(self.Changed(self, self.value))

    def action_decrease(self) -> None:
        self._adjust(-self.step)

    def action_increase(self) -> None:
        self._adjust(self.step)


class VisualPanel(Widget):
    DEFAULT_CSS = """
    VisualPanel {
        dock: right;
        width: 30;
        display: none;
        border-left: solid $accent;
        padding: 1;
    }
    VisualPanel.visible {
        display: block;
    }
    VisualPanel Label {
        margin-top: 1;
        text-style: bold;
    }
    VisualPanel RadioSet {
        height: auto;
        margin-bottom: 1;
    }
    VisualPanel #visual-help {
        dock: bottom;
        height: auto;
        color: $text-muted;
        margin-top: 1;
    }
    """

    class StyleChanged(Message):
        def __init__(self, licorice: bool, vdw: bool) -> None:
            super().__init__()
            self.licorice = licorice
            self.vdw = vdw

    class LightingChanged(Message):
        def __init__(
            self,
            ambient: float,
            diffuse: float,
            specular: float,
            shininess: float,
        ) -> None:
            super().__init__()
            self.ambient = ambient
            self.diffuse = diffuse
            self.specular = specular
            self.shininess = shininess

    class SizeChanged(Message):
        def __init__(self, atom_scale: float, bond_radius: float) -> None:
            super().__init__()
            self.atom_scale = atom_scale
            self.bond_radius = bond_radius

    class IsovalueChanged(Message):
        def __init__(self, isovalue: float) -> None:
            super().__init__()
            self.isovalue = isovalue

    class VibrationSpeedChanged(Message):
        def __init__(self, phase_step: float) -> None:
            super().__init__()
            self.phase_step = phase_step

    class TrajectorySpeedChanged(Message):
        def __init__(self, trajectory_fps: float) -> None:
            super().__init__()
            self.trajectory_fps = trajectory_fps

    class CellChanged(Message):
        def __init__(self, show_cell: bool) -> None:
            super().__init__()
            self.show_cell = show_cell

    def __init__(self) -> None:
        super().__init__()
        self._licorice = False
        self._vdw = False
        self._has_isosurfaces = False
        self._has_trajectory = False
        self._has_normal_modes = False
        self._has_lattice = False
        self._isovalue_min = 0.001
        self._isovalue_max = 0.10
        self._isovalue_step = 0.005

    def set_state(
        self,
        *,
        licorice: bool,
        vdw: bool = False,
        ambient: float,
        diffuse: float,
        specular: float,
        shininess: float,
        atom_scale: float,
        bond_radius: float,
        isovalue: float = 0.05,
        has_isosurfaces: bool = False,
        has_trajectory: bool = False,
        has_normal_modes: bool = False,
        vibrational_phase_step: float = 0.30,
        trajectory_fps: float = 12.0,
        has_lattice: bool = False,
        show_cell: bool = True,
        isovalue_min: float = 0.001,
        isovalue_max: float = 0.10,
        isovalue_step: float = 0.005,
    ) -> None:
        self._licorice = licorice
        self._vdw = vdw
        self._has_isosurfaces = has_isosurfaces
        self._has_trajectory = has_trajectory
        self._has_normal_modes = has_normal_modes
        self._has_lattice = has_lattice
        self._isovalue_min = isovalue_min
        self._isovalue_max = isovalue_max
        self._isovalue_step = isovalue_step
        if self.is_mounted:
            self._sync_widgets(
                ambient=ambient,
                diffuse=diffuse,
                specular=specular,
                shininess=shininess,
                atom_scale=atom_scale,
                bond_radius=bond_radius,
                isovalue=isovalue,
                vibrational_phase_step=vibrational_phase_step,
                trajectory_fps=trajectory_fps,
                show_cell=show_cell,
            )

    def _sync_widgets(
        self,
        *,
        ambient: float,
        diffuse: float,
        specular: float,
        shininess: float,
        atom_scale: float,
        bond_radius: float,
        isovalue: float = 0.05,
        vibrational_phase_step: float = 0.30,
        trajectory_fps: float = 12.0,
        show_cell: bool = True,
    ) -> None:
        radio_set = self.query_one(_NavRadioSet)
        if self._licorice:
            idx = 1
        elif self._vdw:
            idx = 2
        else:
            idx = 0
        radio_set.query(RadioButton)[idx].value = True
        self.query_one("#slider-atom-scale", Slider).value = atom_scale
        self.query_one("#slider-bond-radius", Slider).value = bond_radius
        self.query_one("#slider-ambient", Slider).value = ambient
        self.query_one("#slider-diffuse", Slider).value = diffuse
        self.query_one("#slider-specular", Slider).value = specular
        self.query_one("#slider-shininess", Slider).value = shininess
        isovalue_slider = self.query_one("#slider-isovalue", Slider)
        isovalue_slider.min_val = self._isovalue_min
        isovalue_slider.max_val = self._isovalue_max
        isovalue_slider.step = self._isovalue_step
        isovalue_slider.value = isovalue
        self.query_one("#slider-vibrational-speed", Slider).value = vibrational_phase_step
        self.query_one("#slider-trajectory-speed", Slider).value = trajectory_fps
        self.query_one("#checkbox-show-cell", Toggle).value = show_cell
        self._update_visibility()
        self.refresh()

    def compose(self) -> ComposeResult:
        yield Label("Periodic", id="label-periodic")
        yield Toggle("Show Box", value=True, id="checkbox-show-cell")
        yield Label("Animation", id="label-animation")
        yield Slider(
            "Vibrational Speed",
            value=0.30,
            min_val=0.05,
            max_val=1.50,
            step=0.05,
            id="slider-vibrational-speed",
        )
        yield Slider(
            "Trajectory FPS",
            value=12.0,
            min_val=1.0,
            max_val=60.0,
            step=1.0,
            decimals=0,
            id="slider-trajectory-speed",
        )
        yield Label("Isosurface", id="label-isovalue")
        yield Slider(
            "Isovalue",
            value=0.05,
            max_val=0.10,
            min_val=0.001,
            step=0.005,
            decimals=3,
            id="slider-isovalue",
        )
        yield Label("Style")
        with _NavRadioSet():
            yield RadioButton("CPK", value=True, id="radio-cpk")
            yield RadioButton("Licorice", id="radio-licorice")
            yield RadioButton("VDW", id="radio-vdw")
        yield Label("Sizes", id="label-sizes")
        yield Slider(
            "Atom scale",
            value=0.35,
            min_val=0.10,
            max_val=1.00,
            id="slider-atom-scale",
        )
        yield Slider(
            "Bond radius",
            value=0.08,
            min_val=0.02,
            max_val=0.30,
            step=0.02,
            id="slider-bond-radius",
        )
        yield Label("Lighting")
        yield Slider(
            "Ambient",
            value=0.35,
            min_val=0.0,
            max_val=1.0,
            id="slider-ambient",
        )
        yield Slider(
            "Diffuse",
            value=0.60,
            min_val=0.0,
            max_val=1.0,
            id="slider-diffuse",
        )
        yield Slider(
            "Specular",
            value=0.40,
            min_val=0.0,
            max_val=1.0,
            id="slider-specular",
        )
        yield Slider(
            "Shininess",
            value=32.0,
            min_val=1.0,
            max_val=128.0,
            step=4.0,
            id="slider-shininess",
        )
        yield Static(
            "n/p nav; (shift-)tab toggle",
            id="visual-help",
        )

    def _update_visibility(self) -> None:
        self.query_one("#slider-atom-scale", Slider).display = not self._licorice
        self.query_one("#slider-bond-radius", Slider).display = not self._vdw
        self.query_one("#label-sizes", Label).display = True
        self.query_one("#label-isovalue", Label).display = self._has_isosurfaces
        self.query_one("#slider-isovalue", Slider).display = self._has_isosurfaces
        self.query_one("#slider-vibrational-speed", Slider).display = self._has_normal_modes
        self.query_one("#slider-trajectory-speed", Slider).display = self._has_trajectory
        self.query_one("#label-animation", Label).display = (
            self._has_normal_modes or self._has_trajectory
        )
        self.query_one("#label-periodic", Label).display = self._has_lattice
        self.query_one("#checkbox-show-cell", Toggle).display = self._has_lattice

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        self._licorice = event.pressed.id == "radio-licorice"
        self._vdw = event.pressed.id == "radio-vdw"
        self._update_visibility()
        self.post_message(self.StyleChanged(self._licorice, self._vdw))

    def on_toggle_changed(self, event: Toggle.Changed) -> None:
        if event.toggle.id == "checkbox-show-cell":
            self.post_message(self.CellChanged(bool(event.value)))

    def on_slider_changed(self, event: Slider.Changed) -> None:
        sid = event.slider.id or ""
        if sid == "slider-isovalue":
            self.post_message(self.IsovalueChanged(event.value))
        elif sid == "slider-vibrational-speed":
            self.post_message(self.VibrationSpeedChanged(event.value))
        elif sid == "slider-trajectory-speed":
            self.post_message(self.TrajectorySpeedChanged(event.value))
        elif sid.startswith("slider-atom") or sid.startswith("slider-bond"):
            self.post_message(
                self.SizeChanged(
                    atom_scale=self.query_one("#slider-atom-scale", Slider).value,
                    bond_radius=self.query_one("#slider-bond-radius", Slider).value,
                )
            )
        else:
            self.post_message(
                self.LightingChanged(
                    ambient=self.query_one("#slider-ambient", Slider).value,
                    diffuse=self.query_one("#slider-diffuse", Slider).value,
                    specular=self.query_one("#slider-specular", Slider).value,
                    shininess=self.query_one("#slider-shininess", Slider).value,
                )
            )
