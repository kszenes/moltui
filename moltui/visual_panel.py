from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Label, RadioButton, RadioSet, Static


class _NavRadioSet(RadioSet):
    """RadioSet with j/k/tab navigation."""

    BINDINGS = [
        *RadioSet.BINDINGS,
        Binding("j", "next_button", show=False),
        Binding("k", "previous_button", show=False),
        Binding("tab", "jump_to_lighting", show=False),
    ]

    def action_jump_to_lighting(self) -> None:
        self.screen.query_one(VisualPanel).query_one(Slider).focus()


class Slider(Static, can_focus=True):
    """Minimal horizontal slider using keyboard left/right."""

    BINDINGS = [
        Binding("left,h", "decrease", "Decrease", show=False),
        Binding("right,l", "increase", "Increase", show=False),
        Binding("up,k", "focus_prev", "Prev", show=False),
        Binding("down,j", "focus_next", "Next", show=False),
        Binding("tab", "jump_to_style", "Style", show=False),
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
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.label = label
        self.value = value
        self.min_val = min_val
        self.max_val = max_val
        self.step = step

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
        return f"{prefix}{self.label}: {self.value:.2f} [{bar}]{arrows}"

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

    def action_focus_prev(self) -> None:
        self.screen.focus_previous()

    def action_focus_next(self) -> None:
        self.screen.focus_next()

    def action_jump_to_style(self) -> None:
        self.screen.query_one(_NavRadioSet).focus()


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
    """

    class StyleChanged(Message):
        def __init__(self, licorice: bool) -> None:
            super().__init__()
            self.licorice = licorice

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

    def __init__(self) -> None:
        super().__init__()
        self._licorice = False

    def set_state(
        self,
        *,
        licorice: bool,
        ambient: float,
        diffuse: float,
        specular: float,
        shininess: float,
    ) -> None:
        self._licorice = licorice
        if self.is_mounted:
            self._sync_widgets(
                ambient=ambient,
                diffuse=diffuse,
                specular=specular,
                shininess=shininess,
            )

    def _sync_widgets(
        self,
        *,
        ambient: float,
        diffuse: float,
        specular: float,
        shininess: float,
    ) -> None:
        radio_set = self.query_one(_NavRadioSet)
        idx = 1 if self._licorice else 0
        radio_set.query(RadioButton)[idx].value = True
        self.query_one("#slider-ambient", Slider).value = ambient
        self.query_one("#slider-diffuse", Slider).value = diffuse
        self.query_one("#slider-specular", Slider).value = specular
        self.query_one("#slider-shininess", Slider).value = shininess
        self.refresh()

    def compose(self) -> ComposeResult:
        yield Label("Style")
        with _NavRadioSet():
            yield RadioButton("CPK (ball & stick)", value=True, id="radio-cpk")
            yield RadioButton("Licorice", id="radio-licorice")
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

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        self._licorice = event.pressed.id == "radio-licorice"
        self.post_message(self.StyleChanged(self._licorice))

    def on_slider_changed(self, event: Slider.Changed) -> None:
        ambient = self.query_one("#slider-ambient", Slider).value
        diffuse = self.query_one("#slider-diffuse", Slider).value
        specular = self.query_one("#slider-specular", Slider).value
        shininess = self.query_one("#slider-shininess", Slider).value
        self.post_message(
            self.LightingChanged(
                ambient=ambient,
                diffuse=diffuse,
                specular=specular,
                shininess=shininess,
            )
        )
