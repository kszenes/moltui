"""Kitty graphics protocol rendering for MolTUI.

Uses the Kitty "Unicode placeholder" virtual placement mode (U=1).
The image is transmitted once via a direct APC write; subsequent render_line
calls return Strips containing U+10EEEE placeholder grapheme clusters.  The
terminal substitutes the image pixels for those characters, so Textual's
normal text-rendering path drives display with no race or flicker.
"""
from __future__ import annotations

import base64
import io
import os

import numpy as np


def _tty_write(data: bytes) -> None:
    """Write *data* directly to /dev/tty, bypassing Textual's stdout capture."""
    with open("/dev/tty", "wb", buffering=0) as tty:
        tty.write(data)


# Assumed terminal cell pixel dimensions — tune if fonts differ
_CELL_W = 8
_CELL_H = 16

_KITTY_IMAGE_ID = 1


def detect_kitty_support() -> bool:
    """Return True when the running terminal supports Kitty graphics."""
    term = os.environ.get("TERM", "")
    term_program = os.environ.get("TERM_PROGRAM", "")
    return "kitty" in term or term_program in ("WezTerm", "ghostty", "kitty")


def write_kitty_image_at(
    pixels: np.ndarray,
    screen_x: int,
    screen_y: int,
    cols: int,
    rows: int,
    image_id: int = _KITTY_IMAGE_ID,
) -> None:
    """Encode *pixels* and place them at terminal cell (*screen_x*, *screen_y*).

    Uses ``z=1`` so the image floats above the text layer — Textual's blank
    strips beneath it do not cause flicker.  Saves and restores the cursor so
    Textual's own cursor state is undisturbed.
    """
    from PIL import Image  # always available via scikit-image

    h, w, _ = pixels.shape
    img = Image.fromarray(pixels, "RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False, compress_level=1)
    b64 = base64.standard_b64encode(buf.getvalue())

    # a=T: transmit+display at cursor; z=1: above text; c,r: cell footprint; q=2: quiet
    header = f"a=T,f=100,I={image_id},s={w},v={h},c={cols},r={rows},z=1,q=2".encode()
    _tty_write(
        f"\x1b7\x1b[{screen_y + 1};{screen_x + 1}H".encode()
        + b"\x1b_G" + header + b";" + b64 + b"\x1b\\"
        + b"\x1b8"
    )


def delete_kitty_image(image_id: int = _KITTY_IMAGE_ID) -> None:
    """Ask the terminal to free the stored image."""
    _tty_write(f"\x1b_Ga=d,I={image_id},q=2\x1b\\".encode())


def pixel_dims(cols: int, rows: int) -> tuple[int, int]:
    """Return render pixel width/height for a *cols*×*rows* terminal area."""
    return cols * _CELL_W, rows * _CELL_H
