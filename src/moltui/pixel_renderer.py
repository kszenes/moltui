"""Kitty graphics protocol rendering for MolTUI.

The image is encoded as PNG, transmitted via a chunked APC write
(a=T, z=1), and placed at the widget's screen coordinates using cursor
save/restore. z=1 floats the image above Textual's text layer so blank
cell strips underneath don't cause flicker. _rebuild_kitty stores the
pixel buffer and schedules _paint_kitty via call_after_refresh, ensuring
the transmission happens after Textual has flushed its own frame.
"""
from __future__ import annotations

import base64
import io
import os
import sys

import numpy as np

_CHUNK_SIZE = 4096

# Assumed terminal cell pixel dimensions; updated by query_cell_px().
# Used only for render resolution — correctness doesn't depend on accuracy.
_cell_px: tuple[int, int] = (8, 16)

_KITTY_IMAGE_ID = 1


def _tty_write(data: bytes) -> None:
    """Write *data* directly to /dev/tty, bypassing Textual's stdout capture."""
    with open("/dev/tty", "wb", buffering=0) as tty:
        tty.write(data)


def detect_kitty_support() -> bool:
    """Return True when the running terminal supports Kitty graphics."""
    if sys.platform == "win32":
        return False
    term = os.environ.get("TERM", "")
    term_program = os.environ.get("TERM_PROGRAM", "")
    return "kitty" in term or term_program in ("WezTerm", "ghostty", "kitty")


def query_cell_px() -> tuple[int, int]:
    """Query the terminal for cell pixel dimensions via CSI 16t.

    Returns (cell_w, cell_h) on success, or (8, 16) as fallback.
    Must be called before the Textual app takes over the terminal.
    Updates the module-level _cell_px used by pixel_dims().
    """
    global _cell_px
    if sys.platform == "win32":
        return _cell_px
    try:
        import re
        import select
        import termios
        import tty

        fd = os.open("/dev/tty", os.O_RDWR | os.O_NOCTTY)
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            os.write(fd, b"\x1b[16t")
            resp = b""
            deadline = 0.3
            while deadline > 0:
                r, _, _ = select.select([fd], [], [], min(0.05, deadline))
                if not r:
                    break
                chunk = os.read(fd, 256)
                if not chunk:
                    break
                resp += chunk
                deadline -= 0.05
                if re.search(rb"\x1b\[6;\d+;\d+t", resp):
                    # Drain any remaining bytes so they don't pollute Textual's stdin.
                    while select.select([fd], [], [], 0.05)[0]:
                        os.read(fd, 256)
                    break
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
            os.close(fd)
        m = re.search(rb"\x1b\[6;(\d+);(\d+)t", resp)
        if m:
            cell_h, cell_w = int(m.group(1)), int(m.group(2))
            # Sanity-check: realistic cell sizes are 4–64 px wide, 8–128 px tall.
            if 4 <= cell_w <= 64 and 8 <= cell_h <= 128:
                _cell_px = (cell_w, cell_h)
    except Exception:
        pass
    return _cell_px


def write_kitty_image_at(
    pixels: np.ndarray,
    screen_x: int,
    screen_y: int,
    cols: int,
    rows: int,
    image_id: int = _KITTY_IMAGE_ID,
) -> None:
    """Encode *pixels* and place them at terminal cell (*screen_x*, *screen_y*).

    Uses z=1 so the image floats above the text layer. Payload is chunked
    at 4096 bytes using m=1/m=0 flags to stay within terminal APC limits.
    Saves and restores the cursor so Textual's own cursor state is undisturbed.
    """
    from PIL import Image

    h, w, _ = pixels.shape
    img = Image.fromarray(pixels, "RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False, compress_level=1)
    b64 = base64.standard_b64encode(buf.getvalue())

    chunks = [b64[i : i + _CHUNK_SIZE] for i in range(0, len(b64), _CHUNK_SIZE)]
    frames: list[bytes] = []
    for idx, chunk in enumerate(chunks):
        more = 0 if idx == len(chunks) - 1 else 1
        if idx == 0:
            header = (
                f"a=T,f=100,I={image_id},s={w},v={h},c={cols},r={rows},z=1,m={more},q=2"
            ).encode()
        else:
            header = f"m={more},q=2".encode()
        frames.append(b"\x1b_G" + header + b";" + chunk + b"\x1b\\")

    _tty_write(
        f"\x1b7\x1b[{screen_y + 1};{screen_x + 1}H".encode()
        + b"".join(frames)
        + b"\x1b8"
    )


def delete_kitty_image(image_id: int = _KITTY_IMAGE_ID) -> None:
    """Ask the terminal to free the stored image."""
    if sys.platform == "win32":
        return
    _tty_write(f"\x1b_Ga=d,I={image_id},q=2\x1b\\".encode())


def pixel_dims(cols: int, rows: int) -> tuple[int, int]:
    """Return render pixel width/height for a *cols*×*rows* terminal area."""
    cw, ch = _cell_px
    return cols * cw, rows * ch
