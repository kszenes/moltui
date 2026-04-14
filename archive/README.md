# Archive

## kitty_viewer.py

Kitty graphics protocol rendering backend, removed from `moltui/app.py` on 2026-04-14.

Used the [Kitty graphics protocol](https://sw.kovidgoyal.net/kitty/graphics-protocol/) to send rendered pixel buffers directly to the terminal via temp file + zlib compression. This gave true-pixel rendering but only worked in Kitty-compatible terminals.

The project now uses braille-based text rendering exclusively, which works in any terminal (including over SSH).

The file is self-contained: it includes the `KittyViewer` class and all helper functions (`_kitty_send`, `_kitty_delete`, `_get_terminal_pixel_size`, `_get_temp_path`).
