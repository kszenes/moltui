# Changelog

## [v0.4.0] - 2026-04-26

- Add support for TREXIO files (geometry and MOs) (closed [#10](https://github.com/kszenes/moltui/issues/10))
- Improve AO construction caching for faster rendering
- Add integration tests for parsing

## [v0.3.7] - 2026-04-23

- Add animation speed slider (contributed by [@Dpgibney](https://github.com/Dpgibney) in [#11](https://github.com/kszenes/moltui/pull/11))

## [v0.3.6] - 2026-04-21

- Fix Molcas Molden format issue (contributed by [@fra-mazz](https://github.com/fra-mazza) in [#8](https://github.com/kszenes/moltui/pull/8))

## [v0.3.5] - 2026-04-21

- Revert to fast rendering for export

## [v0.3.4] - 2026-04-21

- Fix accidental removed cube visualization
- Fix small rendering gaps in bonds

## [v0.3.3] - 2026-04-21

- Fix broken navigation in visual panel

## [v0.3.2] - 2026-04-21

- Add more vim keys for datatable navigation
- Fix debounce for repeated key presses
- Fix buggy navigation with `n` and `p`

## [v0.3.1] - 2026-04-21

- Recompute bonds on each trajectory frame

## [v0.3.0] - 2026-04-20

- Add multi-XYZ trajectory cycling with geometry panel updates (suggested by [@jonmarks12](https://github.com/jonmarks12) in [#1](https://github.com/kszenes/moltui/pull/1))
- Add normal mode visualizer for Molden files
- Add normal mode support for ORCA `.hess` files
- Update keybindings layout
- Fix geometry panel highlight bug when sorting
- Fix molecule not highlighted when quantity is absent
- Fix descriptive error message for Molden normal-modes

## [v0.2.3] - 2026-04-20

- Add proper support for g shells in Molden files

## [v0.2.2] - 2026-04-20

- Fix support for both spherical and Cartesian MOLDEN formats

## [v0.2.1] - 2026-04-18

- Fix highlighted element lost when sorting tables
- Switch to `hatch-vcs` for automatic versioning

## [v0.2.0] - 2026-04-18

- Add isovalue slider and adjustable isovalue setting
- Add VDW (van der Waals) rendering style
- Add PNG export functionality
- Add visual settings dialog
- Add atom number toggle
- Add visualization side panel
- Add support for ORCA `.gbw` files
- Add support for Z-matrix (`.zmat`) format
- Fix rotation direction
