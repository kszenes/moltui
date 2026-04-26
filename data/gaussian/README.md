## Source

The `.fchk` files in this directory are taken verbatim from the [iodata test data](https://github.com/theochem/iodata/tree/main/iodata/test/data). The paired `.molden` files are produced by running each fchk through `iodata.dump_one` (orbital files) or `cclib.io.ccwrite` (the freq file — `iodata`'s molden writer drops `[FREQ]`/`[FR-NORM-COORD]`).

Selection covers the parser's main schema dimensions:

- `h2o_sto3g.fchk` — RHF, s/p only (smoke test)
- `ch3_hf_sto3g.fchk` — UHF (alpha + beta MO blocks)
- `water_ccpvdz_pure_hf_g03.fchk` — RHF with pure-spherical d shells
- `o2_cc_pvtz_cart.fchk` — RHF with Cartesian d and f shells
- `peroxide_tsopt.fchk` — H₂O₂ TS optimisation with `Vib-Modes`/`Vib-E2`/`Cartesian Force Constants` (cross-validates the Hessian-diagonalisation path against Gaussian's own analysis)
- `peroxide_irc.fchk` — H₂O₂ along the IRC, Hessian only (no precomputed modes — exercises the Hessian fallback in `parse_fchk`)
