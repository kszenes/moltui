## Source

The `.fchk` files in this directory are taken verbatim from the [iodata test data](https://github.com/theochem/iodata/tree/main/iodata/test/data). The paired `.molden` files are produced by running each fchk through `iodata.dump_one`.

Selection covers the parser's main schema dimensions:

- `h2o_sto3g.fchk` — RHF, s/p only (smoke test)
- `ch3_hf_sto3g.fchk` — UHF (alpha + beta MO blocks)
- `water_ccpvdz_pure_hf_g03.fchk` — RHF with pure-spherical d shells
- `o2_cc_pvtz_cart.fchk` — RHF with Cartesian d and f shells
