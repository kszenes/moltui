#!/usr/bin/env python3
"""Run `load_molecule` against every QC-input-shaped file in cclib-data.

Discovers candidate files under data/regression/ matching the QC-input
extensions/filenames moltui claims to support, attempts parsing each, and
prints a pass/fail report grouped by detected format.

This is a smoke test for parser robustness against real-world inputs from
upstream QC packages — not a unit test (no oracle for expected geometry).
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from regression_baseline import check_baseline  # noqa: E402

from moltui.app import _detect_filetype  # noqa: E402
from moltui.parsers import load_molecule  # noqa: E402

REGRESSION = REPO_ROOT / "data" / "regression"

GLOB_PATTERNS = [
    # QC inputs
    "*.inp",
    "*.in",
    "*.com",
    "*.gjf",
    "*.nw",
    "*.nwi",
    "*.qcin",
    "*.input",
    "*.minp",
    "*.gms",
    "*.dat",
    "coord",
    "MINP",
    "ZMAT",
    # Native moltui formats
    "*.xyz",
    "*.extxyz",
    "*.cif",
    "*.molden",
    "*.fchk",
    "*.fch",
    "*.hess",
    "*.cube",
    "*.cub",
    "*.zmat",
    "*.zmatrix",
]

PROGRAM_DIRS = [
    "ORCA",
    "QChem",
    "Gaussian",
    "NWChem",
    "Turbomole",
    "Molcas",
    "Molpro",
    "Psi4",
    "Jaguar",
    "GAMESS",
    "FChk",
    "io",
    "method",
]


# Filenames known not to be QC geometry inputs even though their extension
# matches our globs (e.g. Turbomole helper-script transcripts and spectrum
# data files that happen to live next to a `coord` file).
NON_INPUT_FILENAMES = {
    "define.input",
    "freeh.input",
    "ir-spek.dat",
    "raman-spek.dat",
}


def discover() -> list[Path]:
    files: set[Path] = set()
    for prog in PROGRAM_DIRS:
        prog_root = REGRESSION / prog
        if not prog_root.is_dir():
            continue
        for pattern in GLOB_PATTERNS:
            files.update(prog_root.rglob(pattern))
    return sorted(p for p in files if p.name not in NON_INPUT_FILENAMES)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        help="JSON baseline. Passes when ok counts do not drop and failures do not increase.",
    )
    args = parser.parse_args()

    if not REGRESSION.is_dir():
        print(f"!! cclib-data not found at {REGRESSION}", file=sys.stderr)
        return 1

    candidates = discover()
    if not candidates:
        print("!! no candidate files found", file=sys.stderr)
        return 1

    by_kind: dict[str, list[tuple[Path, str | None]]] = defaultdict(list)
    parse_errors: dict[str, list[tuple[Path, str]]] = defaultdict(list)
    detect_errors: list[tuple[Path, str]] = []

    for path in candidates:
        rel = path.relative_to(REGRESSION)
        try:
            kind = _detect_filetype(str(path))
        except Exception as exc:
            detect_errors.append((rel, str(exc).splitlines()[0]))
            continue
        try:
            mol = load_molecule(path)
            by_kind[kind].append((rel, f"{len(mol.atoms)} atoms"))
        except Exception as exc:
            msg = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
            parse_errors[kind].append((rel, msg))

    total = len(candidates)
    parsed = sum(len(v) for v in by_kind.values())
    failed = sum(len(v) for v in parse_errors.values()) + len(detect_errors)
    print(f"\n=== Summary: {parsed}/{total} parsed; {failed} failed ===\n")
    stats = {
        kind: (len(by_kind.get(kind, [])), len(parse_errors.get(kind, [])))
        for kind in sorted(set(by_kind) | set(parse_errors))
    }
    for kind, (ok, bad) in stats.items():
        print(f"  {kind:<18}  {ok:>3} ok / {bad:>3} fail")

    if parse_errors:
        print("\n--- Parse failures ---")
        for kind, items in sorted(parse_errors.items()):
            print(f"\n  [{kind}]")
            for rel, msg in items:
                print(f"    {rel}: {msg}")
    if detect_errors:
        print("\n--- Detection failures ---")
        for rel, msg in detect_errors:
            print(f"    {rel}: {msg}")

    if args.baseline:
        failure_paths = {str(rel) for rel, _msg in detect_errors}
        for items in parse_errors.values():
            failure_paths.update(str(rel) for rel, _msg in items)
        return (
            0
            if check_baseline(
                baseline_path=args.baseline,
                total=total,
                parsed=parsed,
                failed=failed,
                stats=stats,
                stats_key="by_kind",
                failure_paths=failure_paths,
            )
            else 1
        )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
