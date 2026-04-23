#!/usr/bin/env python3
"""Extract verbatim QC inputs from cclib-data output logs and parse them.

Some QC programs echo the full input file at the top of their output.
This script extracts those embedded inputs from the cclib regression
data, writes them to `data/extracted_inputs/<program>/`, and runs each
through `moltui.parsers.load_molecule` to confirm the parser handles
real-world (not just hand-crafted) inputs.

Handles GAMESS, Q-Chem, Orca, Psi4, Molcas, and Molpro — programs with
unambiguous input-echo markers in their outputs.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from regression_baseline import check_baseline  # noqa: E402

from moltui.parsers import load_molecule  # noqa: E402

REGRESSION = REPO_ROOT / "data" / "regression"
EXTRACTED = REPO_ROOT / "data" / "extracted_inputs"


def extract_gamess(text: str) -> str | None:
    """Strip ` INPUT CARD>` prefix from GAMESS output to recover input."""
    lines = []
    for raw in text.splitlines():
        m = re.match(r"^\s*INPUT CARD>(.*)$", raw)
        if m:
            lines.append(m.group(1).rstrip())
    return "\n".join(lines) + "\n" if lines else None


def _is_qchem_divider(line: str) -> bool:
    """The Q-Chem `User input:` divider is a long all-dash line (~60 chars).

    Distinct from `--` fragment separators that may appear inside
    `$molecule` blocks for EDA / counterpoise jobs.
    """
    s = line.strip()
    return len(s) >= 10 and set(s) == {"-"}


def extract_qchem(text: str) -> str | None:
    """Pull verbatim input between `User input:` and the trailing dash divider."""
    lines = text.splitlines()
    for i, l in enumerate(lines):
        if l.strip() == "User input:":
            j = i + 1
            if j < len(lines) and _is_qchem_divider(lines[j]):
                j += 1
            start = j
            while j < len(lines) and not _is_qchem_divider(lines[j]):
                j += 1
            block = "\n".join(lines[start:j]).strip()
            return block + "\n" if block else None
    return None


def extract_orca(text: str) -> str | None:
    """Strip Orca's `|<lineno>> ` echo prefix; stop at `****END OF INPUT****`."""
    lines: list[str] = []
    saw_any = False
    for raw in text.splitlines():
        m = re.match(r"^\|\s*\d+>\s?(.*)$", raw)
        if m:
            lines.append(m.group(1).rstrip())
            saw_any = True
            continue
        if saw_any and "END OF INPUT" in raw.upper():
            break
    return "\n".join(lines) + "\n" if lines else None


def _is_long_dash_divider(line: str, min_len: int = 10) -> bool:
    s = line.strip()
    return len(s) >= min_len and set(s) == {"-"}


def extract_psi4(text: str) -> str | None:
    """Pull verbatim input between `==> Input File <==` and the trailing dash divider."""
    lines = text.splitlines()
    for i, l in enumerate(lines):
        if "==> Input File <==" in l:
            j = i + 1
            while j < len(lines) and (not lines[j].strip() or _is_long_dash_divider(lines[j])):
                j += 1
                if j > i + 1 and _is_long_dash_divider(lines[j - 1]):
                    break
            start = j
            while j < len(lines) and not _is_long_dash_divider(lines[j]):
                j += 1
            block = "\n".join(lines[start:j]).strip()
            return block + "\n" if block else None
    return None


def extract_molcas(text: str) -> str | None:
    """Pull input between `++ ---  Input file  ---` and the closing `--` block."""
    lines = text.splitlines()
    for i, l in enumerate(lines):
        s = l.strip()
        if s.startswith("++") and "Input file" in s:
            j = i + 1
            start = j
            while j < len(lines):
                t = lines[j].strip()
                if t.startswith("--") and ("---" in t or t == "--"):
                    break
                j += 1
            block = "\n".join(lines[start:j]).strip()
            return block + "\n" if block else None
    return None


def extract_molpro(text: str) -> str | None:
    """Pull Molpro input from start of file (after banner lines) to the
    `Variables initialized` / `Checking input` / `PROGRAM SYSTEM MOLPRO` line.
    """
    lines = text.splitlines()
    start: int | None = None
    for i, raw in enumerate(lines):
        s = raw.strip()
        if s.startswith("***"):
            start = i
            break
    if start is None:
        return None
    j = start
    while j < len(lines):
        s = lines[j].strip()
        if (
            "Variables initialized" in s
            or s.startswith("Checking input")
            or "PROGRAM SYSTEM MOLPRO" in s
        ):
            break
        j += 1
    # Strip leading single-space indent that Molpro uses for echoed lines.
    block_lines = [l[1:] if l.startswith(" ") else l for l in lines[start:j]]
    block = "\n".join(block_lines).rstrip()
    return block + "\n" if block else None


EXTRACTORS = {
    "GAMESS": extract_gamess,
    "QChem": extract_qchem,
    "ORCA": extract_orca,
    "Psi4": extract_psi4,
    "Molcas": extract_molcas,
    "Molpro": extract_molpro,
}

# Per-program extension to write extracted inputs as.
EXTENSIONS = {
    "GAMESS": ".inp",
    "QChem": ".in",
    "ORCA": ".inp",
    "Psi4": ".dat",
    "Molcas": ".input",
    "Molpro": ".com",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        help="JSON baseline. Passes when ok counts do not drop and failures do not increase.",
    )
    args = parser.parse_args()

    if not REGRESSION.is_dir():
        print(f"!! {REGRESSION} not found; clone cclib-data first", file=sys.stderr)
        return 1
    EXTRACTED.mkdir(parents=True, exist_ok=True)

    extracted: dict[str, list[Path]] = defaultdict(list)
    for prog, extractor in EXTRACTORS.items():
        prog_root = REGRESSION / prog
        if not prog_root.is_dir():
            continue
        for out in sorted(prog_root.rglob("*.out")) + sorted(prog_root.rglob("*.log")):
            try:
                text = out.read_text(errors="replace")
            except OSError:
                continue
            payload = extractor(text)
            if not payload:
                continue
            ext = EXTENSIONS.get(prog, ".inp")
            target_dir = EXTRACTED / prog / out.relative_to(prog_root).parent
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / (out.stem + ext)
            target.write_text(payload)
            extracted[prog].append(target)

    if not extracted:
        print("!! no inputs extracted")
        return 1

    print("\nExtracted inputs:")
    for prog, paths in extracted.items():
        print(f"  {prog:<10}  {len(paths):>3} files → {EXTRACTED / prog}")

    ok: dict[str, int] = defaultdict(int)
    failures: dict[str, list[tuple[Path, str]]] = defaultdict(list)
    for prog, paths in extracted.items():
        for p in paths:
            try:
                mol = load_molecule(p)
                if len(mol.atoms) > 0:
                    ok[prog] += 1
                else:
                    failures[prog].append((p, "0 atoms parsed"))
            except Exception as exc:
                msg = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
                failures[prog].append((p.relative_to(EXTRACTED), msg))

    total = sum(len(v) for v in extracted.values())
    parsed = sum(ok.values())
    print(f"\n=== Parsing result: {parsed}/{total} ===\n")
    stats = {
        prog: (ok.get(prog, 0), len(failures.get(prog, [])))
        for prog in sorted(set(extracted) | set(failures))
    }
    for prog, (good, bad) in stats.items():
        print(f"  {prog:<10}  {good:>3} ok / {bad:>3} fail")

    if failures:
        print("\n--- Failures ---")
        for prog, items in sorted(failures.items()):
            # Cluster by error message; show one exemplar per cluster.
            by_msg: dict[str, list[Path]] = defaultdict(list)
            for path, msg in items:
                by_msg[msg].append(path)
            for msg, paths in by_msg.items():
                print(f"\n  [{prog}] ({len(paths)} files) {msg}")
                for p in paths[:3]:
                    print(f"    {p}")
                if len(paths) > 3:
                    print(f"    ... and {len(paths) - 3} more")

    if args.baseline:
        failure_paths = {
            str(path.relative_to(EXTRACTED) if path.is_absolute() else path)
            for items in failures.values()
            for path, _msg in items
        }
        return (
            0
            if check_baseline(
                baseline_path=args.baseline,
                total=total,
                parsed=parsed,
                failed=total - parsed,
                stats=stats,
                stats_key="by_program",
                failure_paths=failure_paths,
            )
            else 1
        )
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
