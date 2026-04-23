#!/usr/bin/env python3
"""Shared baseline checking for cclib-data regression scripts."""

from __future__ import annotations

import json
from pathlib import Path


def check_baseline(
    *,
    baseline_path: Path,
    total: int,
    parsed: int,
    failed: int,
    stats: dict[str, tuple[int, int]],
    stats_key: str,
    failure_paths: set[str],
) -> bool:
    baseline = json.loads(baseline_path.read_text())
    failures: list[str] = []

    if total < baseline["total"]:
        failures.append(f"candidate count regressed: {total} < {baseline['total']}")
    if parsed < baseline["parsed"]:
        failures.append(f"parsed count regressed: {parsed} < {baseline['parsed']}")
    if failed > baseline["failed"]:
        failures.append(f"failure count regressed: {failed} > {baseline['failed']}")

    for group, expected in baseline[stats_key].items():
        ok, bad = stats.get(group, (0, 0))
        if ok < expected["ok"]:
            failures.append(f"{group} ok count regressed: {ok} < {expected['ok']}")
        if bad > expected["fail"]:
            failures.append(f"{group} failure count regressed: {bad} > {expected['fail']}")

    allowed_failures = set(baseline.get("allowed_failures", []))
    unexpected_failures = sorted(failure_paths - allowed_failures)
    if unexpected_failures:
        failures.append("unexpected failing files:")
        failures.extend(f"  {path}" for path in unexpected_failures[:20])
        if len(unexpected_failures) > 20:
            failures.append(f"  ... and {len(unexpected_failures) - 20} more")

    if failures:
        print("\n--- Baseline regressions ---")
        for item in failures:
            print(f"  {item}")
        return False
    print(f"\nBaseline check passed: {baseline_path}")
    return True
