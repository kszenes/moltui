from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .elements import Element, get_element, get_element_by_number


def parse_fortran_float(token: str) -> float:
    """Parse floats that may use Fortran D/d exponent notation."""
    return float(token.replace("D", "E").replace("d", "e"))


def resolve_element_token(token: str) -> Element:
    """Resolve an element symbol or atomic number token to an Element."""
    try:
        return get_element_by_number(int(token))
    except ValueError:
        return get_element(token)


def parse_float_vec3(
    tokens: list[str] | tuple[str, ...],
    *,
    parser: Callable[[str], float] = float,
) -> np.ndarray:
    if len(tokens) < 3:
        raise ValueError("expected at least three float tokens")
    return np.array([parser(tokens[0]), parser(tokens[1]), parser(tokens[2])], dtype=np.float64)


def read_scalar_grid_values(
    lines: list[str],
    start_idx: int,
    n_points: tuple[int, int, int],
    *,
    parser: Callable[[str], float] = float,
    stop_prefixes: tuple[str, ...] = (),
    order: str = "C",
) -> tuple[np.ndarray, int]:
    """Read a flat scalar grid from text lines and reshape it to n_points."""
    n_values = n_points[0] * n_points[1] * n_points[2]
    values: list[float] = []
    idx = start_idx
    upper_stop_prefixes = tuple(prefix.upper() for prefix in stop_prefixes)
    while idx < len(lines) and len(values) < n_values:
        stripped = lines[idx].strip()
        if stripped:
            upper = stripped.upper()
            if upper_stop_prefixes and upper.startswith(upper_stop_prefixes):
                break
            values.extend(parser(tok) for tok in stripped.split())
        idx += 1
    if len(values) < n_values:
        raise ValueError("scalar grid ended before all values were read")
    return np.array(values[:n_values], dtype=np.float64).reshape(n_points, order=order), idx
