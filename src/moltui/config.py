"""User configuration for MolTUI.

Reads ~/.config/moltui/config.toml on startup. Missing keys fall back to
built-in defaults; out-of-range values are silently clamped.

Example config:

    [rendering]
    ambient = 0.50
    diffuse = 0.60
    specular = 0.40
    shininess = 32.0
    atom_scale = 0.35
    bond_radius = 0.08
    hd = false
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class MoltuiConfig:
    ambient: float = 0.50
    diffuse: float = 0.60
    specular: float = 0.40
    shininess: float = 32.0
    atom_scale: float = 0.35
    bond_radius: float = 0.08
    hd: bool = False


def _clamp(value: object, lo: float, hi: float, default: float) -> float:
    try:
        return max(lo, min(hi, float(value)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def load_config() -> MoltuiConfig:
    """Load ~/.config/moltui/config.toml, returning defaults if absent or unreadable."""
    path = Path.home() / ".config" / "moltui" / "config.toml"
    if not path.exists():
        return MoltuiConfig()

    try:
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore[no-redef]
    except ImportError:
        return MoltuiConfig()

    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except Exception:
        return MoltuiConfig()

    cfg = MoltuiConfig()
    r = data.get("rendering", {})
    if not isinstance(r, dict):
        return cfg

    cfg.ambient = _clamp(r.get("ambient", cfg.ambient), 0.0, 1.0, cfg.ambient)
    cfg.diffuse = _clamp(r.get("diffuse", cfg.diffuse), 0.0, 1.0, cfg.diffuse)
    cfg.specular = _clamp(r.get("specular", cfg.specular), 0.0, 1.0, cfg.specular)
    cfg.shininess = _clamp(r.get("shininess", cfg.shininess), 1.0, 200.0, cfg.shininess)
    cfg.atom_scale = _clamp(r.get("atom_scale", cfg.atom_scale), 0.1, 2.0, cfg.atom_scale)
    cfg.bond_radius = _clamp(r.get("bond_radius", cfg.bond_radius), 0.01, 1.0, cfg.bond_radius)

    hd = r.get("hd", cfg.hd)
    if isinstance(hd, bool):
        cfg.hd = hd

    return cfg
