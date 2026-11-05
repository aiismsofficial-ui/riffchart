"""Project root, so the CLI works from any working directory, and config loading with local overrides."""
from __future__ import annotations

from pathlib import Path

import yaml

PROJECT = Path(__file__).resolve().parents[2]


def project_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else PROJECT / p


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | Path) -> dict:
    """Read a YAML config, then <name>.local.yaml next to it if present (this PC's folders and other
    personal settings; never committed), merged over it key by key (lists are replaced)."""
    p = project_path(path)
    cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    local = p.with_name(p.stem + ".local" + p.suffix)
    if local.exists():
        cfg = _merge(cfg, yaml.safe_load(local.read_text(encoding="utf-8")) or {})
    return cfg
