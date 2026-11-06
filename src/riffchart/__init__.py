"""RiffChart: AI 5-fret guitar charts for Clone Hero and YARG."""
from __future__ import annotations


def _version() -> str:
    """From pyproject.toml (the app runs from its own folder, so this is always the installed version)."""
    from pathlib import Path
    try:
        import tomllib
        p = Path(__file__).resolve().parents[2] / "pyproject.toml"
        return tomllib.loads(p.read_text(encoding="utf-8"))["project"]["version"]
    except Exception:
        try:
            from importlib.metadata import version
            return version("riffchart")
        except Exception:
            return "unknown"


__version__ = _version()
