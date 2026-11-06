"""First-run setup: where finished songs go, which folders are never touched, and Clone Hero's song options.

Choices are written to <config>.local.yaml next to the app's config (configs/ui.local.yaml: this PC's settings,
never committed), so they survive updates. Only the keys set here are rewritten; other lines and comments stay.
"""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

import yaml

# Clone Hero settings.ini [custom]: 1 = use the song folder's own video / background / highway
CH_OPTIONS = {"song_videos": "Song videos", "song_backgrounds": "Song backgrounds", "song_highway": "Song highways"}
SETUP_KEYS = ("export_dir", "clone_hero_settings", "protected", "setup_done")


def local_path(config_path: str | Path) -> Path:
    p = Path(config_path)
    return p.with_name(p.stem + ".local" + p.suffix)


def read_local(config_path) -> dict:
    p = local_path(config_path)
    try:
        d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        return d if isinstance(d, dict) else {}
    except (OSError, yaml.YAMLError):
        return {}


def write_local(config_path, updates: dict) -> Path:
    """Set top-level keys in the .local.yaml file, keeping every other line (and comment) as it is."""
    p = local_path(config_path)
    lines = p.read_text(encoding="utf-8").splitlines() if p.exists() else [
        "# This PC's settings (written by RiffChart's setup; never committed). Merged over the main config."]
    out, skip = [], False
    for line in lines:
        m = re.match(r"([A-Za-z_][\w-]*)\s*:", line)
        if m:                                              # a top-level key starts here
            skip = m.group(1) in updates
        elif line[:1] not in (" ", "\t", "-") and line.strip():
            skip = False                                   # top-level comment or anything else
        if not skip:
            out.append(line)
    while out and not out[-1].strip():
        out.pop()
    for k, v in updates.items():
        out += yaml.safe_dump({k: v}, default_flow_style=False, allow_unicode=True, sort_keys=False, width=100000).rstrip().splitlines()
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text("\n".join(out) + "\n", encoding="utf-8")
    tmp.replace(p)
    yaml.safe_load(p.read_text(encoding="utf-8"))          # still valid
    return p


def default_settings() -> Path:
    """Where Clone Hero keeps settings.ini (Windows: Documents\\Clone Hero)."""
    docs = Path.home() / "Documents"
    for c in (docs / "Clone Hero" / "settings.ini", Path(os.environ.get("USERPROFILE", "~")) / "Documents" / "Clone Hero" / "settings.ini",
              Path.home() / ".clonehero" / "settings.ini", Path.home() / ".config" / "unity3d" / "srylain Inc_" / "Clone Hero" / "settings.ini"):
        c = c.expanduser()
        if c.exists():
            return c
    return docs / "Clone Hero" / "settings.ini"


def read_options(settings: Path) -> dict[str, int | None]:
    """song_videos / song_backgrounds / song_highway from [custom] (None: not set, Clone Hero's default is on)."""
    out: dict[str, int | None] = {k: None for k in CH_OPTIONS}
    if not settings or not Path(settings).exists():
        return out
    inside = False
    for line in Path(settings).read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if s.startswith("["):
            inside = s.lower() == "[custom]"
        elif inside and "=" in s:
            k, v = (x.strip() for x in s.split("=", 1))
            if k in out:
                try:
                    out[k] = int(v)
                except ValueError:
                    pass
    return out


def set_options(settings: Path, values: dict[str, int]) -> bool:
    """Set [custom] options in settings.ini (only CH_OPTIONS). Backs the file up first (once, as
    settings.ini.bak-riffchart). Clone Hero rewrites settings.ini when it exits: run this while it's closed.
    Returns True when the file changed."""
    values = {k: int(bool(v)) for k, v in values.items() if k in CH_OPTIONS}
    text = Path(settings).read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    try:
        start = next(i for i, l in enumerate(lines) if l.strip().lower() == "[custom]")
    except StopIteration:
        lines += ["", "[custom]"]
        start = len(lines) - 1
    end = start + 1
    while end < len(lines) and not lines[end].strip().startswith("["):
        end += 1
    seen = set()
    for i in range(start + 1, end):
        k = lines[i].split("=", 1)[0].strip() if "=" in lines[i] else ""
        if k in values:
            lines[i] = f"{k} = {values[k]}"
            seen.add(k)
    new = [f"{k} = {v}" for k, v in values.items() if k not in seen]
    last = max((i for i in range(start, end) if lines[i].strip()), default=start)
    lines[last + 1:last + 1] = new
    out = "\n".join(lines) + "\n"
    if out == text or out.rstrip("\n") == text.rstrip("\n"):
        return False
    bak = Path(settings).with_name(Path(settings).name + ".bak-riffchart")
    if not bak.exists():
        shutil.copy2(settings, bak)
    Path(settings).write_text(out, encoding="utf-8")
    return True
