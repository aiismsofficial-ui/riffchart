"""Clone Hero integration: find the exe, launch a song preview, register a song folder in settings.ini.

Preview command line (https://wiki.clonehero.net/books/clone-hero-manual/page/chart-previews-via-the-command-line):
    "Clone Hero.exe" --song "<folder>" --player Guitar,<Difficulty> [-i]
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path


from ..paths import load_config, project_path


def find_exe(cli: str | None = None) -> Path | None:
    cands = [cli, os.environ.get("CLONE_HERO_EXE")]
    if project_path("configs/preview.yaml").exists():
        cands.append(load_config("configs/preview.yaml").get("clone_hero_exe"))
    cands += [r"C:\Program Files\Clone Hero\Clone Hero.exe", r"C:\Program Files (x86)\Clone Hero\Clone Hero.exe",
              str(Path.home() / "AppData/Local/Programs/Clone Hero/Clone Hero.exe"),
              str(Path.home() / "Desktop/Clone Hero/Clone Hero.exe")]
    for c in cands:
        if c and Path(c).is_file():
            return Path(c)
    return None


def preview_command(exe: Path, folder: Path, difficulty: str = "expert") -> list[str]:
    return [str(exe), "--song", str(Path(folder).resolve()), "--player", f"Guitar,{difficulty.capitalize()}", "-i"]


def launch_preview(folder: Path, difficulty: str = "expert", exe: Path | None = None) -> list[str]:
    """Start Clone Hero on one song folder (it must contain the audio). Returns the command line."""
    exe = exe or find_exe()
    if exe is None:
        raise FileNotFoundError("Clone Hero.exe not found: set clone_hero_exe in configs/preview.yaml")
    cmd = preview_command(exe, folder, difficulty)
    subprocess.Popen(cmd, cwd=str(exe.parent))
    return cmd


def is_running() -> bool:
    if os.name != "nt":
        return False
    out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq Clone Hero.exe", "/NH"], capture_output=True, text=True).stdout
    return "Clone Hero.exe" in out


def song_folders(settings: Path) -> list[str]:
    """Song folders configured in Clone Hero's settings.ini ([directories] pathN = ...)."""
    if not settings.exists():
        return []
    out, inside = [], False
    for line in settings.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if s.startswith("["):
            inside = s.lower() == "[directories]"
        elif inside and re.match(r"path\d+\s*=", s):
            out.append(s.split("=", 1)[1].strip())
    return out


def add_song_folder(settings: Path, folder: Path) -> bool:
    """Append `folder` to [directories] unless it is already there. Backs settings.ini up first.
    Clone Hero rewrites settings.ini when it exits, so this must run while it is closed.
    Returns True when the file was changed."""
    folder_s = str(Path(folder).resolve())
    if any(Path(p).resolve() == Path(folder_s) for p in song_folders(settings) if p):
        return False
    lines = settings.read_text(encoding="utf-8", errors="replace").splitlines()
    try:
        start = next(i for i, l in enumerate(lines) if l.strip().lower() == "[directories]")
    except StopIteration:
        lines += ["", "[directories]"]
        start = len(lines) - 1
    end, used = start + 1, set()
    while end < len(lines) and not lines[end].strip().startswith("["):
        m = re.match(r"\s*path(\d+)\s*=", lines[end])
        if m:
            used.add(int(m.group(1)))
        end += 1
    last = max((i for i in range(start + 1, end) if lines[i].strip()), default=start)
    n = next(i for i in range(len(used) + 1) if i not in used)
    lines.insert(last + 1, f"path{n} = {folder_s}")
    bak = settings.with_name(settings.name + ".bak-riffchart")
    if not bak.exists():
        shutil.copy2(settings, bak)
    settings.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True
