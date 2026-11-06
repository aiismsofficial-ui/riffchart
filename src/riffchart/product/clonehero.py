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


EXE = "Clone Hero.exe"


def exe_path(p: str | Path | None) -> Path | None:
    """A path the user gave: the exe itself, or the folder it's in."""
    if not p or not str(p).strip():
        return None
    p = Path(str(p).strip().strip('"')).expanduser()
    if p.is_dir():
        p = p / EXE
    return p if p.is_file() and p.suffix.lower() == ".exe" else None


def from_player_log() -> Path | None:
    """Clone Hero is a portable zip people unpack anywhere, but its Unity log names the folder it ran from
    ("... at path D:/Games/Clone Hero/Clone Hero_Data/UnitySubsystems")."""
    logs = Path.home() / "AppData" / "LocalLow" / "srylain Inc_" / "Clone Hero"
    for name in ("Player.log", "Player-prev.log"):
        try:
            text = (logs / name).read_text(encoding="utf-8", errors="replace")[:200_000]
        except OSError:
            continue
        for m in re.finditer(r"([A-Za-z]:[\\/][^\r\n'\"]*?)[\\/]Clone Hero_Data", text):
            exe = exe_path(m.group(1))
            if exe:
                return exe
    return None


def find_exe(cli: str | None = None, configured: str | None = None) -> Path | None:
    """Clone Hero.exe: given on the command line, CLONE_HERO_EXE, the app's setup (configured), configs/
    preview.yaml, the folder Clone Hero last ran from (its log), then the usual places."""
    cands = [cli, os.environ.get("CLONE_HERO_EXE"), configured]
    if project_path("configs/preview.yaml").exists():
        cands.append(load_config("configs/preview.yaml").get("clone_hero_exe"))
    for c in cands:
        exe = exe_path(c)
        if exe:
            return exe
    exe = from_player_log()
    if exe:
        return exe
    home = Path.home()
    places = [Path(os.environ.get("ProgramFiles", r"C:\Program Files")), Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")),
              home / "AppData/Local/Programs", home / "Desktop", home / "Downloads", home / "Documents", home / "Games"]
    for drive in "CDEFG":
        places += [Path(f"{drive}:/"), Path(f"{drive}:/Games")]
    for base in places:
        for name in ("Clone Hero", "CloneHero"):
            exe = exe_path(base / name)
            if exe:
                return exe
    return None


def preview_command(exe: Path, folder: Path, difficulty: str = "expert") -> list[str]:
    return [str(exe), "--song", str(Path(folder).resolve()), "--player", f"Guitar,{difficulty.capitalize()}", "-i"]


def launch_preview(folder: Path, difficulty: str = "expert", exe: Path | None = None) -> list[str]:
    """Start Clone Hero on one song folder (it must contain the audio). Returns the command line."""
    exe = exe or find_exe()
    if exe is None:
        raise FileNotFoundError("Clone Hero.exe wasn't found: choose it under Settings in the library")
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
