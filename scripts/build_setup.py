"""Build RiffChart-Setup-<version>.exe (Inno Setup 6) from the public copy of the app.

    python scripts/build_setup.py [--out ../riffchart-dist] [--iscc "C:/.../ISCC.exe"]

Exports the public files (scripts/export_public.py, leak check included) into <out>/src, then compiles
installer/riffchart.iss with them. The result is <out>/RiffChart-Setup-<version>.exe, a few MB: the
models and Python are downloaded by install.bat when Setup runs. Attach it to the GitHub release.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def find_iscc(given: str | None) -> Path:
    cands = [given, os.environ.get("ISCC")]
    for base in (os.environ.get("LOCALAPPDATA", ""), os.environ.get("ProgramFiles(x86)", ""),
                 os.environ.get("ProgramFiles", "")):
        if base:
            cands.append(str(Path(base) / ("Programs" if base == os.environ.get("LOCALAPPDATA") else "")
                             / "Inno Setup 6" / "ISCC.exe"))
    for c in cands:
        if c and Path(c).is_file():
            return Path(c)
    sys.exit("ISCC.exe (Inno Setup 6) not found: install it (winget install JRSoftware.InnoSetup) or pass --iscc")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT.parent / "riffchart-dist"))
    ap.add_argument("--iscc")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    src = out / "src"
    version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    if src.exists():
        shutil.rmtree(src)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "export_public.py"), str(src)], check=True)
    iscc = find_iscc(a.iscc)
    subprocess.run([str(iscc), "/Q", f"/DSrc={src}", f"/DAppVersion={version}", f"/DOut={out}",
                    str(ROOT / "installer" / "riffchart.iss")], check=True)
    exe = out / f"RiffChart-Setup-{version}.exe"
    print(f"built {exe} ({exe.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
