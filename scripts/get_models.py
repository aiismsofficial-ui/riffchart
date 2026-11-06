"""Check / download the model files listed in configs/models.yaml, then fetch the Demucs model.

    python scripts/get_models.py [--check] [--no-demucs]

Exit code 0: everything is in place. 2: something is missing (the message says what and where it goes).
Downloads go to <path>.part first and are checked against their SHA-256 before they replace anything.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from riffchart.paths import load_config, project_path  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path, size: int) -> Path:
    if not url.startswith("https://"):
        raise ValueError(f"refusing a non-https download: {url}")
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "RiffChart installer"})
    with urllib.request.urlopen(req, timeout=60) as r, open(part, "wb") as out:
        total = int(r.headers.get("Content-Length") or size or 0)
        done, last = 0, -1
        while block := r.read(1 << 20):
            out.write(block)
            done += len(block)
            pct = int(done * 100 / total) if total else -1
            if pct != last and pct % 5 == 0:
                print(f"\r  {done / 2**20:,.0f} / {total / 2**20:,.0f} MB ({pct}%)", end="", flush=True)
                last = pct
    print()
    return part


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="only check, download nothing")
    ap.add_argument("--no-demucs", action="store_true")
    args = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):         # the Windows console may not be UTF-8
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    cfg = load_config("configs/models.yaml")
    missing = []
    for f in cfg.get("files") or []:
        dest = project_path(f["path"])
        if dest.exists():
            if dest.stat().st_size == f.get("bytes", dest.stat().st_size) and sha256(dest) == f["sha256"]:
                print(f"✓ {f['name']}")
                continue
            print(f"! {f['name']}: {dest} is not the expected file", end="")
            if args.check or not f.get("url"):
                print(" (left as it is)")
                missing.append(f)
                continue
            print(" (downloading it again)")
        if not f.get("url") or args.check:
            print(f"✗ {f['name']}: missing. Put it at {dest}")
            missing.append(f)
            continue
        print(f"↓ {f['name']} ({f.get('bytes', 0) / 2**20:,.0f} MB)")
        try:
            part = download(f["url"], dest, f.get("bytes", 0))
            if sha256(part) != f["sha256"]:
                part.unlink(missing_ok=True)
                raise ValueError("the download doesn't match its checksum")
            part.replace(dest)
            print(f"✓ {f['name']}")
        except Exception as e:
            print(f"✗ {f['name']}: download failed ({e}). Run the installer again, or put the file at {dest}")
            missing.append(f)
    if cfg.get("demucs") and not args.no_demucs and not args.check:
        try:
            from demucs.pretrained import get_model
            print(f"↓ separation model ({cfg['demucs']}, from Meta, once)")
            get_model(cfg["demucs"])
            print("✓ separation model")
        except Exception as e:                      # Demucs fetches it again on the first song
            print(f"! separation model not fetched now ({e}); it downloads with the first song instead")
    if missing:
        print("\nThe app needs the files marked ✗ before it can chart songs. Everything else works already.")
        return 2
    print("\nAll models are in place.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
