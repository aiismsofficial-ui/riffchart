"""Check / download the model files listed in configs/models.yaml, then fetch the Demucs model.

    python scripts/get_models.py [--check] [--no-demucs]

Exit code 0: everything is in place. 2: something is missing (the message says what and where it goes).
3: the model repo is private and needs a Hugging Face read token: set HF_TOKEN (the installer asks for one).
The token is only sent to huggingface.co itself (also on its own redirects, e.g. /api/resolve-cache/...), never
to the download servers it redirects to (the CDN gets a signed link instead).
Downloads go to <path>.part first and are checked against their SHA-256 before they replace anything.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.error
import urllib.parse
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


HF_HOST = "huggingface.co"


def _token() -> str:
    return os.environ.get("HF_TOKEN", "").strip()


def _authorize(req: urllib.request.Request) -> urllib.request.Request:
    """Add the token only for https://huggingface.co itself (unredirected: urllib never copies it onward)."""
    u = urllib.parse.urlsplit(req.full_url)
    if _token() and u.scheme == "https" and u.hostname == HF_HOST:
        req.add_unredirected_header("Authorization", f"Bearer {_token()}")
    return req


class _HFRedirect(urllib.request.HTTPRedirectHandler):
    """Hugging Face redirects small files to huggingface.co/api/resolve-cache/... (needs the token again) and
    large ones to its CDN with a signed link (must not get the token). Re-add it only for huggingface.co."""
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        return _authorize(new) if new is not None else None


def _open(req: urllib.request.Request, timeout: int = 60):
    return urllib.request.build_opener(_HFRedirect).open(_authorize(req), timeout=timeout)


def token_problem(repo: str) -> str:
    """Why the token didn't work: not valid at all, or valid but can't see the repo (a fine-grained token
    without access to the organisation)."""
    try:
        with _open(urllib.request.Request(f"https://{HF_HOST}/api/whoami-v2",
                                          headers={"User-Agent": "RiffChart installer"}), timeout=30) as r:
            import json
            name = json.loads(r.read()).get("name", "?")
    except urllib.error.HTTPError as e:
        if e.code == 401:
            return "Hugging Face doesn't accept that token. Check it was copied whole (it starts with hf_)."
        return f"Hugging Face couldn't check the token (HTTP {e.code})."
    except Exception as e:
        return f"Hugging Face couldn't check the token ({e})."
    return (f"The token works (account {name}) but can't see {repo}. Make a token of type Read, or give a "
            f"fine-grained token read access to the {repo.split('/')[0]} organisation's repos.")


def download(url: str, dest: Path, size: int) -> Path:
    if not url.startswith("https://"):
        raise ValueError(f"refusing a non-https download: {url}")
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "RiffChart installer"})
    with _open(req) as r, open(part, "wb") as out:
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
    missing, need_token = [], False
    repo = (cfg.get("huggingface") or "").strip().strip("/")
    for f in cfg.get("files") or []:
        dest = project_path(f["path"])
        if not f.get("url") and repo and f.get("file"):
            f["url"] = f"https://huggingface.co/{repo}/resolve/main/{f['file']}"
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
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 404) and "huggingface.co" in f["url"]:
                need_token = True
                why = ("the token was refused" if _token() else
                       "the model repo is private, so a Hugging Face read token is needed")
                print(f"✗ {f['name']}: {why}")
            else:
                print(f"✗ {f['name']}: download failed (HTTP {e.code}). Run the installer again, or put the file at {dest}")
            missing.append(f)
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
    if need_token and _token() and repo:
        print("\n" + token_problem(repo))
    if missing:
        print("\nThe app needs the files marked ✗ before it can chart songs. Everything else works already.")
        return 3 if need_token else 2
    print("\nAll models are in place.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
