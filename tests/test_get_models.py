"""scripts/get_models.py: Hugging Face URLs, checksum check, nothing replaced by a bad download (no network)."""
import hashlib
import importlib.util
import io
from pathlib import Path

import yaml

spec = importlib.util.spec_from_file_location("get_models", Path(__file__).parents[1] / "scripts" / "get_models.py")
gm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gm)


class FakeResp(io.BytesIO):
    headers = {}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_download_from_huggingface(tmp_path, monkeypatch):
    good = b"model bytes" * 1000
    cfg = {"huggingface": "someone/riffchart-models", "files": [
        {"name": "A", "path": str(tmp_path / "a" / "model.pt"), "file": "model-a.pt",
         "sha256": hashlib.sha256(good).hexdigest(), "bytes": len(good)}]}
    monkeypatch.setattr(gm, "load_config", lambda p: cfg)
    seen = []

    def fake_open(req, timeout=0):
        seen.append(req.full_url)
        return FakeResp(served[0])
    monkeypatch.setattr(gm, "_open", lambda req, timeout=60: fake_open(gm._authorize(req), timeout))
    served = [b"tampered"]
    assert gm.main(["--no-demucs"]) == 2                                  # wrong checksum: refused
    assert not (tmp_path / "a" / "model.pt").exists() and not list((tmp_path / "a").glob("*.part"))
    assert seen[0] == "https://huggingface.co/someone/riffchart-models/resolve/main/model-a.pt"
    served = [good]
    assert gm.main(["--no-demucs"]) == 0
    assert (tmp_path / "a" / "model.pt").read_bytes() == good
    assert gm.main(["--no-demucs", "--check"]) == 0                       # now present: nothing fetched
    assert len(seen) == 2


def test_models_yaml_lists_every_file():
    cfg = yaml.safe_load((Path(__file__).parents[1] / "configs" / "models.yaml").read_text(encoding="utf-8"))
    paths = {f["path"] for f in cfg["files"]}
    assert paths == {"data/checkpoints/chart/full_v2/step60000.pt", "data/checkpoints/beat_finetune/best.ckpt",
                     "data/features/model/mel_stats.npz", "data/checkpoints/reduce/reduce.pkl"}
    assert all(len(f["sha256"]) == 64 and f["file"] for f in cfg["files"])


def test_private_repo_needs_token(tmp_path, monkeypatch):
    import urllib.error
    good = b"weights" * 100
    cfg = {"huggingface": "riffchart/riffchart-models", "files": [
        {"name": "A", "path": str(tmp_path / "m.pt"), "file": "m.pt", "sha256": hashlib.sha256(good).hexdigest(),
         "bytes": len(good)}]}
    monkeypatch.setattr(gm, "load_config", lambda p: cfg)
    reqs = []

    def fake_open(req, timeout=0):
        reqs.append(req)
        if "Authorization" not in req.unredirected_hdrs:
            raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, None)
        return FakeResp(good)
    monkeypatch.setattr(gm, "_open", lambda req, timeout=60: fake_open(gm._authorize(req), timeout))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert gm.main(["--no-demucs"]) == 3                                  # private: ask for a token
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    assert gm.main(["--no-demucs"]) == 0
    r = reqs[-1]
    assert r.unredirected_hdrs["Authorization"] == "Bearer hf_test"
    assert "Authorization" not in r.headers                              # so redirects never carry it
    cfg["files"][0]["url"] = "https://example.com/m.pt"                  # another host never gets the token
    (tmp_path / "m.pt").unlink()
    reqs.clear()
    gm.main(["--no-demucs"])
    assert "Authorization" not in reqs[-1].unredirected_hdrs


def test_token_follows_huggingface_redirects_only(monkeypatch):
    import urllib.request
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    h = gm._HFRedirect()
    req = gm._authorize(urllib.request.Request("https://huggingface.co/riffchart/riffchart-models/resolve/main/a.npz"))
    same = h.redirect_request(req, None, 307, "", {}, "https://huggingface.co/api/resolve-cache/models/x/a.npz")
    cdn = h.redirect_request(req, None, 302, "", {}, "https://us.aws.cdn.hf.co/xet-bridge-us/abc?Signature=x")
    evil = h.redirect_request(req, None, 302, "", {}, "https://huggingface.co.evil.io/a")
    assert same.unredirected_hdrs.get("Authorization") == "Bearer hf_test"
    assert "Authorization" not in cdn.unredirected_hdrs and "Authorization" not in cdn.headers
    assert "Authorization" not in evil.unredirected_hdrs
    plain = gm._authorize(urllib.request.Request("http://huggingface.co/x"))          # not https: no token
    assert "Authorization" not in plain.unredirected_hdrs
