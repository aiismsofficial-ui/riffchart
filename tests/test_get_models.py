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
    monkeypatch.setattr(gm.urllib.request, "urlopen", fake_open)
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
