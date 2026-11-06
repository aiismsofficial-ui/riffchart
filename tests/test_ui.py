"""riffchart-ui API: queue, library, export, read-only protection, Clone Hero settings (no GPU: fake charter)."""
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from riffchart.product import clonehero
from riffchart.product.core import safe_name
from riffchart.product.meta import read_meta
from riffchart.ui.jobs import Options
from riffchart.ui.library import ProtectedPathError, check_writable
from riffchart.ui.server import create_app

CHART = ("[Song]\n{\n  Resolution = 192\n  Offset = 0\n}\n[SyncTrack]\n{\n  0 = TS 4\n  0 = B 120000\n}\n"
         "[Events]\n{\n}\n[ExpertSingle]\n{\n  0 = N 0 0\n  192 = N 1 0\n  384 = N 2 96\n}\n"
         "[HardSingle]\n{\n  0 = N 0 0\n}\n")


class FakeCharter:
    """Stands in for RiffCharter.chart_file: writes a tiny song folder, reports the stages."""
    def __init__(self):
        self.calls = []

    def chart_file(self, path, out_root, difficulties, style, write_stems=True, overwrite=False, on_stage=None,
                   meta=None, folder_name=None):
        self.calls.append((Path(path), list(difficulties), style))
        m = meta or read_meta(path)
        folder = Path(out_root) / (folder_name or safe_name(f"{m.artist} - {m.title}"))
        if (folder / "notes.chart").exists() and not overwrite:
            return None
        for s in ("load", "separate", "beats", "chart", "post", "write"):
            on_stage and on_stage(s)
        if Path(path).stem == "broken":
            raise RuntimeError("cannot decode")
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "notes.chart").write_text(CHART, encoding="utf-8")
        (folder / "song.ini").write_text(f"[song]\nname = {m.title}\nartist = {m.artist}\nsong_length = 61000\n"
                                         "diff_guitar = 3\ncharter = riffchart\n", encoding="utf-8")
        (folder / "song.ogg").write_bytes(b"OggS")
        return SimpleNamespace(folder=folder, n_notes={"expert": 3, "hard": 1}, seconds={"chart": 1.0})


@pytest.fixture
def env(tmp_path):
    lib_ro = tmp_path / "my library"
    (lib_ro / "Band - Old").mkdir(parents=True)
    (lib_ro / "Band - Old" / "song.ogg").write_bytes(b"x")
    settings = tmp_path / "settings.ini"
    settings.write_text("[general]\nx = 1\n\n[directories]\npath0 = C:\\songs\n\n[video]\ny = 2\n", encoding="utf-8")
    cfg = {"workspace": tmp_path / "ws" / "songs", "uploads": tmp_path / "ws" / "uploads",
           "export_dir": tmp_path / "export", "protected": [lib_ro], "clone_hero_settings": settings,
           "log_file": tmp_path / "ui.log",
           "stage_weights": {"load": .1, "separate": .2, "beats": .1, "chart": .5, "post": .05, "write": .05}}
    fake = FakeCharter()
    app = create_app(cfg, make_charter=lambda: fake)
    c = TestClient(app)
    yield SimpleNamespace(client=c, cfg=cfg, fake=fake, app=app, lib_ro=lib_ro, tmp=tmp_path)
    app.state.queue.shutdown()


def wait_idle(c, timeout=10):
    t = time.time()
    while time.time() - t < timeout:
        jobs = c.get("/api/jobs").json()
        if jobs and all(j["status"] not in ("queued", "running") for j in jobs):
            return jobs
        time.sleep(0.05)
    raise TimeoutError(jobs)


def test_options_validation():
    assert Options.parse({"difficulties": ["easy", "expert"]}).difficulties == ["expert", "easy"]
    for bad in ({"difficulties": []}, {"density": 3}, {"chords": "lots"}):
        with pytest.raises(ValueError):
            Options.parse(bad)


def test_upload_charts_lists_and_exports(env):
    c = env.client
    r = c.post("/api/jobs/upload", files=[("files", ("Album/Muse - Uprising.mp3", b"ID3", "audio/mpeg")),
                                          ("files", ("cover.jpg", b"x", "image/jpeg"))],
               data={"options": json.dumps({"difficulties": ["expert", "hard"], "density": -0.5, "chords": "low"})})
    assert r.status_code == 200, r.text
    assert len(r.json()["added"]) == 1 and "cover.jpg" in r.json()["skipped"][0]
    jobs = wait_idle(c)
    assert [j["status"] for j in jobs] == ["done"] and jobs[0]["song"] == "Muse - Uprising"
    path, diffs, style = env.fake.calls[0]
    assert diffs == ["expert", "hard"] and style.density == -0.5 and style.chords == "low"
    assert not path.exists()                                       # upload copy cleaned up after charting

    songs = c.get("/api/songs").json()
    assert len(songs) == 1
    s = songs[0]
    assert (s["title"], s["artist"], s["notes"], s["length_s"]) == ("Uprising", "Muse", {"expert": 3, "hard": 1}, 61.0)
    assert not s["exported"]

    r = c.post("/api/songs/Muse - Uprising/export", json={})
    assert r.status_code == 200
    assert (env.cfg["export_dir"] / "Muse - Uprising" / "notes.chart").read_text(encoding="utf-8") == CHART
    assert c.post("/api/songs/Muse - Uprising/export", json={}).status_code == 409     # no silent overwrite
    assert c.post("/api/songs/Muse - Uprising/export", json={"overwrite": True}).status_code == 200
    assert c.get("/api/songs").json()[0]["exported"]
    assert c.get("/api/songs/Muse - Uprising/file/song.ogg").content == b"OggS"


def test_skip_existing_then_retry_overwrites(env):
    c = env.client
    up = lambda: c.post("/api/jobs/upload", files=[("files", ("A - B.ogg", b"x", "audio/ogg"))])   # noqa: E731
    up(); wait_idle(c)
    up(); jobs = wait_idle(c)
    assert [j["status"] for j in jobs] == ["done", "skipped"]
    # skipped -> retry re-charts with overwrite (the upload copy is gone, so retry needs the file: refused)
    assert c.post(f"/api/jobs/{jobs[1]['id']}/retry").status_code == 409
    assert c.post("/api/jobs/clear").json()["removed"] == 2 and c.get("/api/jobs").json() == []


def test_failed_job_keeps_upload_for_retry(env):
    c = env.client
    c.post("/api/jobs/upload", files=[("files", ("broken.wav", b"x", "audio/wav"))])
    jobs = wait_idle(c)
    assert jobs[0]["status"] == "failed" and "cannot decode" in jobs[0]["message"]
    assert Path(jobs[0]["source"]).exists()
    assert c.post(f"/api/jobs/{jobs[0]['id']}/retry").status_code == 200
    assert wait_idle(c)[0]["status"] == "failed"


def test_add_by_path_reads_library_without_writing(env):
    c = env.client
    before = sorted(p.name for p in env.lib_ro.rglob("*"))
    r = c.post("/api/jobs/paths", json={"paths": [str(env.lib_ro), str(env.tmp / "nope")]})
    assert len(r.json()["added"]) == 1 and "not found" in r.json()["skipped"][0]
    assert wait_idle(c)[0]["song"] == "Band - Old"
    assert sorted(p.name for p in env.lib_ro.rglob("*")) == before                     # library untouched


def test_protected_folders_refuse_writes(env, tmp_path):
    with pytest.raises(ProtectedPathError):
        check_writable(env.lib_ro / "x", [env.lib_ro])
    with pytest.raises(ProtectedPathError):
        check_writable(env.tmp, [env.lib_ro])                       # a parent of the library also counts
    check_writable(env.tmp / "elsewhere", [env.lib_ro])
    cfg = dict(env.cfg, export_dir=env.lib_ro / "exports", workspace=tmp_path / "ws2")
    app = create_app(cfg, make_charter=FakeCharter, start_worker=False)
    lib = app.state.lib
    (lib.workspace / "S").mkdir()
    (lib.workspace / "S" / "notes.chart").write_text(CHART, encoding="utf-8")
    with pytest.raises(ProtectedPathError):
        lib.export("S")
    with pytest.raises(ProtectedPathError):
        create_app(dict(env.cfg, workspace=env.lib_ro / "ws"), make_charter=FakeCharter, start_worker=False)


def test_song_ids_cannot_escape_workspace(env):
    c = env.client
    for bad in ("..", "..%2F..%2Fsettings.ini", "nope"):
        assert c.get(f"/api/songs/{bad}").status_code in (404, 405)
    assert c.post("/api/songs/nope/export", json={}).status_code == 404


def test_queue_survives_restart(env):
    q = env.app.state.queue
    q.shutdown(); time.sleep(0.1)
    j = q.add(env.tmp / "x.mp3", "x", Options())
    j.status = "running"; q._save()
    app2 = create_app(env.cfg, make_charter=FakeCharter, start_worker=False)
    j2 = app2.state.queue.jobs[j.id]
    assert j2.status == "queued" and j2.message == "restarted"


def test_clone_hero_settings_add_folder(tmp_path):
    s = tmp_path / "settings.ini"
    s.write_text("[general]\nx = 1\n\n[directories]\npath0 = C:\\a\npath1 = C:\\b\n\n[video]\ny = 2\n", encoding="utf-8")
    new = tmp_path / "exports"
    assert clonehero.add_song_folder(s, new)
    assert clonehero.song_folders(s) == ["C:\\a", "C:\\b", str(new.resolve())]
    assert "[video]\ny = 2" in s.read_text(encoding="utf-8")
    assert (tmp_path / "settings.ini.bak-riffchart").exists()
    assert not clonehero.add_song_folder(s, new)                    # already there: unchanged
    cmd = clonehero.preview_command(Path("C:/CH/Clone Hero.exe"), tmp_path, "hard")
    assert cmd[1:4] == ["--song", str(tmp_path.resolve()), "--player"] and cmd[4] == "Guitar,Hard"


def test_status_and_static(env):
    c = env.client
    st = c.get("/api/status").json()
    assert st["model"] == "not loaded" and not st["clone_hero"]["export_registered"]
    assert "RiffChart" in c.get("/").text
    r = c.get("/app.js")
    assert r.status_code == 200 and r.headers["cache-control"] == "no-cache"     # no stale page after updates
    assert c.get("/audio.js").status_code == 200 and "cache-control" not in c.get("/api/status").headers


# ---------------------------------------------------------------- editor data (milestone 2)
from riffchart.formats.chart_parser import parse_chart_text            # noqa: E402
from riffchart.formats.chart_writer import write_chart_text            # noqa: E402
from riffchart.ui.chartjson import json_to_song, peaks, song_to_json   # noqa: E402

RICH = ("[Song]\n{\n  Resolution = 192\n  Offset = 0\n}\n[SyncTrack]\n{\n  0 = TS 4\n  0 = B 120000\n"
        "  768 = TS 3\n  768 = B 150000\n}\n[Events]\n{\n  0 = E \"section Intro\"\n  768 = E \"section Verse\"\n}\n"
        "[ExpertSingle]\n{\n  0 = N 0 0\n  48 = N 1 0\n  96 = N 2 0\n  96 = N 3 0\n  192 = N 7 96\n"
        "  384 = N 4 0\n  384 = N 5 0\n  480 = N 0 0\n  480 = N 6 0\n  768 = S 2 384\n  768 = N 1 192\n"
        "  1152 = E solo\n  1152 = N 2 0\n  1344 = E soloend\n}\n[EasySingle]\n{\n  0 = N 0 0\n}\n")


def test_chart_json_round_trip_is_lossless():
    a = parse_chart_text(RICH)
    d = song_to_json(a, {"delay": "50"})
    assert d["offset_s"] == 0.05 and d["sections"] == [[0, "Intro"], [768, "Verse"]]
    assert d["ts"] == [[0, 4, 4], [768, 3, 4]] and d["tempo"] == [[0, 120.0], [768, 150.0]]
    ex = d["tracks"]["expert"]
    assert ex["star_power"] == [[768, 384]] and ex["solos"] and ex["notes"][0] == [0, 1, 0, 0]
    assert json.loads(json.dumps(d)) == d                                # plain JSON
    b = json_to_song(json.loads(json.dumps(d)))
    assert write_chart_text(b) == write_chart_text(a)                    # same .chart out


def test_json_to_song_merges_duplicates_and_drops_bad_rows():
    d = song_to_json(parse_chart_text(RICH))
    d["tracks"]["expert"]["notes"] += [[0, 2, 96, 0], [5000, 0, 0, 0], [-10, 1, 0, 0]]
    tr = json_to_song(d).tracks["expert"]
    assert int(tr.mask[0]) == 0b11 and int(tr.sustain[0]) == 96          # G + R chord at tick 0
    assert 5000 not in tr.tick and (tr.tick >= 0).all()


def test_peaks_envelope_and_cache(tmp_path):
    import numpy as np
    import soundfile as sf
    sr = 8000
    x = np.zeros(sr * 2, np.float32)
    x[sr:] = 0.5 * np.sin(np.arange(sr) / sr * 2 * np.pi * 220)        # silent second, then a tone
    f = tmp_path / "guitar.wav"
    sf.write(f, x, sr)
    p = peaks(f, tmp_path / "cache", per_s=100)
    import base64
    q = np.frombuffer(base64.b64decode(p["data"]), np.uint8)
    assert p["name"] == "guitar" and len(q) == 200 and abs(p["duration_s"] - 2.0) < 0.02
    assert q[:95].max() == 0 and q[110:].min() > 200
    assert len(list((tmp_path / "cache").glob("*.json"))) == 1
    assert peaks(f, tmp_path / "cache", per_s=100) == p                  # served from cache


def test_chart_and_peaks_endpoints(env):
    c = env.client
    c.post("/api/jobs/upload", files=[("files", ("Muse - Uprising.mp3", b"ID3", "audio/mpeg"))])
    wait_idle(c)
    d = c.get("/api/songs/Muse - Uprising/chart").json()
    assert d["title"] == "Uprising" and d["stems"] == ["song.ogg"] and d["resolution"] == 192
    assert [n[0] for n in d["tracks"]["expert"]["notes"]] == [0, 192, 384]
    pk = c.get("/api/songs/Muse - Uprising/peaks").json()
    assert pk[0]["name"] == "song" and "error" in pk[0]                  # fake ogg: reported, not a 500
    assert c.get("/editor.html").status_code == 200


# ---------------------------------------------------------------- saving (milestone 3)
def _charted(env, name="Muse - Uprising"):
    env.client.post("/api/jobs/upload", files=[("files", (f"{name}.mp3", b"ID3", "audio/mpeg"))])
    wait_idle(env.client)
    return env.cfg["workspace"] / name


def test_save_writes_chart_keeps_history_and_types(env):
    c, folder = env.client, _charted(env)
    before = (folder / "notes.chart").read_text(encoding="utf-8")
    d = c.get("/api/songs/Muse - Uprising/chart").json()
    ex = d["tracks"]["expert"]
    ex["notes"] = [[0, 1, 0, 0], [48, 2, 0, 1], [96, 4, 0, 0], [192, 0b11000, 96, 0], [480, 32, 0, 2]]
    ex["star_power"] = [[0, 100]]
    d["sections"] = [[0, "Intro"]]
    r = c.post("/api/songs/Muse - Uprising/chart", json=d)
    assert r.status_code == 200, r.text
    hist = list((folder / ".riffchart" / "history").glob("*-notes.chart"))
    assert len(hist) == 1 and hist[0].read_text(encoding="utf-8") == before        # old version kept
    d2 = c.get("/api/songs/Muse - Uprising/chart").json()
    # final types survive exactly: 96 is a natural HOPO sent as strum, so the file needs a forced flag
    assert d2["tracks"]["expert"]["notes"] == ex["notes"]
    lines = (folder / "notes.chart").read_text(encoding="utf-8")
    assert "  96 = N 5 0" in lines and "  48 = N 5 0" not in lines and "  480 = N 6 0" in lines
    assert d2["tracks"]["expert"]["star_power"] == [[0, 100]] and d2["sections"] == [[0, "Intro"]]
    assert d2["tracks"]["hard"] == d["tracks"]["hard"]                                # untouched difficulty
    assert "Name = \"Uprising\"" in (folder / "notes.chart").read_text(encoding="utf-8")


def test_save_refuses_bad_or_empty_and_moves_mid_aside(env):
    c, folder = env.client, _charted(env)
    d = c.get("/api/songs/Muse - Uprising/chart").json()
    assert c.post("/api/songs/Muse - Uprising/chart", json={"tracks": {}}).status_code == 400
    empty = dict(d, tracks={k: dict(v, notes=[]) for k, v in d["tracks"].items()})
    assert c.post("/api/songs/Muse - Uprising/chart", json=empty).status_code == 400
    (folder / "notes.mid").write_bytes(b"MThd")
    assert c.post("/api/songs/Muse - Uprising/chart", json=d).status_code == 200
    assert not (folder / "notes.mid").exists() and (folder / "notes.mid.bak").exists()
    assert any(p.name.endswith("notes.mid") for p in (folder / ".riffchart" / "history").iterdir())


def test_export_skips_history(env):
    c, folder = env.client, _charted(env)
    d = c.get("/api/songs/Muse - Uprising/chart").json()
    c.post("/api/songs/Muse - Uprising/chart", json=d)
    c.post("/api/songs/Muse - Uprising/export", json={})
    assert not (env.cfg["export_dir"] / "Muse - Uprising" / ".riffchart").exists()


# ---------------------------------------------------------------- playability check (milestone 4)
def test_check_locates_issues(env):
    c = env.client
    tempo = [[0, 120.0]]                       # 192 ticks = 0.5 s
    notes = [[0, 1, 300, 0],                   # sustain runs past the next note (tick 192)
             [192, 2, 0, 0],
             [194, 4, 0, 0],                   # 5 ms later: stacked
             [384, 4, 0, 1],                   # HOPO... after a different fret: fine
             [400, 4, 0, 1],                   # HOPO on the same fret as before -> bad_hopo (and too fast)
             [768, 0b11111, 0, 0],             # 5-fret chord: too big
             [960, 1, 4, 0]]                   # 4-tick tail: below the sustain cutoff, not a tail
    body = {"resolution": 192, "tempo": tempo, "ts": [[0, 4, 4]], "sections": [], "offset_s": 0,
            "audio_s": 60, "tracks": {"expert": {"notes": notes, "star_power": [], "solos": []}}}
    r = c.post("/api/check", json=body).json()
    got = {(x["rule"], x["tick"]) for x in r["expert"]["issues"]}
    assert ("sustain_overlap", 0) in got and ("stacked", 194) in got and ("bad_hopo", 400) in got
    assert ("chord_size", 768) in got and ("too_fast", 400) in got
    assert not any(t == 960 for _, t in got)
    assert r["expert"]["errors"] == sum(x["severity"] == "error" for x in r["expert"]["issues"])
    assert r["hard"] == {"issues": [], "errors": 0, "warnings": 0}
    x = next(x for x in r["expert"]["issues"] if x["rule"] == "sustain_overlap")
    assert "after the next note" in x["detail"] and x["title"]
    body["audio_s"] = 1.0                      # notes after the audio end
    assert any(x["rule"] == "out_of_audio" for x in c.post("/api/check", json=body).json()["expert"]["issues"])
    assert c.post("/api/check", json={"tracks": 1}).status_code == 400


def test_find_issues_matches_check_counts():
    import numpy as np
    from riffchart.eval.metrics import Events
    from riffchart.eval.playability import RULES, check, find_issues
    from riffchart.ui.checker import load_rules
    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0, 30, 400))
    ev = Events(t, rng.integers(1, 64, 400).astype(np.uint8), rng.choice([0, 0, 0.02, 0.3], 400), rng.integers(0, 3, 400).astype(np.uint8))
    rules = load_rules()["expert"]
    c = check(ev, rules, 30.0)
    issues = find_issues(ev, rules, 30.0)
    for r in RULES:
        assert abs(c[f"pv_{r}"] * 0.5 - sum(1 for k, _ in issues if k == r)) < 1e-9


# ---------------------------------------------------------------- library actions (UI rework)
def test_clean_names_from_downloads():
    from riffchart.product.meta import SongMeta, clean
    assert clean(SongMeta("whiskey & coke (Official Visualizer) - LØLØ", "LØLØ")) == SongMeta("whiskey & coke", "LØLØ")
    assert clean(SongMeta("One [Remastered 2008]", "Metallica")).title == "One"
    assert clean(SongMeta("Song (feat. X) [Official HD Video]", "Y")).title == "Song (feat. X)"
    assert clean(SongMeta("Video Killed the Radio Star", "The Buggles")).title == "Video Killed the Radio Star"
    assert clean(SongMeta("Sellout (Live)", "Biohazard")).title == "Sellout (Live)"
    assert clean(SongMeta("a - b (Lyrics)", "Unknown Artist")) == SongMeta("b", "a")


def test_rechart_in_place_keeps_names_and_history(env):
    import numpy as np
    import soundfile as sf
    c, folder = env.client, _charted(env)
    sr = 8000
    sf.write(folder / "song.ogg", np.zeros((sr, 2), np.float32), sr, format="OGG", subtype="VORBIS")
    sf.write(folder / "guitar.ogg", 0.1 * np.ones((sr, 2), np.float32), sr, format="OGG", subtype="VORBIS")
    c.post("/api/songs/Muse - Uprising/meta", json={"title": "Uprising (my edit)", "artist": "Muse"})
    r = c.post("/api/songs/Muse - Uprising/rechart", json={"options": {"density": -0.5, "difficulties": ["expert"]}})
    assert r.status_code == 200, r.text
    jobs = wait_idle(c)
    assert jobs[-1]["status"] == "done" and jobs[-1]["song"] == "Muse - Uprising"
    path, diffs, style = env.fake.calls[-1]
    assert style.density == -0.5 and diffs == ["expert"] and not path.exists()     # temp mix removed
    assert [s["id"] for s in c.get("/api/songs").json()] == ["Muse - Uprising"]   # same folder, no duplicate


def test_rename_and_remove_song(env):
    c, folder = env.client, _charted(env)
    r = c.post("/api/songs/Muse - Uprising/meta", json={"title": "Uprising", "artist": "MUSE"})
    assert r.json()["artist"] == "MUSE" and "artist = MUSE" in (folder / "song.ini").read_text(encoding="utf-8")
    assert c.post("/api/songs/Muse - Uprising/meta", json={"title": " "}).status_code == 400
    assert c.delete("/api/songs/Muse - Uprising").status_code == 200
    assert c.get("/api/songs").json() == [] and not folder.exists()
    trash = list((env.cfg["workspace"].parent / "trash").iterdir())
    assert len(trash) == 1 and (trash[0] / "notes.chart").exists()                 # moved, not deleted


PNG = b"\x89PNG\r\n\x1a\n" + bytes(40)
JPG = b"\xff\xd8\xff\xe0" + bytes(40)


def test_song_details_cover_and_update_in_clone_hero(env):
    import os
    import time
    c, folder = env.client, _charted(env)
    base = "/api/songs/Muse - Uprising"
    r = c.post(f"{base}/meta", json={"title": "Uprising", "artist": "Muse", "album": "The Resistance",
                                      "genre": "Rock", "year": "2009", "diff_guitar": "4"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert (d["album"], d["genre"], d["year"], d["diff_guitar"]) == ("The Resistance", "Rock", "2009", "4")
    ini = (folder / "song.ini").read_text(encoding="utf-8")
    assert "album = The Resistance" in ini and "year = 2009" in ini and ini.count("diff_guitar") == 1
    assert ini.startswith("[song]") and "song_length = " in ini and "charter = " in ini   # other lines kept
    # clearing a field removes the line; newlines can't sneak extra keys into song.ini
    c.post(f"{base}/meta", json={"album": "", "genre": "Rock\ncharter = me"})
    ini = (folder / "song.ini").read_text(encoding="utf-8")
    assert "album" not in ini and "genre = Rock charter = me" in ini and ini.count("charter =") == 2
    assert "\ncharter = me" not in ini
    for bad in ({"year": "09/2009"}, {"diff_guitar": "9"}, {"title": ""}):
        assert c.post(f"{base}/meta", json=bad).status_code == 400
    # cover: png / jpg by content, one at a time, anything else refused
    assert c.post(f"{base}/art", files={"file": ("x.png", b"GIF89a....")}).status_code == 400
    r = c.post(f"{base}/art", files={"file": ("cover.jpeg", JPG)})
    assert r.json()["art"].startswith("album.jpg") and (folder / "album.jpg").read_bytes() == JPG
    c.post(f"{base}/art", files={"file": ("cover.png", PNG)})
    assert (folder / "album.png").exists() and not (folder / "album.jpg").exists()
    assert c.get(f"{base}/file/album.png").content == PNG
    # save & update in Clone Hero: exports (first time), then edits show as stale until updated again
    r = c.post(f"{base}/meta", json={"title": "Uprising", "export": True})
    assert r.json()["exported"] and not r.json()["export_stale"]
    dest = env.cfg["export_dir"] / "Muse - Uprising"
    assert (dest / "album.png").exists() and "genre = Rock" in (dest / "song.ini").read_text(encoding="utf-8")
    c.post(f"{base}/meta", json={"artist": "MUSE"})
    later = time.time() + 5
    os.utime(folder / "song.ini", (later, later))
    assert c.get(base).json()["export_stale"]
    assert c.get("/api/songs").json()[0]["artist"] == "MUSE"                 # list isn't stuck on a cached name
    r = c.post(f"{base}/meta", json={"artist": "MUSE", "export": True})
    assert not r.json()["export_stale"] and "artist = MUSE" in (dest / "song.ini").read_text(encoding="utf-8")
    assert c.delete(f"{base}/art").json()["art"] == "" and c.get(base).json()["export_stale"]


def test_cover_art_from_tags(tmp_path):
    import numpy as np
    import soundfile as sf
    from mutagen.flac import FLAC, Picture
    from riffchart.product.meta import cover_art
    f = tmp_path / "a.flac"
    sf.write(f, np.zeros(800, np.float32), 8000)
    assert cover_art(f) is None
    fl = FLAC(f)
    back, front = Picture(), Picture()
    back.type, back.mime, back.data = 4, "image/png", PNG
    front.type, front.mime, front.data = 3, "image/jpeg", JPG
    fl.add_picture(back)
    fl.add_picture(front)
    fl.save()
    assert cover_art(f) == JPG                                                # front cover wins
    assert cover_art(tmp_path / "missing.mp3") is None


def test_tidy_names_preview_and_apply(env):
    c, folder = env.client, _charted(env)
    ini = folder / "song.ini"
    ini.write_text(ini.read_text(encoding="utf-8").replace("name = Uprising", "name = Uprising - MuseVEVO"),
                   encoding="utf-8")
    ch = c.get("/api/tidy").json()
    assert [(x["old"]["title"], x["new"]["title"]) for x in ch] == [("Uprising - MuseVEVO", "Uprising")]
    c.post("/api/songs/Muse - Uprising/export", json={})
    r = c.post("/api/tidy", json={"ids": ["Muse - Uprising"], "export": True}).json()
    assert r["renamed"] == ["Muse - Uprising"] and r["exported"] == ["Muse - Uprising"] and not r["failed"]
    assert "name = Uprising\n" in ini.read_text(encoding="utf-8")
    assert "name = Uprising\n" in (env.cfg["export_dir"] / "Muse - Uprising" / "song.ini").read_text(encoding="utf-8")
    assert c.get("/api/tidy").json() == []
    assert c.post("/api/tidy", json={"ids": ["nope"]}).json()["renamed"] == []


def test_background_video_and_online_cover_endpoints(env, monkeypatch):
    import io
    from PIL import Image
    from riffchart.product import covers

    def img(color, fmt="PNG"):
        b = io.BytesIO()
        Image.new("RGB", (200, 200), color).save(b, fmt)
        return b.getvalue()
    c, folder = env.client, _charted(env)
    base = "/api/songs/Muse - Uprising"
    d = c.post(f"{base}/art", files={"file": ("c.png", img((200, 30, 30)))}).json()
    assert d["background"]["mode"] == "auto" and d["background"]["file"].startswith("background.jpg")
    assert c.get(f"{base}/file/background.jpg").status_code == 200
    d = c.post(f"{base}/background", json={"mode": "none"}).json()
    assert d["background"] == {"file": "", "mode": "none"} and not (folder / "background.jpg").exists()
    assert c.post(f"{base}/background", json={"mode": "own"}).status_code == 400
    d = c.post(f"{base}/background/file", files={"file": ("bg.jpg", img((1, 1, 1), "JPEG"))}).json()
    assert d["background"]["mode"] == "own" and (folder / "background.jpg").exists()
    d = c.post(f"{base}/background", json={"mode": "auto"}).json()
    assert d["background"]["mode"] == "auto"
    # video: listed, removed with its start time
    (folder / "video.mp4").write_bytes(b"0" * 2048)
    ini = folder / "song.ini"
    ini.write_text(ini.read_text(encoding="utf-8") + "video_start_time = 120\n", encoding="utf-8")
    assert c.get(base).json()["video"]["file"] == "video.mp4"
    d = c.delete(f"{base}/video").json()
    assert d["video"] is None and "video_start_time" not in ini.read_text(encoding="utf-8")
    # online covers: search goes through covers.search; picked covers only from cover hosts
    monkeypatch.setattr(covers, "search", lambda a, t, al: {"results": [{"q": [a, t, al]}], "errors": []})
    assert c.get("/api/covers/search", params={"artist": "Muse", "title": "Uprising"}).json()["results"] == [
        {"q": ["Muse", "Uprising", ""]}]
    assert c.get("/api/covers/search").status_code == 400
    assert c.post(f"{base}/art/url", json={"url": "https://example.com/a.jpg"}).status_code == 400
    monkeypatch.setattr(covers, "fetch", lambda url, n: img((0, 0, 250), "JPEG"))
    d = c.post(f"{base}/art/url", json={"url": "https://is1-ssl.mzstatic.com/a/1200x1200bb.jpg"}).json()
    assert d["art"].startswith("album.jpg") and d["background"]["file"].startswith("background.jpg")
    # export carries the visuals, and a changed background marks the export stale
    c.post(f"{base}/export", json={})
    dest = env.cfg["export_dir"] / "Muse - Uprising"
    assert (dest / "album.jpg").exists() and (dest / "background.jpg").exists()
    assert not (dest / ".riffchart").exists()
    c.post(f"{base}/background", json={"mode": "none"})
    assert c.get(base).json()["export_stale"]


def test_highway_endpoints(env):
    import io
    from PIL import Image

    def img(color, size=(200, 200)):
        b = io.BytesIO()
        Image.new("RGB", size, color).save(b, "PNG")
        return b.getvalue()
    c, folder = env.client, _charted(env)
    base = "/api/songs/Muse - Uprising"
    assert c.get(base).json()["highway"]["mode"] == "riffchart"
    d = c.post(f"{base}/highway", json={"mode": "cover"}).json()
    assert d["highway"]["mode"] == "cover" and d["highway"]["file"].startswith("highway.png")
    assert c.post(f"{base}/highway", json={"mode": "own"}).status_code == 400
    d = c.post(f"{base}/highway/file", files={"file": ("h.png", img((9, 9, 9), (512, 1024)))}).json()
    assert d["highway"]["mode"] == "own"
    c.post(f"{base}/export", json={})
    assert (env.cfg["export_dir"] / "Muse - Uprising" / "highway.png").exists()
    d = c.post(f"{base}/highway", json={"mode": "none"}).json()
    assert d["highway"]["file"] == "" and d["export_stale"]


def test_highway_designer_endpoints(env):
    import io
    from PIL import Image
    b = io.BytesIO()
    Image.new("RGB", (64, 64), (200, 30, 30)).save(b, "PNG")
    c, folder = env.client, _charted(env)
    other = _charted(env, "Band - Other")
    blank = c.get("/api/highways").json()
    assert blank["designs"] == [] and blank["default"] == "riffchart" and blank["clone_hero"]
    design = json.dumps(dict(blank["blank"], name="Red Room", rail="#ff2020"))
    r = c.post("/api/highways/render", data={"design": design}, files={"pattern": ("p.png", b.getvalue())})
    assert r.status_code == 200 and r.content[:4] == b"\x89PNG"
    assert c.post("/api/highways/render", data={"design": '{"rail": "red"}'}).status_code == 400
    saved = c.post("/api/highways", data={"design": design}, files={"pattern": ("p.png", b.getvalue())}).json()
    hid = saved["design"]["id"]
    assert hid == "red-room" and saved["design"]["has_pattern"] and saved["updated"] == []
    assert c.get(f"/api/highways/{hid}/image").content[:4] == b"\x89PNG"
    # one song, then exported; a later edit of the design redraws it and updates the up-to-date export
    d = c.post("/api/songs/Muse - Uprising/highway", json={"mode": "design", "design": hid}).json()
    assert d["highway"]["mode"] == "design" and d["highway"]["design"]["name"] == "Red Room"
    assert c.post("/api/songs/Muse - Uprising/highway", json={"mode": "design", "design": "nope"}).status_code == 404
    c.post("/api/songs/Muse - Uprising/export", json={})
    exp = env.cfg["export_dir"] / "Muse - Uprising" / "highway.png"
    before = exp.read_bytes()
    time.sleep(1.1)
    upd = c.post("/api/highways", data={"design": json.dumps(dict(saved["design"], rail="#20ff20")),
                                        "pattern_from": hid}).json()
    assert upd["updated"] == ["Muse - Uprising"] and upd["exported"] == ["Muse - Uprising"]
    assert upd["design"]["has_pattern"] and exp.read_bytes() != before
    # all songs (own highways left alone) + library default
    c.post("/api/songs/Band - Other/highway/file", files={"file": ("h.png", b.getvalue())})
    r = c.post("/api/highways/apply", json={"highway": f"design:{hid}", "songs": "all", "default": True}).json()
    assert r["updated"] == ["Muse - Uprising"] and r["skipped_own"] == ["Band - Other"]
    assert r["default"] == f"design:{hid}" and c.get("/api/highways").json()["default"] == f"design:{hid}"
    assert c.post("/api/highways/apply", json={"highway": "rainbow", "songs": "all"}).status_code == 400
    # install in Clone Hero (Custom/Highways next to its settings.ini)
    inst = c.post(f"/api/highways/{hid}/install").json()["installed"]
    assert Path(inst).parent == env.tmp / "Custom" / "Highways" and Path(inst).name == "RiffChart - Red Room.png"
    assert c.delete(f"/api/highways/{hid}").json()["default"] == "riffchart"
    assert c.get("/api/songs/Muse - Uprising").json()["highway"]["design"]["name"] == "Red Room"
    assert other.exists()


def test_save_keeps_chart_offset_and_shifted_tempo(env):
    """A chart's own [Song] Offset survives an editor save, and a sync nudge (new lead-in tempo) is written."""
    c, folder = env.client, _charted(env)
    (folder / "notes.chart").write_text(CHART.replace("Offset = 0", "Offset = 0.25"), encoding="utf-8")
    d = c.get("/api/songs/Muse - Uprising/chart").json()
    assert abs(d["offset_s"] - 0.25) < 1e-9
    d["tempo"] = [[0, 100.0], [192, 120.0]]                       # what the editor sends after a nudge
    assert c.post("/api/songs/Muse - Uprising/chart", json=d).status_code == 200
    text = (folder / "notes.chart").read_text(encoding="utf-8")
    assert "Offset = 0.25" in text and "0 = B 100000" in text and "192 = B 120000" in text


def test_preview_start(env):
    c, folder = env.client, _charted(env)
    assert c.get("/api/songs/Muse - Uprising/chart").json()["preview_s"] is None
    r = c.post("/api/songs/Muse - Uprising/preview", json={"start_s": 12.3456})
    assert r.status_code == 200 and r.json()["preview_s"] == 12.346
    assert "preview_start_time = 12346" in (folder / "song.ini").read_text(encoding="utf-8")
    assert c.get("/api/songs/Muse - Uprising/chart").json()["preview_s"] == 12.346
    for bad in ({"start_s": -1}, {"start_s": 999}, {"start_s": "soon"}, {}):
        assert c.post("/api/songs/Muse - Uprising/preview", json=bad).status_code == 400


def test_history_list_and_open(env):
    c, folder = env.client, _charted(env)
    base = "/api/songs/Muse - Uprising"
    assert c.get(f"{base}/history").json() == []
    d = c.get(f"{base}/chart").json()
    d["tracks"]["expert"]["notes"] = d["tracks"]["expert"]["notes"][:1]
    assert c.post(f"{base}/chart", json=d).status_code == 200            # keeps the 3-note version
    d["tracks"]["expert"]["notes"] = []
    d["tracks"]["hard"]["notes"] = [[0, 2, 0, 0], [192, 4, 0, 0]]
    assert c.post(f"{base}/chart", json=d).status_code == 200            # same second: still two versions
    h = c.get(f"{base}/history").json()
    assert [v["notes"]["expert"] for v in h] == [1, 3] and all(v["reason"] == "save" for v in h)
    assert h[0]["replaced"] >= h[1]["replaced"] and h[1]["made"] > 0
    old = c.get(f"{base}/history/{h[1]['name']}").json()
    assert len(old["tracks"]["expert"]["notes"]) == 3 and old["resolution"] == 192 and old["tempo"]
    for bad in ("nope-notes.chart", "..%2Fsong.ini", "../../song.ini"):
        assert c.get(f"{base}/history/{bad}").status_code == 404
    (folder / ".riffchart" / "history" / "20200101-000000-rechart-notes.chart").write_text("junk", encoding="utf-8")
    h = c.get(f"{base}/history").json()
    assert h[-1]["reason"] == "rechart"


def test_convert_video_endpoint(env):
    from test_visuals import _encode
    c, folder = env.client, _charted(env)
    base = "/api/songs/Muse - Uprising"
    assert c.post(f"{base}/video/convert").status_code == 404                # no video yet
    _encode(folder / "video.mp4", audio=False, secs=1.0)
    v = c.get(base).json()["video"]
    assert v["codec"] == "mpeg4" and not v["ok"]
    c.post(f"{base}/video/convert")
    for _ in range(100):
        v = c.get(base).json()["video"]
        if not v.get("converting"):
            break
        time.sleep(0.05)
    assert v["file"] == "video.webm" and v["codec"] == "vp8" and v["ok"] and "converting" not in v


def test_first_run_setup(env, monkeypatch):
    from riffchart.ui import setup
    monkeypatch.setattr(clonehero, "is_running", lambda: False)
    conf = env.tmp / "ui.yaml"
    (env.tmp / "ui.local.yaml").write_text("# my notes\nport: 9000\nexport_dir: old\n", encoding="utf-8")
    env.cfg["_config"] = conf
    st = env.cfg["clone_hero_settings"]
    st.write_text(st.read_text(encoding="utf-8") + "\n[custom]\nsong_highway = 0\nsong_videos = 1\n", encoding="utf-8")
    s = env.client.get("/api/setup").json()
    assert s["clone_hero"]["found"] and s["clone_hero"]["folders"] == ["C:\\songs"]
    assert s["clone_hero"]["options"] == {"song_videos": 1, "song_backgrounds": None, "song_highway": 0}
    songs = env.tmp / "Finished songs"
    body = {"export_dir": str(songs), "clone_hero_settings": str(st), "protected": [str(env.lib_ro)], "register": True,
            "options": {"song_videos": True, "song_backgrounds": True, "song_highway": True}}
    r = env.client.post("/api/setup", json=body).json()
    assert r["registered"] and r["options_changed"] and r["done"] and songs.is_dir()
    text = st.read_text(encoding="utf-8")
    assert "song_highway = 1" in text and "song_backgrounds" not in text.split("[custom]")[0]
    assert str(songs.resolve()) in text and (env.tmp / "settings.ini.bak-riffchart").exists()
    local = (env.tmp / "ui.local.yaml").read_text(encoding="utf-8")
    assert local.startswith("# my notes\nport: 9000\n") and "export_dir: old" not in local
    loaded = setup.read_local(conf)
    assert loaded["export_dir"] == str(songs) and loaded["protected"] == [str(env.lib_ro)] and loaded["setup_done"]
    assert env.client.get("/api/status").json()["export_dir"] == str(songs)        # used right away
    # refused: songs folder inside a protected one; Clone Hero running while its settings would change
    bad = dict(body, export_dir=str(env.lib_ro / "x"))
    assert env.client.post("/api/setup", json=bad).status_code == 400
    assert env.client.post("/api/setup", json=dict(body, export_dir="relative")).status_code == 400
    monkeypatch.setattr(clonehero, "is_running", lambda: True)
    assert env.client.post("/api/setup", json=dict(body, options={"song_videos": False})).status_code == 409
    assert env.client.post("/api/setup", json=dict(body, register=False, options={})).status_code == 200


def test_set_ch_options_only_touches_custom(tmp_path):
    from riffchart.ui.setup import read_options, set_options
    st = tmp_path / "settings.ini"
    st.write_text("[game]\nsong_videos = 7\n\n[video]\nx = 1\n", encoding="utf-8")
    assert set_options(st, {"song_videos": 1, "nonsense": 1}) and not set_options(st, {"song_videos": 1})
    t = st.read_text(encoding="utf-8")
    assert "[game]\nsong_videos = 7" in t and t.rstrip().endswith("[custom]\nsong_videos = 1") and "nonsense" not in t
    assert read_options(st)["song_videos"] == 1


def test_find_clone_hero_exe(tmp_path, monkeypatch, env):
    ch = tmp_path / "Somewhere" / "Clone Hero"
    (ch / "Clone Hero_Data").mkdir(parents=True)
    (ch / "Clone Hero.exe").write_bytes(b"MZ")
    assert clonehero.exe_path(ch) == ch / "Clone Hero.exe"                 # the folder is enough
    assert clonehero.exe_path(ch / "Clone Hero.exe") == ch / "Clone Hero.exe"
    assert clonehero.exe_path(tmp_path) is None and clonehero.exe_path("") is None
    # Clone Hero's own log says where it ran from
    logs = tmp_path / "home" / "AppData" / "LocalLow" / "srylain Inc_" / "Clone Hero"
    logs.mkdir(parents=True)
    (logs / "Player.log").write_text(f"[Subsystems] Discovering subsystems at path {ch.as_posix()}/Clone Hero_Data/UnitySubsystems\n",
                                     encoding="utf-8")
    monkeypatch.setattr(clonehero.Path, "home", classmethod(lambda cls: tmp_path / "home"))
    monkeypatch.delenv("CLONE_HERO_EXE", raising=False)
    assert clonehero.from_player_log() == ch / "Clone Hero.exe"
    # chosen in Setup: saved and used
    monkeypatch.setattr(clonehero, "is_running", lambda: False)
    env.cfg["_config"] = env.tmp / "ui.yaml"
    body = {"export_dir": str(env.tmp / "songs"), "clone_hero_settings": str(env.cfg["clone_hero_settings"]),
            "clone_hero_exe": str(ch), "protected": []}
    r = env.client.post("/api/setup", json=body)
    assert r.status_code == 200 and r.json()["clone_hero"]["exe"] == str(ch / "Clone Hero.exe")
    assert env.client.post("/api/setup", json=dict(body, clone_hero_exe=str(tmp_path / "nope"))).status_code == 400
