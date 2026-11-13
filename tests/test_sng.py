"""The .sng writer against the spec (github.com/mdsitton/SngFileFormat) and the Moonscraper finder."""
import struct

from riffchart.product.sng import find_moonscraper, moonscraper_exe, read_sng, song_files, write_sng


def _song(tmp_path):
    f = tmp_path / "Band - Song"
    (f / ".riffchart").mkdir(parents=True)
    (f / "notes.chart").write_text("[Song]\n{\n}\n", encoding="utf-8")
    (f / "song.ogg").write_bytes(bytes(range(256)) * 40)              # > 16 and > 256 bytes: mask wraps
    (f / "album.jpg").write_bytes(b"\xff\xd8jpeg")
    (f / "song.ini").write_text("[song]\nname = Song\nartist = Band\n", encoding="utf-8")
    (f / "notes.mid.bak").write_bytes(b"x")
    (f / ".riffchart" / "lyrics.json").write_text("{}", encoding="utf-8")
    return f


def test_sng_round_trip_and_layout(tmp_path):
    f = _song(tmp_path)
    assert [x.name for x in song_files(f)] == ["album.jpg", "notes.chart", "song.ogg"]   # no ini, backups, .riffchart
    out = write_sng(f, tmp_path / "x.sng", {"name": "Song", "artist": "Band", "album": ""})
    b = out.read_bytes()
    assert b[:6] == b"SNGPKG" and struct.unpack_from("<I", b, 6)[0] == 1
    meta, files = read_sng(out)
    assert meta == {"name": "Song", "artist": "Band"}                  # empty fields left out
    assert files == {p.name: p.read_bytes() for p in song_files(f)}
    # independent check of one file straight from the bytes: absolute offset, per-file mask index
    mask = b[10:26]
    mlen = struct.unpack_from("<Q", b, 26)[0]
    p = 26 + 8 + mlen + 16                                              # first FileMeta
    nl = b[p]
    size, off = struct.unpack_from("<QQ", b, p + 1 + nl)
    raw = bytes(x ^ mask[i % 16] ^ (i & 0xFF) for i, x in enumerate(b[off:off + size]))
    assert raw == (f / b[p + 1:p + 1 + nl].decode()).read_bytes()
    assert len(b) == 26 + 8 + mlen + 8 + struct.unpack_from("<Q", b, 26 + 8 + mlen)[0] + 8 + sum(len(v) for v in files.values())


def test_sng_needs_a_chart(tmp_path):
    f = tmp_path / "empty"
    f.mkdir()
    (f / "song.ogg").write_bytes(b"x")
    try:
        write_sng(f, tmp_path / "e.sng", {})
        assert False
    except ValueError:
        pass


def test_moonscraper_paths(tmp_path, monkeypatch):
    d = tmp_path / "Moonscraper Chart Editor v1.5"
    d.mkdir()
    exe = d / "Moonscraper Chart Editor.exe"
    exe.write_bytes(b"MZ")
    assert moonscraper_exe(d) == exe and moonscraper_exe(exe) == exe and moonscraper_exe(tmp_path / "no") is None
    monkeypatch.setenv("MOONSCRAPER_EXE", str(d))
    assert find_moonscraper(None) == exe
