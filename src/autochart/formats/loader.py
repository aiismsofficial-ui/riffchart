"""Locate and load a song folder's chart into the internal format."""
from __future__ import annotations

from pathlib import Path

from .chart_parser import parse_chart
from .midi_parser import parse_midi
from .song import Song
from .textio import parse_ini

# YARG/CH prefer notes.mid over notes.chart when both exist.
CHART_FILES = ("notes.mid", "notes.midi", "notes.chart")


def find_chart_file(folder: Path) -> Path | None:
    names = {p.name.lower(): p for p in folder.iterdir() if p.is_file()}
    for n in CHART_FILES:
        if n in names:
            return names[n]
    return None


def load_song(folder: str | Path) -> tuple[Song, dict[str, str], Path]:
    folder = Path(folder)
    chart = find_chart_file(folder)
    if chart is None:
        raise FileNotFoundError(f"no notes.mid/notes.chart in {folder}")
    ini_path = next((p for p in folder.iterdir() if p.name.lower() == "song.ini"), None)
    ini = parse_ini(ini_path) if ini_path else {}
    song = parse_midi(chart, ini) if chart.suffix.lower() in (".mid", ".midi") else parse_chart(chart, ini)
    return song, ini, chart
