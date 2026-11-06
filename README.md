# RiffChart

AI autocharter for **Clone Hero** and **YARG**: give it a song, get a playable 5-fret guitar chart in
all four difficulties (Expert, Hard, Medium, Easy), with star power, HOPOs, taps and sustains. It comes
with a local web app to manage your songs and a chart editor to review and fix the result.

Everything runs on your own PC. Nothing is uploaded anywhere.

> **Status: in development.** The code is here; the trained models are **not published yet**, so the
> app cannot chart songs out of the box. See [Models](#models).

## What it does

- **Charts a song in about 30 seconds** on a modern NVIDIA GPU (3-4 minute song): separates the guitar
  from the mix, finds the beats and builds a tempo map, writes the notes with a transformer model, then
  cleans the chart up so it is playable.
- **Four difficulties that step down properly:** Hard, Medium and Easy are generated with the song and
  kept below the next harder difficulty in every 16-beat stretch.
- **Style controls:** sparser or denser, fewer or more chords, and what to play along with (the guitar,
  other instruments where the guitar is quiet, or the whole song's melody, for songs with little guitar).
- **Clone Hero-ready output:** `notes.chart`, `song.ini` (with an estimated difficulty and preview
  start), `song.ogg` and a separate `guitar.ogg`, so the guitar mutes when you miss.

### The app

- **Library:** drop in audio files or folders; songs chart in the background with a progress bar.
  Fix a song's details (title, artist, album, genre, year, difficulty, cover) and update the copy in
  Clone Hero in one click; covers embedded in your audio files are picked up automatically, or find one
  online (opt-in: only the artist and title are sent).
- **Backgrounds:** a background made from the cover (blurred and darkened) or your own image, and charting a
  music video file keeps the video to play behind the highway (converted to a format Clone Hero plays when
  needed).
- **Highways:** every song gets a RiffChart highway, one coloured from its cover, or your own. The
  highway designer (colours, edge glow, a pattern image, your own text) shows a live 3D preview; use a
  design on one song, all songs or new songs, or install it in Clone Hero as your everyday highway.
  Export to your Clone Hero songs folder, or open a song straight in Clone Hero.
- **Editor:** vertical highway, one difficulty or all four side by side, synced playback (with speed
  control, note clap and metronome), and full editing: add, move and delete notes, sustains, HOPO / tap,
  star power, solos, sections, undo and redo. A sync nudge moves the whole chart earlier or later
  against the audio in 5 ms steps, the song-list preview can be set at the playhead, and every earlier
  version of a chart can be looked at and restored.
- **Live playability check:** problems such as impossible chords, notes too fast for the difficulty or
  sustains running into the next note are marked on the highway as you edit.
- **Re-chart a section:** select a few bars (a solo, a riff), and let the AI chart just that part again.
  Try several takes, listen to each with the section looping, and keep the one you like.

## Setup (Windows)

Requirements: Windows 10/11, Python 3.12, [uv](https://docs.astral.sh/uv/), 16 GB RAM recommended, ~6 GB disk.
A graphics card is optional: RiffChart runs on the CPU (a 4-minute song took ~90 s on an Intel i7-14700KF)
and is about twice as fast on an NVIDIA GTX 16 / RTX 20-series card or newer with 6 GB+ (~40 s on an RTX 5090).
AMD Radeon RX 9000 / 7900 cards may work with AMD's ROCm build of PyTorch for Windows (untested).
`device: auto | cuda | cpu` in `configs/product.yaml` picks where it runs (auto: the GPU if there is one).

```bash
python -m pip install --user uv
uv sync --extra dev --python 3.12
.venv/Scripts/python -m pytest -q
```

The first time the app starts it asks where finished songs should go, which of your chart folders it must
never change, and whether to switch on Clone Hero's song videos, backgrounds and highways (Settings in the
app's header changes these later). It saves the answers to `configs/ui.local.yaml`.

Settings live in `configs/*.yaml`. Put anything specific to your PC (your folders) in a matching
`configs/<name>.local.yaml`, which is gitignored and merged over the tracked file. For example
`configs/ui.local.yaml`:

```yaml
export_dir: D:/Clone Hero/Songs/riffchart
protected:                      # your own chart folders: the app never writes inside these
  - D:/Clone Hero/Songs
```

## Usage

Once models are available (see below):

```bash
.venv/Scripts/riffchart-ui                                   # the app, opens http://127.0.0.1:8765/
.venv/Scripts/riffchart "Artist - Song.mp3" --out "D:/Clone Hero/Songs/riffchart"
.venv/Scripts/riffchart "D:/music/some album" --out <folder> --style density=-0.3,chords=low
```

The command line takes mp3, m4a, ogg, opus, webm, flac, wav or mp4 files (and a few more), or folders of them, and skips songs that are
already charted (`--overwrite` to redo). Run `riffchart --help` for all options.

## Models

RiffChart needs two trained models: the chart model (`configs/product.yaml: chart_ckpt`) and a
fine-tuned beat tracker. They are not in this repository and have not been released yet. Stem
separation uses [Demucs](https://github.com/facebookresearch/demucs) (`htdemucs_6s`), which downloads
its own weights on first use.

You can train your own from a library of charts with matching audio. The pipeline (data audit,
artist-disjoint splits, stem separation, features, training, evaluation) is in `src/riffchart/` and
described in [DESIGN.md](DESIGN.md). Your charts and audio stay on your PC: `data/` is gitignored and
the chart library is only ever read.

## How it works

See [DESIGN.md](DESIGN.md) for the pipeline, the model and the post-processing.

## Licence

GPL-3.0-or-later (see [LICENSE](LICENSE)). The licence covers this code. Models, when released, will
come with their own terms.

## Credits

Built on [PyTorch](https://pytorch.org/), [Demucs](https://github.com/facebookresearch/demucs),
[Beat This!](https://github.com/CPJKU/beat_this),
[FastAPI](https://fastapi.tiangolo.com/) and [mutagen](https://github.com/quodlibet/mutagen).
Chart formats follow the conventions documented by the Clone Hero and Moonscraper communities.
