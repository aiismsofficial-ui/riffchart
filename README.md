# autochart

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
  Export to your Clone Hero songs folder, or open a song straight in Clone Hero.
- **Editor:** vertical highway, one difficulty or all four side by side, synced playback (with speed
  control, note clap and metronome), and full editing: add, move and delete notes, sustains, HOPO / tap,
  star power, solos, sections, undo and redo.
- **Live playability check:** problems such as impossible chords, notes too fast for the difficulty or
  sustains running into the next note are marked on the highway as you edit.
- **Re-chart a section:** select a few bars (a solo, a riff), and let the AI chart just that part again.
  Try several takes, listen to each with the section looping, and keep the one you like.

## Setup (Windows)

Requirements: Python 3.12, an NVIDIA GPU with a recent driver (CUDA 13 wheels), [uv](https://docs.astral.sh/uv/).

```bash
python -m pip install --user uv
uv sync --extra dev --python 3.12
.venv/Scripts/python -m pytest -q
```

Settings live in `configs/*.yaml`. Put anything specific to your PC (your folders) in a matching
`configs/<name>.local.yaml`, which is gitignored and merged over the tracked file. For example
`configs/ui.local.yaml`:

```yaml
export_dir: D:/Clone Hero/Songs/autochart
protected:                      # your own chart folders: the app never writes inside these
  - D:/Clone Hero/Songs
```

## Usage

Once models are available (see below):

```bash
.venv/Scripts/autochart-ui                                   # the app, opens http://127.0.0.1:8765/
.venv/Scripts/autochart "Artist - Song.mp3" --out "D:/Clone Hero/Songs/autochart"
.venv/Scripts/autochart "D:/music/some album" --out <folder> --style density=-0.3,chords=low
```

The command line takes mp3, ogg, opus, flac or wav files, or folders of them, and skips songs that are
already charted (`--overwrite` to redo). Run `autochart --help` for all options.

## Models

autochart needs two trained models: the chart model (`configs/product.yaml: chart_ckpt`) and a
fine-tuned beat tracker. They are not in this repository and have not been released yet. Stem
separation uses [Demucs](https://github.com/facebookresearch/demucs) (`htdemucs_6s`), which downloads
its own weights on first use.

You can train your own from a library of charts with matching audio. The pipeline (data audit,
artist-disjoint splits, stem separation, features, training, evaluation) is in `src/autochart/` and
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
