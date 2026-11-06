# RiffChart

AI autocharter for **Clone Hero** and **YARG**: give it a song, get a playable 5-fret guitar chart in
all four difficulties (Expert, Hard, Medium, Easy), with star power, HOPOs, taps and sustains. It comes
with a local web app to manage your songs and a chart editor to review and fix the result.

Everything runs on your own PC. Your songs and charts are never uploaded anywhere.

**Version 1.0** for Windows 10 / 11. Download, double-click `install.bat`, and chart your first song a few
minutes later. A graphics card is optional.

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
- **Difficulty rating:** each song's guitar difficulty (0-6) is estimated from its Expert chart the way
  human charters rate songs.
- **Takes what you have:** mp3, m4a, ogg, opus, flac, wav, webm and music videos (mp4, mkv). Names left by
  downloads and converter sites ("(Official Video)", "y2mate.com - ...", "Artist - Topic") are tidied.

### The app

- **First run:** a short setup finds Clone Hero, picks the folder finished songs go to, protects your own
  chart folders from any changes, and switches on Clone Hero's song videos, backgrounds and highways.
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

Requirements: Windows 10/11, 16 GB RAM recommended, about 3 GB of disk space, and an internet connection
for the install. A graphics card is optional: RiffChart runs on the CPU (a 3-minute song took about 60 s on
a desktop Intel i7-14700KF, and a few minutes on a laptop without a graphics card; plug the laptop in, it's
much faster). It's about twice as fast on an NVIDIA GTX 16 / RTX 20-series card or newer with 6 GB+
(~40 s on an RTX 5090).
AMD Radeon RX 9000 / 7900 cards may work with AMD's ROCm build of PyTorch for Windows (untested).
`device: auto | cuda | cpu` in `configs/product.yaml` picks where it runs (auto: the GPU if there is one).

**Install:** download this repository (Code > Download ZIP), unzip it somewhere like `C:\RiffChart`, and
double-click **install.bat**. It needs no admin rights and installs nothing system-wide. Everything goes in that
folder:
- **uv:** the tool that installs Python and the app's packages.
- **Python 3.12.**
- **PyTorch:** for an NVIDIA card if you have one with a recent driver, otherwise the CPU version. It asks.
- **The model files.**
- **Shortcuts:** RiffChart on the Desktop and in the Start menu.

Expect about 3 GB with an NVIDIA card, or 1.5 GB for the CPU version. Run install.bat again at any time to repair
or update; finished steps are skipped. To uninstall, run **uninstall.bat**. It removes the app and its
models but keeps your charted songs unless you choose otherwise; then delete the folder.

**Start:** the RiffChart shortcut, or `RiffChart.bat`. It opens the app in your browser; close the RiffChart
window to stop it.

For development:

```bash
python -m pip install --user uv
uv sync --extra dev --extra cuda --python 3.12      # or --extra cpu
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

From the shortcut, or from the command line:

```bash
.venv/Scripts/riffchart-ui                                   # the app, opens http://127.0.0.1:8765/
.venv/Scripts/riffchart "Artist - Song.mp3" --out "D:/Clone Hero/Songs/riffchart"
.venv/Scripts/riffchart "D:/music/some album" --out <folder> --style density=-0.3,chords=low
```

The command line takes mp3, m4a, ogg, opus, webm, flac, wav or mp4 files (and a few more), or folders of them, and skips songs that are
already charted (`--overwrite` to redo). Run `riffchart --help` for all options.

## Models

The installer downloads RiffChart's models from Hugging Face,
[riffchart/riffchart-models](https://huggingface.co/riffchart/riffchart-models), and checks each file's
checksum. There are four files, about 300 MB together:
- the chart model;
- a beat tracker, fine-tuned from [Beat This](https://github.com/CPJKU/beat_this);
- the chart model's input statistics;
- the difficulty reducer.

The list is in `configs/models.yaml`, and `scripts/get_models.py` fetches or checks them. Stem separation uses
[Demucs](https://github.com/facebookresearch/demucs) (`htdemucs_6s`), which is downloaded from Meta.

You can train your own from a library of charts with matching audio. The pipeline (data audit,
artist-disjoint splits, stem separation, features, training, evaluation) is in `src/riffchart/` and
described in [DESIGN.md](DESIGN.md). Your charts and audio stay on your PC: `data/` is gitignored and
the chart library is only ever read.

## How it works

See [DESIGN.md](DESIGN.md) for the pipeline, the model and the post-processing.

## Licence

The code is GPL-3.0 (see `LICENSE`). The model files are CC BY-NC 4.0: free to use and share with credit,
not for commercial use. The beat tracker builds on Beat This (MIT).

The charts you make are for your own play. The songs themselves are copyrighted, so think twice before
sharing song folders that contain the audio.

## Licence

GPL-3.0-or-later (see [LICENSE](LICENSE)). The licence covers this code. Models, when released, will
come with their own terms.

## Credits

Built on [PyTorch](https://pytorch.org/), [Demucs](https://github.com/facebookresearch/demucs),
[Beat This!](https://github.com/CPJKU/beat_this),
[FastAPI](https://fastapi.tiangolo.com/) and [mutagen](https://github.com/quodlibet/mutagen).
Chart formats follow the conventions documented by the Clone Hero and Moonscraper communities.
