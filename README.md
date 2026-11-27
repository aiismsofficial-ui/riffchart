# RiffChart: free AI charts for Clone Hero and YARG

Want to play your own songs in **Clone Hero** or **YARG**, the popular free rhythm games where you play along
on a guitar controller as notes scroll down a highway? Every song needs a *chart*, the notes you play, and
making one by hand takes hours.
RiffChart makes it for you: drop in any song (an mp3, a YouTube download, a music video) and a minute or
two later it's a playable custom song with **guitar and bass** on **Easy, Medium, Hard and Expert**, plus
lyrics, its own highway and background. No charting experience needed.

Everything runs on your own PC. Your songs and charts are never uploaded anywhere.

**[Download RiffChart 1.2](https://github.com/aiismsofficial-ui/riffchart/releases/latest)** for Windows 10 / 11:
run **RiffChart-Setup.exe** (or unzip the ZIP and double-click `install.bat`), and play your first song a few
minutes later. A graphics card is optional. Community, help and showcases: **r/RiffChart** on Reddit.

## New to this?

- **Clone Hero** ([clonehero.net](https://clonehero.net)) and **YARG** ([yarg.in](https://yarg.in)) are free
  rhythm games for PC, the most popular way to play guitar rhythm games today. You play along with a guitar controller (or the keyboard) as notes scroll down a
  highway. They come with no songs of their own: you add songs ("custom songs" or "charts"), each a folder
  with the music and a chart file.
- **Charting** a song by hand, in an editor such as Moonscraper, means placing every note in time with
  the music. RiffChart's AI listens to the song and does it in a minute or two: guitar, bass, star power,
  the tempo, section names and lyrics.
- **Where the songs go:** RiffChart finds Clone Hero on your PC and copies finished songs straight into
  its songs folder; in the game, Settings > Scan Songs. For YARG, add the same folder in its song settings.
- **How good is it?** A good hand-made chart still beats it, and if a song already has one, play that.
  RiffChart is for the songs nobody has charted yet, and as a first draft you can fix in its built-in editor.

RiffChart is an independent project, not affiliated with Clone Hero or YARG. It doesn't download music:
you chart songs you have.

## What it does

- **Charts a song in about a minute** on a modern NVIDIA GPU (3-4 minute song; about 30 s for the guitar alone, plus the bass and keys): separates the guitar
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
- **Lyrics:** charts get the song's lyrics, timed word by word to the singing, shown by Clone Hero and YARG
  (looked up on lrclib.net, a free lyrics database: only the artist, title and length are sent; switch it off
  in Settings). When a song isn't there, or you'd rather nothing is sent, the singing is written down by
  speech recognition (Whisper) running on your own PC. Songs already in your library get them from Song
  details.
- **Separate audio per instrument:** guitar, bass, drums, vocals and keys each get their own track, so you
  can turn instruments up or down in Clone Hero and YARG. Songs charted with an older version can be
  **brought up to date** from the library (own instrument tracks, a bass chart, lyrics, sharper backgrounds)
  without re-charting them.
- **Bass charts:** every song also gets a 5-fret bass part (all four difficulties), charted from the bass
  track by the same model. Against human bass charts on songs it never saw, its notes land on the beat about
  as well as the guitar charts do (76-79% matched on Expert/Hard, vs 75% for copying the guitar part); which
  fret each note gets is rougher, so the editor has a **Bass** tab to fix it with the same tools. A dedicated
  bass model is planned.
- **Keys charts (experimental, off by default):** the same from the piano / keyboard track, on songs that have
  one loud enough. Rougher than guitar and bass (about 54% of notes matched on Expert, and busier than human
  keys parts), so treat them as a first draft for the editor's **Keys** tab. Turn them on in Settings.
- **Pitched vocals for YARG:** a song's lyrics can become sung notes at the pitch the singer hits, so YARG can
  score your singing.
- **Report a spot:** press F in the editor where a chart goes wrong and type what's wrong; the library's
  Feedback file gathers them into one small file (no audio) you can attach to an issue, so the model can learn
  from it.
- **Batch actions:** select several songs (or all) to export, re-chart, bring up to date, or give them a
  highway or an animated background in one go.
- **Share or hand-edit:** download any song as a single `.sng` file (Clone Hero and YARG read these), or open
  its chart in Moonscraper Chart Editor.
- **Backgrounds:** a background made from the cover (blurred and darkened) or your own image, and charting a
  music video file keeps the video to play behind the highway (converted to a format Clone Hero plays when
  needed). Songs without a music video can get an **animated background**: 16 presets (embers, bokeh,
  aurora, starfield, stage lights, plasma, rain, thunderstorm, synthwave sunset, smoke, equalizer, nebula,
  laser show, hyper tunnel, fireflies, soundwave) drawn over
  the song's own background, or installed in Clone Hero for every song.
- **Highways:** every song gets a RiffChart highway, one coloured from its cover, or your own. The
  designer has colours, edge styles (neon tube, double, chrome) and glow, lane styles and fret-coloured lanes,
  centre glow, dark sides, a pattern image with blend modes and your own text, with a live 3D preview.
  **Animated highways:** 16 looping effects (sparks, warp, rain, edge lightning, edge fire, synthwave grid,
  plasma, aurora, pulse, colour cycle, sheen, chevrons, digital rain, ripples, lava, lane chase), installed in Clone Hero as your everyday video highway (Clone
  Hero only animates its own highways; on a single song a design shows still). Use a design on one song, all
  songs or new songs, or install it in Clone Hero.
  Export to your Clone Hero songs folder, or open a song straight in Clone Hero.
- **Editor:** vertical highway, one difficulty or all four side by side, synced playback (with speed
  control, note clap and metronome), zoom set in Clone Hero's own **Note Speed** (7 = what the game shows by
  default), a **Game view** (G) that plays the chart on a tilted highway the way Clone Hero shows it, and full
  editing: add, move and delete notes, sustains, HOPO / tap,
  star power, solos, sections, undo and redo. A sync nudge moves the whole chart earlier or later
  against the audio in 5 ms steps, the song-list preview can be set at the playhead, and every earlier
  version of a chart can be looked at and restored. Sections (Intro, Verse 1, Chorus 1, Guitar Solo…) are named automatically from the audio,
  and can be redone or renamed in the editor. A **tempo map editor** lets you fix BPM and time signatures, tap a tempo, or drag beat lines onto the
  beats in the waveform. **Lyrics** can be fixed in the editor too: change the words, drag them on the
  waveform, or tap along to time them. After editing Expert, **Easier from Expert** makes
  Hard, Medium and Easy again from it in one click (undoable).
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

**Install:** download **RiffChart-Setup.exe** from the latest release (Releases, on the right of the GitHub
page) and run it. Windows shows "Windows protected your PC" for new programs that aren't code-signed: click
**More info**, then **Run anyway**. Setup needs no admin rights. It copies the app into your own Programs folder,
then a window downloads what it needs (several minutes; an NVIDIA card with a recent driver is used
automatically, otherwise the CPU version):
- **uv:** the tool that installs Python and the app's packages.
- **Python 3.12** and **PyTorch**.
- **The model files.**

Expect about 3 GB with an NVIDIA card, or 1.5 GB for the CPU version. RiffChart then appears in the Start menu
(and on the desktop if you ticked it) and in Windows' **Installed apps**, where you can uninstall it: that
removes the app, Python and the models but keeps your charted songs; the copies in your Clone Hero songs folder
are never touched.

**Update to a new version:** the library shows a note when one is out. Run the new RiffChart-Setup.exe: it
installs over the old one and keeps your songs, highway designs and settings. Coming from **RiffChart 1.0**
(the ZIP you unzipped yourself)? On Setup's folder page click **Browse** and choose that RiffChart folder, so
your library comes along (or unzip the new ZIP over it and run install.bat). Then **Bring up to date** in the
library adds what's new to the songs you charted before.

**Without the Setup:** download the release's ZIP instead, unzip it somewhere like `C:\RiffChart` (right-click
the ZIP first, Properties, tick **Unblock**), and double-click **install.bat**; **uninstall.bat** removes it
again. Run install.bat again at any time to repair; finished steps are skipped.

**Problems?** [Report them on GitHub](https://github.com/aiismsofficial-ui/riffchart/issues/new/choose) with your version number (Settings),
Windows version, CPU or graphics card, and the last lines in the RiffChart window.

**Start:** the RiffChart shortcut (or `RiffChart.bat` in its folder). It opens the app in your browser; close the RiffChart
window to stop it.

For development:

```bash
python -m pip install --user uv
uv sync --extra dev --extra cuda --extra transcribe --python 3.12      # or --extra cpu
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

## Support RiffChart

RiffChart is free and made by one person in their spare time. If it saves you hours of charting, you can
[tip the dev on Ko-fi](https://ko-fi.com/aiismsofficial) &#9829;. Tips help keep it going
(a dedicated bass model and drums are next on the list). Bug reports, song requests and charts that came out wrong are just as welcome:
open an issue.

## Sharing charts

- **Share the chart, not the song.** A song folder or `.sng` from RiffChart contains the song's audio, which
  is someone else's copyrighted recording. Only share it where you're allowed to share that music. The chart
  itself (`notes.chart` + `song.ini`) has no audio in it; others can line it up with their own copy.
- **Keep the label.** RiffChart charts say `charter = RiffChart` in `song.ini` and "Charted automatically by
  RiffChart" on the loading screen. Please leave that in when you share one (with your name added if you've
  edited it), so players and charters can always tell an AI chart from a hand-made one.
- **Hand-made charts stay best.** RiffChart is for songs nobody has charted, or as a first draft to edit; if
  a song has a good community chart, play that one.

## Licence

The code is GPL-3.0-or-later (see [LICENSE](LICENSE)). The model files are CC BY-NC 4.0: free to use and
share with credit, not for commercial use. The beat tracker builds on Beat This (MIT).

## Credits

Built on [PyTorch](https://pytorch.org/), [Demucs](https://github.com/facebookresearch/demucs),
[Beat This!](https://github.com/CPJKU/beat_this),
[FastAPI](https://fastapi.tiangolo.com/) and [mutagen](https://github.com/quodlibet/mutagen).
Chart formats follow the conventions documented by the Clone Hero and Moonscraper communities.
