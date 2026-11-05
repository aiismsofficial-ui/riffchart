# How RiffChart works

## Charting a song

1. **Load and separate.** The audio is decoded at 44.1 kHz and split into stems with Demucs
   (`htdemucs_6s`). The guitar stem is what the model follows; in "auto" play-along mode, other melodic
   stems (piano, vocals, other) are blended in where the guitar drops out (`product/follow.py`).
2. **Beats and tempo map.** A fine-tuned [Beat This!](https://github.com/CPJKU/beat_this) model finds
   beats and downbeats; they become a Clone Hero tempo map with time-signature changes
   (`timing/tempo_map.py`). Notes live on a grid of 48 steps per beat.
3. **Model input.** Log-mel spectrograms (128 bands) of the followed stem and of the full mix, resampled
   onto the beat grid so every 48th of a beat has one input frame regardless of tempo
   (`model/dataset.py: window_features`).
4. **Chart model.** An encoder-decoder transformer (`model/net.py`, ~54M parameters, rotary attention,
   QK-norm) writes the chart as tokens, 32 beats at a time with the previous 16 beats of its own output
   as context, all four difficulties in one batch (`model/generate.py`).
5. **Post-processing** (`post/`): playability fixes per difficulty (minimum time between notes, chord
   size and width, sustain gaps; thresholds in `configs/eval.yaml`), the difficulty ladder, the density
   control, and star power phrases.
6. **Writing.** `.chart` with forced / tap flags derived from Clone Hero's HOPO rules
   (`formats/chart_writer.py`), `song.ini`, and the audio as Ogg Vorbis.

## Tokens

```
BOS DIFF DENS CHORD [SRC]  <context beats>  SEP  <target beats>  EOS
each beat:   BAR | BEAT, then per chord:   POS_p  [HOPO | TAP]  NOTE_m  [SUS_NEXT | SUS_k]
```

- `POS_p`: one of 24 musical positions of the 48-step beat (straight and triplet subdivisions).
- `NOTE_m`: 31 fret combinations or open; `HOPO` / `TAP`: the note type (strum is implicit).
- `SUS_NEXT`: sustain until the next chord; `SUS_k`: one of a fixed set of sustain lengths.
- `DENS` / `CHORD`: style controls (notes per beat, share of chords), each with an AUTO value; trained
  with conditioning dropout so AUTO lets the model choose.
- `SRC` (optional): which training library a chart came from, so collections with different charting
  styles are not averaged together.

Decoding is grammar-constrained: at every step only tokens that keep the chart well formed are allowed,
positions only increase within a beat, and a per-difficulty minimum time between notes can be enforced.
Sampling uses temperature 0.7.

## Difficulty ladder and density

- **Ladder** (`post/ladder.py`): Hard, Medium and Easy are capped at 90, 75 and 60% of Expert's chords,
  and in every 16-beat stretch never have more notes or multi-fret chords than the next harder
  difficulty. Notes to drop are chosen by a small learned model of which notes human charters keep
  when they reduce a chart (`post/reduce.py`).
- **Density** (`post/density.py`): sparser thins the weakest notes first (down to 50%); denser gives the
  lower difficulties back Expert notes they had dropped (up to +60%) and nudges Expert decoding.

## Re-charting a section

The editor can ask for new takes of a range of bars (`product/section.py`). The model decodes only
those beats, with the chords before the range as its context and the song's existing tempo map reused,
so nothing outside the range moves. Post-processing then runs on the range alone. The audio analysis is
cached per song, so after the first take each one takes a second or two.

## Training data pipeline

For training your own models from a library of charts with audio (`riffchart-audit`, `src/riffchart/data/`):

1. **Scan** song folders (`.chart` or `.mid` plus audio). The library is opened read-only.
2. **Audit** every song: parse the chart, measure how well notes line up with audio onsets, detect
   constant offsets (corrected) and drift (dropped), flag charts made by other autocharters.
3. **Deduplicate** by metadata and chroma similarity.
4. **Split** artist-disjointly into train / val / test. The evaluation splits are frozen once made.
5. **Stems and features** for every kept song (Demucs, then mel spectrograms).
6. **Train** (`model/train.py`, YAML configs in `configs/`, TensorBoard logs, resumable checkpoints).

## Evaluation

`eval/harness.py` scores generated charts against held-out human charts: onset F1 at 25 and 50 ms,
exact fret agreement, melodic contour agreement, HOPO F1, sustain and chord statistics, note density
relative to the human chart, and playability violations per minute (`eval/playability.py`, the same
rules the editor's live check uses). Baselines for comparison live in `baselines/`.
