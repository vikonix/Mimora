# prototypes/

Throwaway experiments and spikes. Code here is **not** part of the shipped app:
it is for trying ideas (new languages, alternative scoring, model comparisons)
before committing them to `pronounce/`, `mimora/`, etc.

## Conventions

- Start every script with `import _bootstrap` (first import). It adds the project
  root to `sys.path`, so you can reuse real project code from anywhere:

  ```python
  import _bootstrap  # noqa: F401
  from pronounce import analyze
  from mimora import audio_io
  ```

- One self-contained script per idea. Don't turn this folder into a package and
  don't let app code import from here.
- Reuse existing dependencies where possible; note any **new** dependency at the
  top of the script and add it to `prototypes/requirements.txt`.

## Install

```bash
pip install -r requirements.txt             # project root
pip install -r pronounce/requirements.txt   # pronounce core
pip install -r prototypes/requirements.txt  # extras for the prototypes (allosaurus)
```

The espeak-ng native binary must be on PATH (already required by `pronounce/`).

All scripts default to the sample data in `records/` (`normalized.wav` user take,
`model.wav` reference, `phrase.txt` text), so they run with **no arguments**.

## Current prototypes

### `allosaurus_pronounce_poc.py` — multi-language pronunciation scoring

A lighter, fully language-parametrized alternative to the current Wav2Vec2 +
embedding-DTW core:

```
text ──espeak-ng──▶ reference IPA ┐
                                  ├─edit distance──▶ score + per-phoneme diff
audio ──phoneme ASR──▶ spoken IPA ┘
```

Adding a language = one row in the `LANGUAGES` table. No training, text-only (no
per-phrase TTS reference). Scoring uses an **articulatory feature distance**
(panphon), so near-misses like the rhotic `r`/`ɹ` cost little while unrelated
swaps cost ~1 — language-general, works for Spanish unchanged.

Two recognizer backends, picked with `--asr`:

- **`w2v2`** (default): `facebook/wav2vec2-xlsr-53-espeak-cv-ft`, a wav2vec2 CTC
  model that emits espeak-style IPA. Accurate and its phone inventory matches the
  espeak reference. Beyond the project deps it needs only panphon.
- **`allosaurus`**: the original universal recognizer. An English baseline check
  (below) showed it is too noisy (~16/100 where the core scores ~95), so it is
  kept only for comparison and additionally needs `allosaurus`.

Install both extras with `pip install -r prototypes/requirements.txt`.

```bash
# No args — w2v2 backend on the English sample in records/:
python prototypes/allosaurus_pronounce_poc.py

# GPU, Allosaurus backend, or your own audio/text:
python prototypes/allosaurus_pronounce_poc.py --device cuda
python prototypes/allosaurus_pronounce_poc.py --asr allosaurus
python prototypes/allosaurus_pronounce_poc.py user.wav --text "hola, ¿cómo estás?" --lang es
```

### `wav2vec2_compare_poc.py` — run the existing core, compare side-by-side

Runs the production `pronounce.analyze` (Wav2Vec2 embeddings + cosine-DTW + CTC
ASR) on a recording so its score can be compared with the light pipeline. Note
the asymmetry: the Wav2Vec2 core needs a **reference audio** of the phrase, the
light route needs only the **text**.

**Compare in English first.** The core is calibrated on English, so the fair
experiment is to run *both* pipelines on the same English recordings (where
Wav2Vec2 is a trusted reference) and check that the lighter route reproduces its
good/bad verdicts. Only then switch the core to Spanish via `--model` /
`--espeak` / `--lang es` (`--device cuda` on a CUDA GPU). `--compare` runs the
light pipeline (`--asr w2v2` by default) on the same user audio and prints both
scores.

```bash
# Step 1 — fair baseline on the bundled English sample (records/), no args:
python prototypes/wav2vec2_compare_poc.py --compare

# Step 2 — only after step 1 looks good: move the core to Spanish.
python prototypes/wav2vec2_compare_poc.py user.wav reference.wav \
    --text "hola, ¿cómo estás?" --lang es \
    --model facebook/wav2vec2-large-xlsr-53-spanish --espeak es \
    --device cuda --compare
```
