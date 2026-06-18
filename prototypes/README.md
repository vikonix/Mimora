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
pip install -r prototypes/requirements.txt  # extras for the prototypes (panphon)
```

The espeak-ng native binary must be on PATH (already required by `pronounce/`).

All scripts default to the sample data in `records/` (`normalized.wav` user take,
`model.wav` reference, `phrase.txt` text), so they run with **no arguments**.

## Current prototypes

### `w2v2_pronounce_poc.py` — multi-language pronunciation scoring

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

Recognizer: **`w2v2`** — `facebook/wav2vec2-xlsr-53-espeak-cv-ft`, a wav2vec2 CTC
model that emits espeak-style IPA. Accurate and its phone inventory matches the
espeak reference. Beyond the project deps it needs only panphon. (An earlier
universal phone recognizer was tried and dropped as too noisy — ~16/100 where the
core scores ~95.)

Install the extra with `pip install -r prototypes/requirements.txt`.

```bash
# No args — runs on the English sample in records/:
python prototypes/w2v2_pronounce_poc.py

# GPU, or your own audio/text:
python prototypes/w2v2_pronounce_poc.py --device cuda
python prototypes/w2v2_pronounce_poc.py user.wav --text "hola, ¿cómo estás?" --lang es
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

### Evaluation harness — score many recordings, compare engines

`run_eval.py` runs every engine over a whole **dataset** instead of a single
recording, so an engine can be judged on statistics, not one phrase. It is built
from small, swappable pieces:

- `eval_core.py` — the shared contracts (`Sample`, `EngineResult`, the `Engine`
  protocol: `init` / `parse` / `close`), the dataset walker, and the comparison
  statistics (Pearson, Spearman, MAE, bias, verdict agreement). Stdlib + numpy
  only.
- `core_prod.py` — the **reference** engine: a thin wrapper over the production
  `pronounce.analyze` (Wav2Vec2 + cosine-DTW). Needs the reference recording.
- `core_w2v2.py` — the **test** engine: a thin adapter over
  `w2v2_pronounce_poc` (espeak reference → wav2vec2 phonemes → edit
  distance). Text-only; ignores the reference recording. The espeak reference and
  the recognizer use slightly different IPA conventions for the same sounds, so
  `w2v2_pronounce_poc._PHONE_FOLD` canonicalizes both sides (e.g. `ɹ→r`,
  `æ→a`, `ɚ→ə`, `oʊ→o`) — Spanish-safe (near-identity there) and the main lever
  to tune during calibration.

Adding another engine = one more `core_*.py` exposing `init`/`parse`/`close`,
then listing it in `run_eval.py`'s `test_engines`.

**Dataset layout.** A *sample folder* is a copy of `records/`
(`normalized.wav` = attempt scored, `model.wav` = reference take, `phrase.txt` =
text). A *dataset* is a folder of sample folders; a *collection* is a folder of
datasets. The harness figures out which is which **by content, not by name**, so
folder names are free:

```
VKO/                         <- collection (pass this)
  mic/                       <- dataset
    001/{normalized.wav, model.wav, phrase.txt}
    002/{...}
  bt/                        <- dataset
    001/{...}
  mistakes/                  <- dataset
    001/{...}
```

Filenames are overridable (`--user-name` etc.); a reference recording is
currently **required** (the prod engine needs it), so sample folders missing one
are skipped with a note (e.g. an empty `bt/004`).

```bash
# Point at the top level; mic/bt/mistakes are discovered automatically.
# --good/--bad label classes so the run also reports ROC-AUC + best threshold:
python prototypes/run_eval.py "C:/VOICE_DATASET/ENGLISH/VKO" --good mic --bad mistakes

# Or name datasets directly. GPU, capped for a quick smoke run:
python prototypes/run_eval.py vko_mic vko_bt --device cuda --limit 5
```

**What the log shows.** First a **RUN CONFIG** header recording each engine's
parameters (models, device, thresholds, weights) so the numbers are reproducible.
Then, per sample, a block with each engine's score, verdict and sub-scores
(w2v2: reference/spoken IPA, phoneme score, recall, per-phone distance, bad
baseline; prod: its `[pronounce]` line). With `--verbose`, the w2v2 block also
prints the full `ref hyp flag` alignment table (`~` near-miss, `*` mismatch) for
spotting systematic inventory differences. Then, per dataset and pooled across
all datasets, the agreement of each test engine vs the `core_prod` reference
(Pearson, Spearman, MAE, bias, verdict agreement). Finally, if `--good`/`--bad`
are given, **class separability** for every engine: the good-vs-bad ROC-AUC
(ranking quality, ignores score offset) and the best threshold with its accuracy.

The CSV carries the same sub-scores as extra columns
(`<engine>_score`, `<engine>_passed`, `<engine>_<subscore>`), so calibration can
be done offline from the CSV without re-running the models.

Output (both **overwritten each run**, unlike the shared appending
`prototype.log`): a per-sample `eval_results.csv` (every engine's score +
pass/fail) and a full `eval_run.log` (per-phrase blocks + per-dataset, pooled and
separability summaries). Paths overridable via `--csv` / `--log`. English only
for now (the reference core is calibrated on English). TTS-generated references
and mp3 datasets (Forvo, etc.) are a later, separate task.
