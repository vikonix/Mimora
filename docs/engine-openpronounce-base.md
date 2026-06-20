# Base Engine — OpenPronounce (Acoustic + Transcription)

*A short overview for technically literate readers. Focus: the ideas and the
workflow, not the code.*

## The core idea

The base engine answers one question: **does the learner's recording sound like
a reference rendering of the same phrase?**

It does this by comparing two audio clips — the user's attempt and a
machine-generated "model" pronunciation of the same sentence — and measuring how
far apart they are. The closer the two sound, the higher the score. Because it
leans on a concrete reference recording, it is precise on English (the language
it is tuned for) but needs that reference to exist for every phrase.

The engine is adapted from the open-source **OpenPronounce** core and lives in
the `pronounce/` module. It is application-agnostic: it takes audio in, returns a
score and feedback out.

## The workflow

```
user audio  ─┐
             ├─► [1] prepare & align ─► [2] three measurements ─► [3] one score 0–100
model audio ─┘        (DTW)              acoustic / phoneme / word
```

1. **Reference.** The app synthesizes the target phrase with a text-to-speech
   voice (Kokoro). That clip is the "model" — the sound the learner is aiming for.
2. **Prepare.** Both clips are resampled, peak-normalized and trimmed of silence
   so that only the spoken content is compared.
3. **Compare** on three independent axes (below).
4. **Combine** the three into a single 0–100 score, plus per-word feedback and
   two prosody curves (pitch and energy) drawn for both clips.

## How it "recognizes" the voice

The engine uses a neural speech model (**Wav2Vec2**) in two different ways:

- **Acoustic fingerprint.** Wav2Vec2 turns each clip into a sequence of
  *embeddings* — dense numeric snapshots of how the audio sounds, frame by frame.
  These capture pronunciation detail without first turning speech into letters.
- **Transcription.** A second pass uses the same family of model as a speech
  recognizer to write down *what words* it heard in the user's clip.

The key move on the acoustic side is **alignment**. People speak at different
speeds, so the two clips never line up one-to-one. A classic algorithm,
**Dynamic Time Warping (DTW)**, stretches and compresses the timelines until the
two embedding sequences match as well as possible, then reports the average
distance per aligned step. This makes the comparison fair regardless of tempo.

## How the score is produced

Three measurements are taken, each turned into a 0–100 sub-score:

- **Acoustic similarity (40%)** — the average DTW distance between the two
  embedding sequences. This is the heart of the engine: it reflects *how* the
  sounds were produced.
- **Phoneme accuracy (30%)** — the transcription is converted to phonemes and
  compared to the expected phonemes; the share that matches becomes the score.
- **Word accuracy (30%)** — how closely the recognized words match the target
  text. This catches missing or wrong words.

A crucial detail makes the acoustic number trustworthy: **good/bad anchoring.**
Any two vowels already sound somewhat alike, so a raw distance would flatter even
nonsense. To prevent that, every utterance gets its own "completely wrong"
ceiling, estimated by pairing unrelated frames at random. The measured distance is
then placed on a scale running from *good* (a typical native-quality attempt,
calibrated per user over time) to *bad* (that random ceiling). The result is that
gibberish lands near 0 and a clean attempt near 100, rather than everything
drifting toward a misleading middle.

## Strengths and limits, in one breath

It is **accurate on English** and rewards genuinely good pronunciation, because
it listens to the actual acoustics rather than just symbols. The trade-offs:
it **requires a reference recording** for every phrase, and it is **tuned for
English** — adding a new language means new models and recalibration, not a
simple switch. Those limits are exactly what the new engine is designed to remove.
