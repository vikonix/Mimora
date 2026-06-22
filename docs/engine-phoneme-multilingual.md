<!-- SPDX-License-Identifier: MIT -->
<!-- Copyright (c) 2026 Valery Kovalev -->

# New Engine — Phoneme-Based Multilingual Scoring

*A short overview for technically literate readers. Focus: the ideas and the
workflow, not the code.*

## The core idea

The new engine answers a slightly different question: **did the learner produce
the right sounds, in the right order, for this phrase?**

Instead of comparing two audio clips, it compares two *sequences of phonemes*
(the elementary speech sounds, written in the international IPA alphabet):

- the **expected** phonemes, derived from the written phrase;
- the **spoken** phonemes, recognized from the learner's audio.

This shift has one large payoff: **adding a language becomes almost free.** There
is no reference recording to synthesize and no per-language model to train —
expected phonemes for any of 100+ languages come straight from text. The design
goal is not a perfect grade but **actionable feedback**: showing the learner
*where* they went wrong so they can improve.

It lives as a separate module, selectable alongside the base engine.

## The workflow

```
phrase text ──(espeak-ng)──►  expected phonemes ─┐
                                                 ├─► align & score ─► 0–5 / 0–100
user audio ──(phoneme recognizer)──► spoken phonemes ─┘            + highlighting
```

1. **Expected phonemes.** A grapheme-to-phoneme tool (**espeak-ng**) converts the
   written sentence into its ideal phoneme sequence. This is the language switch:
   pick the language code, get the phonemes — no audio needed.
2. **Spoken phonemes.** The learner's recording is passed through a phoneme
   recognizer that writes down the sounds actually produced.
3. **Align** the two sequences and score them.
4. **Return** a coarse grade plus two highlightings: the target text marking what
   was said well, and the recognized text marking where the errors are.

## How it recognizes the voice

The engine uses **Wav2Vec2-XLSR-53** — a multilingual neural model fine-tuned to
output **phonemes directly** (in espeak-style IPA), rather than words. Two
properties matter:

- It is **multilingual by construction**, so the same model serves every language
  the engine supports.
- Its phoneme alphabet **matches** the one espeak produces, so both sides of the
  comparison speak the same symbolic language and line up cleanly.

The recognizer is the engine's main source of noise: on short or accented speech
it can occasionally invent extra sounds. The scoring is deliberately built to
tolerate this (see below).

## How the score is produced

The two phoneme sequences are aligned with an **edit-distance** algorithm — the
standard way to compare symbol sequences — but with a twist: the cost of a
mismatch is **how articulatorily different the two sounds are**, measured with a
phonetics library (**panphon**). Confusing the two English "r" sounds barely
costs anything; swapping unrelated sounds costs a lot. Feedback therefore reflects
the *severity* of an error, not just its presence.

The final grade blends **two axes**:

- **Pronunciation quality** — how close the spoken sounds are to the expected
  ones. As in the base engine, this is **anchored** between *good* and a
  per-utterance *bad* baseline so that gibberish scores near 0 instead of drifting
  to a flattering middle.
- **Recall** — the share of expected sounds the learner actually produced. This
  separates a partial reading ("said half the phrase") from a full but imperfect
  one, which the distance alone cannot tell apart.

Two safeguards keep the recognizer's noise from distorting the grade: a **cap on
invented sounds** (so a burst of hallucinated phonemes can't sink an otherwise
good attempt) and a **length-aware baseline** (so short phrases aren't judged too
harshly).

## A coarse, encouraging scale

Because the goal is improvement rather than an exam mark, the raw 0–100 number is
collapsed into a **simple 0–5 band**. Coarse buckets hide the small,
recognizer-driven wobble between near-identical attempts, and let the product hold
two promises: a genuinely good attempt reliably lands in the top band, and the
reference rendering of a phrase always sits at the top.

## Strengths and limits, in one breath

It is **language-agnostic and reference-free** — its reason for existing — and it
**locates errors well** (it reliably flags which sounds were off),
which is exactly what the feedback UI needs. The trade-offs: turning speech into
discrete phonemes **discards fine acoustic detail** (timing, stress, vowel
length), so it is a weaker *precise grader* than the acoustic base engine, and its
accuracy varies somewhat across speaker accents. For "help me hear my mistakes,"
those trade-offs are acceptable; for fine-grained scoring, the base engine still
leads.
