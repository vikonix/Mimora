# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Project Overview

Mimora is a local, offline **pronunciation trainer** (Python 3.11/3.12, Tkinter GUI). It speaks an LLM-generated phrase aloud (Kokoro TTS), records the user repeating it, then scores the attempt against the reference with the active pronunciation engine plus prosody. The engine is selected by `config.ENGINE` (settings.json `"engine"`; see `mimora/engine.py`): the **default is `phoneme`** (espeak reference phonemes vs a wav2vec2 phoneme recognizer, feature-weighted edit distance, calibrated 0–5 grade); the alternative `acoustic` engine uses Wav2Vec2-embedding cosine-DTW plus phoneme/word error rates; `none` disables scoring entirely (no recognizer model is loaded, every take is accepted, the GUI shows a neutral "scoring off" read-out - for slow machines). The user repeats the same phrase until the score passes, then generates the next one.

The pronunciation-scoring core in `pronunciation/acoustic/` is adapted from [OpenPronounce](https://github.com/Halleck45/OpenPronounce) (MIT) and reused as a GUI-agnostic library.

## Running the App

```bash
# Root requirements.txt pulls in the subproject files via -r (llm_server,
# pronunciation/acoustic and pronunciation/phoneme), so one install covers all:
pip install -r requirements.txt
python main.py
```

Also requires the native **espeak-ng** binary on `PATH` (used by `phonemizer`) and a GGUF chat model at `config.EXTERNAL_MODEL_PATH`.

**Default LLM backend**: `local_server` - `llm_server/server.py` is launched automatically as a subprocess.
**Alternative**: set `"llm_backend": "lm-studio"` in `config/settings.json` and run LM Studio on `http://localhost:1234`.

## Architecture

- [`main.py`](main.py) - `PronunciationTrainerGUI`: Tkinter GUI, recording, the Prompt→Record→Analyze→Feedback→Loop state machine, threading orchestration, LLM-server subprocess management.
- [`mimora/engine.py`](mimora/engine.py) - engine dispatcher: binds the backend chosen by `config.ENGINE` (`phoneme` default, `acoustic` alternative, `none` = scoring off) and exposes one `analyze(...)` interface, so `main.py` is engine-agnostic and only the selected engine's weights load. `pronunciation/none/` is a no-op engine returning `PronunciationResult(scored=False, passed=True)`; the GUI renders unscored results neutrally (`ui.py _show_unscored_feedback`).
- [`pronunciation/phoneme/speech.py`](pronunciation/phoneme/speech.py) - **default** pronunciation engine (text-only reference): espeak reference phonemes vs a wav2vec2 phoneme recognizer, feature-weighted edit distance, mapped to a calibrated 0–5 grade. Model calibration lives in `pronunciation/phoneme/<lang>_model_calibration.json` (committed, selected by espeak language); a per-user `phoneme_good` override in `pronunciation/phoneme/calibration.json` (gitignored, shaped `{lang: {users: {name: ...}}}`). Samples appended to `logs/phoneme_samples.jsonl`. No GUI dependency.
- [`pronunciation/acoustic/speech.py`](pronunciation/acoustic/speech.py) - alternative pronunciation engine (adapted from OpenPronounce). Single entry point `analyze(user_audio, expected_text, reference_audio) -> PronunciationResult`. Wav2Vec2 embeddings + per-step cosine DTW, phoneme/word error rates. Scoring uses a calibratable acoustic floor (`pronunciation/acoustic/calibration.json` overrides `config.PRONUNCIATION_ACOUSTIC_GOOD`); every attempt's raw components are appended to `logs/acoustic_samples.jsonl`. No GUI/Tkinter dependency. Prosody is no longer computed here (returns `prosody={}`) - see `mimora/prosody.py`.
- [`mimora/prosody.py`](mimora/prosody.py) - engine-agnostic prosody layer: F0/energy contour extraction (librosa/sklearn, no torch). `main.py` calls `compute_prosody(user, reference)` after `analyze` (via `_compute_prosody_safe`, skipped entirely while both prosody charts are hidden - pyin pitch tracking is expensive on slow machines) and fills `result.prosody`, so the pitch/energy charts work identically regardless of the active engine. Pure plotting helpers (`to_semitones`, `resample_series`) stay in [`mimora/prosody_utils.py`](mimora/prosody_utils.py).
- [`pronunciation/acoustic/calibrate.py`](pronunciation/acoustic/calibrate.py) - on-request semi-automatic calibration: fits the acoustic floor from collected samples and writes `pronunciation/acoustic/calibration.json` (`--dry-run` to preview).
- [`mimora/audio_io.py`](mimora/audio_io.py) - shared audio-device infrastructure depended on by both the mic and speaker paths (so neither depends on the other). Exports `reset_portaudio()` (shared PortAudio reset used by recording and playback), `uses_winsound()` (single source of truth for which playback path is taken), `WINSOUND_AVAILABLE`, the winsound lead-in constants, and `KOKORO_SAMPLE_RATE`. The coordinating `AUDIO_LOCK` and pipeline `AUDIO_SAMPLE_RATE` stay in [`config.py`](mimora/config.py) with the other `AUDIO_*` settings.
- [`mimora/tts.py`](mimora/tts.py) - `TTSManager`: Kokoro TTS. `synthesize()` returns the waveform; `play_array(waveform, sample_rate)` plays any waveform. Also exports `loudness_envelope(waveform, sample_rate, fps)` (per-frame RMS → 0..1 track that drives the talking mouth - see [`face_widget.py`](mimora/face_widget.py)). The PortAudio/winsound device helpers it uses live in [`mimora/audio_io.py`](mimora/audio_io.py).
- [`mimora/face_widget.py`](mimora/face_widget.py) - `FaceWidget`: schematic articulation face on a Tk Canvas. A talking-ellipse mouth opens/closes during playback; a smiley reflects the score while idle. Zero deps beyond stdlib `tkinter`. The mouth is driven by `play_levels(levels, fps)` - a pre-computed loudness track the widget's own `after`-loop replays by wall-clock - rather than a live audio callback (see the Windows-audio note below).
- [`mimora/llm.py`](mimora/llm.py) - `LLMManager`: OpenAI-compatible client. `generate_phrase()` produces one practice phrase per request (non-streaming).
- [`mimora/config.py`](mimora/config.py) - all configuration (device, model names, score threshold, practice-text path, phrase-generation settings, audio settings). User overrides live in `config/settings.json`; UI themes in `config/themes/`.
- [`llm_server/server.py`](llm_server/server.py) - standalone FastAPI server loading GGUF via `llama_cpp`; runs as a separate process to avoid GPU contention.
- [`texts/practice_text.txt`](texts/practice_text.txt) - default source text loaded into the input panel at startup; `texts/` holds additional practice texts.

## State Machine (pronunciation loop)

1. **Prompt** - `llm_mgr.generate_phrase(source_text)` → `tts_mgr.synthesize(phrase)`. The synthesized array is stored as `self.reference_audio` and played for the user. Phrase generation + synth + playback all run in one daemon thread (`_generate_and_prompt`).
2. **Record** - shared recording path (`AudioRecorder._record_loop` → `AudioRecorder.get_audio`), 16 kHz mono. Gated by `_can_record()` (a phrase must be ready and nothing else busy).
3. **Analyze** - `_finalize_recording` → `analyze_recording` (daemon thread) calls `engine.analyze(...)` - the dispatcher in `mimora/engine.py`, which selects `pronunciation.acoustic` or `pronunciation.phoneme` by `config.ENGINE`.
4. **Feedback** - `_show_feedback` (via `root.after`) shows score, transcription, problem words; enables replay buttons.
5. **Loop** - if `result.passed` the user can generate the next phrase; otherwise the same phrase/reference are retained for another attempt.

## Key Patterns & Gotchas

- **Threading**: Recording, analysis, model loading, phrase generation, and playback run in daemon threads. **Always update the GUI via `root.after()`**; never read/write Tk widgets from a background thread. Source text is read on the main thread and passed into the worker.

- **Reference audio is synthesized once** ([`mimora/tts.py`](mimora/tts.py) `synthesize()`): the same Kokoro waveform is both played to the user and passed to `analyze()` as the reference. There is no second TTS engine.

- **Sample rates**: recording uses 16 kHz; Kokoro outputs 24 kHz; Wav2Vec2 needs 16 kHz. `engine.analyze` takes `user_sr` and `reference_sr` and `_prepare_waveform` resamples to 16 kHz internally. `play_array` plays the reference at 24 kHz and the user recording at 16 kHz.

- **Pronunciation model lifecycle** ([`pronunciation/acoustic/speech.py`](pronunciation/acoustic/speech.py)): models load lazily; `load_models()` makes loading explicit (call in a background thread at startup) and `warm_up()` removes first-call latency - mirroring `mimora/tts.py`. Device follows `config.WAV2VEC2_DEVICE` (defaults to `config.DEVICE`). `speech.py` reads config via `getattr(..., default)` so it stays usable without config edits.

- **GPU contention**: Wav2Vec2, Kokoro, and `llama_cpp` can compete for VRAM. Mitigations: the LLM runs in a **separate process** (`llm_server/`), and the loop's phases (LLM → Kokoro → Wav2Vec2) run **sequentially**. If VRAM is tight, set `WAV2VEC2_DEVICE = "cpu"` in `mimora/config.py`.

- **Phrase generation** ([`mimora/llm.py`](mimora/llm.py)): `generate_phrase()` is a single non-streaming completion with its own system prompt (`config.PHRASE_GEN_SYSTEM_PROMPT`); it does **not** touch the conversational `self.messages` history. To keep phrases varied, the prompt only includes a sliding window of the source text (`_current_window`, advanced every `PHRASE_GEN_WINDOW_REPEATS` calls) plus a random focus word and opening-style hint; `_clean_phrase` strips quotes/list markers.

- **Audio normalization** ([`main.py`](main.py)): peaks are normalized before analysis; silence below `AUDIO_MIN_PEAK_THRESHOLD = 0.01` skips gain adjustment.

- **Windows audio**: TTS/playback uses `winsound` to bypass PortAudio/MME issues; a `sounddevice` fallback exists for other platforms. A ~150 ms silence lead-in is prepended to avoid clipping the first audio. `config.AUDIO_LOCK` serialises PortAudio init/teardown between the mic and speaker paths.

- **Talking mouth without an audio callback**: `winsound.PlaySound` plays the whole buffer with no per-frame hook, so the face cannot follow live amplitude on Windows. Instead the loudness envelope is pre-computed from the (fully known) waveform via `tts.loudness_envelope()` and the face replays it on its own wall-clock `after`-loop (`FaceWidget.play_levels`). `main.py._play_with_face()` is the single chokepoint wrapping every `play_array` call: it starts the track, plays, then closes the mouth - and prepends closed-mouth frames matching `TTSManager.playback_lead_in_seconds()` so the animation lines up with the Windows warm-up silence. The same path is used on all platforms (no live-RMS branch). Because the envelope uses the same `sample_rate` as playback, the slowed-reference speed (lowered sample rate) stretches the mouth track automatically. `_rest_face_if_current()` stops the mouth on finish/interrupt without clobbering a playback that superseded it (same identity-guard idea as `_playback_finished`).

- **LLM server subprocess**: started in `_start_llm_server()`, polled via `LLMManager.check_connection()` until ready; terminated in `quit_app()` with a 5-second kill fallback.

- **Device detection** ([`mimora/config.py`](mimora/config.py)): CUDA auto-detected via `torch.cuda.is_available()`.

- **espeak-ng** is a native binary dependency of `phonemizer` (separate install, not pip) - required for phoneme extraction and word-error analysis.

## Testing

```bash
python -m unittest discover -s tests -v              # all fast unit tests, no model download
python tests/test_speech.py user.wav [ref.wav]       # optional end-to-end (loads the model)
```

## Code Style (Python)

- No linting/formatting config - follow PEP 8.
- Type hints used throughout (`from typing import Optional, List`).
- Logging via `logging`: `%(asctime)s [%(levelname)s] (%(threadName)s) %(message)s`.
- Use explicit `RuntimeError` with a descriptive message for runtime validation, not `assert`.
- Library deprecation warnings are filtered at the top of `main.py`.
