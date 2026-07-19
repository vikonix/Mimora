# Architecture

Mimora is built on SpeakLoop, an MVP voice-tutor project. Its default **phoneme** scoring engine is Mimora's own; the alternative **acoustic** engine reuses the pronunciation-scoring core of [OpenPronounce](https://github.com/Halleck45/OpenPronounce) (MIT) as a library.

| File | Responsibility |
|---|---|
| `main.py` | `PronunciationTrainerGUI` - Tkinter GUI, recording, the Prompt→Record→Analyze→Feedback→Loop state machine, threading orchestration, LLM-server subprocess management. |
| `mimora/engine.py` | Engine dispatcher - binds the backend chosen by `ENGINE` (settings.json `"engine"`: `phoneme` default, `acoustic` alternative, `none` = scoring off for slow machines) and exposes one `analyze(...)` interface, so `main.py` is engine-agnostic. |
| `mimora/prosody.py` | Engine-agnostic prosody layer: F0/energy contour extraction (no torch). Computed in `main.py` from the raw user/reference audio so the pitch/energy charts work the same across engines. |
| `mimora/tts.py` | `TTSManager` - TTS facade over per-language synthesis backends (`KokoroBackend` 24 kHz for English, `SupertonicBackend` 44.1 kHz for Spanish; selected by the language profile). `synthesize()` returns the waveform; `play_array()` plays any waveform at any rate (your recording at 16 kHz). `loudness_envelope()` precomputes the per-frame mouth-openness track used by the face. |
| `mimora/face_widget.py` | `FaceWidget` - schematic articulation face (Tk Canvas). Talking mouth driven from a precomputed loudness track while audio plays; smiley reflecting the score when idle. Stdlib `tkinter` only. |
| `mimora/progress_widget.py` | `ProgressRing` - circular session-average gauge on the right of the hero score row (mirrors the face on the left). Ring drawn by Pillow (supersampled, antialiased); the average and phrase count are Tk text. Shows the running tally that used to live in the status bar. |
| `mimora/llm.py` | `LLMManager` - OpenAI-compatible client. `generate_phrase()` produces one practice phrase per request. |
| `mimora/llm_server_ctl.py` | `LLMServerController` - starts/stops the local LLM-server subprocess (used by the `local_server` backend). |
| `mimora/recorder.py` | `AudioRecorder` - microphone capture thread, device selection, normalization and WAV dumps; returns the take as one 16 kHz array. |
| `mimora/audio_io.py` | Shared audio-device infrastructure (PortAudio reset, winsound path selection) depended on by both the mic and speaker paths. |
| `mimora/translator.py` | `TranslatorManager` - offline NLLB-200 translation of the practice phrase for the translation panel. |
| `mimora/ui.py` | `TrainerView` - passive Tkinter view (all widgets and copy), composed into the controller; talks to it only via typed callbacks. |
| `mimora/loader.py` | Pure, stateless config-loading helpers (JSON parsing, setting validation, device probe) used by `config.py`. |
| `mimora/prosody_utils.py` | Pure plotting helpers (`to_semitones`, `resample_series`) kept free of the ML/audio stack. |
| `mimora/config.py` | All configuration: device, model names, score threshold, practice-text path, phrase-generation settings, audio settings. |
| `llm_server/server.py` | Standalone FastAPI server loading GGUF models via `llama_cpp`; runs as a separate process to avoid CUDA contention. See [`llm_server/README.md`](../llm_server/README.md). |
| `pronunciation/phoneme/speech.py` | **Default** pronunciation engine - espeak reference phonemes vs a wav2vec2 phoneme recognizer, feature-weighted edit distance, calibrated 0-5 grade. No GUI dependency. |
| `pronunciation/phoneme/calibrate.py` | On-request scoring calibration for the **default** phoneme engine: reads the per-attempt samples from `logs/phoneme_samples.jsonl` and writes `pronunciation/phoneme/calibration.json`. |
| `pronunciation/acoustic/speech.py` | Alternative pronunciation engine (adapted from OpenPronounce). Single entry point `analyze(...)`; Wav2Vec2 embeddings + DTW, phoneme comparison, scoring. No GUI dependency. |
| `pronunciation/acoustic/calibrate.py` | On-request scoring calibration for the acoustic engine: reads the per-attempt samples from `logs/acoustic_samples.jsonl` and writes the acoustic floor to `pronunciation/acoustic/calibration.json`. |
| `config/` | User configuration data: `settings.json` (hand-edited preferences), `hardware_config.json` (machine-derived overrides written by `tools/detect_hardware.py`), and `themes/` (UI color schemes). |
| `texts/practice_text.txt` | Default source text shown in the input panel at startup; put your own practice texts in `texts/` (personal texts stay local - `texts/` is gitignored, only the bundled starter texts are committed). |
| `tools/detect_hardware.py` | Standalone hardware probe (RAM/CPU/GPU/VRAM/audio). Writes `config/hardware_config.json`, whose `config` section supplies machine-derived overrides (e.g. `EXTERNAL_N_GPU_LAYERS`, `WAV2VEC2_DEVICE`) that `mimora/config.py` reads in preference to its defaults. |
| `install.py` | Standalone, idempotent installer: checks Python, detects GPU/CUDA and installs matching torch / llama-cpp-python, installs requirements, checks espeak-ng, pre-downloads the HF models and the GGUF chat model, then runs `detect_hardware.py`. |
