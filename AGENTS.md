# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Project Overview
SpeakLoop is a Python desktop voice tutor app (Tkinter GUI) for language learning using local AI models.

## Running the App
```bash
pip install -r requirements.txt
pip install -r llm_server/requirements.txt
python main.py
```

**Default backend**: `local_server` — `llm_server/server.py` is launched automatically as a subprocess.  
**Alternative**: set `LLM_BACKEND = "lm-studio"` in `config.py` and start LM Studio on `http://localhost:1234`.

## Architecture

- [`main.py`](main.py) — `VoiceTutorGUI`: Tkinter GUI, audio recording, threading orchestration, LLM server subprocess management
- [`stt.py`](stt.py) — `STTManager`: faster-whisper speech-to-text with VAD filtering
- [`llm.py`](llm.py) — `LLMManager`: OpenAI-compatible streaming client; works with both `local_server` and `lm-studio` backends
- [`tts.py`](tts.py) — `TTSManager`: Kokoro TTS with winsound playback on Windows
- [`config.py`](config.py) — all configuration (languages, model settings, backend selection, API endpoints)
- [`llm_server/server.py`](llm_server/server.py) — standalone FastAPI server that loads GGUF models via llama_cpp; runs as a separate process to avoid GPU contention with Kokoro

## Key Patterns & Gotchas

- **Threading**: Recording, TTS queue processing, and model loading run in separate daemon threads. Always use `root.after()` to update GUI from background threads.

- **TTS sentinel pattern** ([`main.py`](main.py)): LLM sentences are buffered in the TTS queue processor and only played after `_TTS_START_SENTINEL` arrives. This ensures llama_cpp releases the GPU before Kokoro synthesis begins, preventing CUDA contention. Do not remove this pattern when working with the `local_server` backend.

- **LLM history rollback** ([`llm.py`](llm.py)): user message is appended inside `try`; on exception the last message is popped to keep user/assistant pairs consistent.

- **Audio normalization** ([`main.py`](main.py)): peaks are normalized before STT; silence below `AUDIO_MIN_PEAK_THRESHOLD=0.01` is skipped.

- **Sentence streaming**: LLM output is split by regex `(?<=[.!?])\s+` and pushed to the TTS queue as sentences complete. Remaining buffer is flushed at end of stream.

- **Conversation history** ([`llm.py`](llm.py)): trimmed to `LLM_HISTORY_MAX_PAIRS=4` turns after each exchange (configurable in `config.py`).

- **Device detection**: CUDA auto-detected via `torch.cuda.is_available()` — sets compute type to `float16` for CUDA, `int8` for CPU.

- **Windows audio**: TTS uses `winsound` to bypass PortAudio/MME driver issues on Windows. A 150ms silence lead-in (`WINSOUND_LEAD_IN_SAMPLES`) is prepended to each audio block to allow Windows Audio Session initialization. The `sound_lock` in `tts.py` serialises PortAudio init/teardown for the recording path.

- **LLM server subprocess**: started in `_start_llm_server()`, polled via `LLMManager.check_connection()` until ready. Terminated gracefully in `quit_app()` with a 5-second kill fallback.

## LLM Backend Selection (`config.py`)

| `LLM_BACKEND` | Description |
|---|---|
| `"local_server"` | Launches `llm_server/server.py` as a subprocess. Recommended for GGUF models. |
| `"lm-studio"` | Connects to a running LM Studio instance at `LM_STUDIO_URL`. |

## Code Style (Python)
- No linting/formatting config — follow PEP 8.
- Type hints used throughout (`from typing import Optional`).
- Logging via `logging` module: `%(asctime)s [%(levelname)s] (%(threadName)s) %(message)s`.
- `assert` is not used for runtime validation — use explicit `RuntimeError` with a descriptive message.
- Warnings filtered for library deprecation noise (see `main.py` top-level `warnings.filterwarnings`).
