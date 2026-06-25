# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev

import os
import sys
import threading
from functools import partial
from pathlib import Path

from . import loader

# Project root — always absolute, regardless of working directory at launch.
# This file lives in mimora/, so the root is one level up.
BASE_DIR = Path(__file__).resolve().parent.parent
# Hand-edited configuration data (settings.json, themes/) lives here.
CONFIG_DIR = BASE_DIR / "config"

# =====================================================================
# Settings files (optional overrides)
# =====================================================================
# Values in this module are layered, lowest priority first:
#   1. built-in defaults — the literals in this file;
#   2. hwconfig/hardware_config.json, "config" section — machine-derived
#      values written by `python hwconfig/detect_hardware.py`;
#   3. settings.json — hand-edited user preferences.
# A missing or broken file simply leaves the lower layers in effect: problems
# are reported to stderr instead of crashing startup, because both files are
# optional and settings.json is edited by hand.


# Machine-derived overrides. Only the "config" section is consumed here;
# the "hardware" section is diagnostics for humans.
_HW = loader.read_json(BASE_DIR / "hwconfig" / "hardware_config.json").get("config")
if not isinstance(_HW, dict):
    _HW = {}

# User preferences. Keys starting with "_" are skipped so they can serve as
# comments (plain JSON has no comment syntax).
_USER = loader.read_json(CONFIG_DIR / "settings.json")

_KNOWN_USER_KEYS = {
    "english_accent",
    "voice",
    "color_theme",
    "pronunciation_score_threshold",
    "max_record_seconds",
    "llm_backend",
    "external_model_path",
    "external_n_ctx",
    "practice_text_file",
    "phrase_gen_window_sentences",
    "phrase_gen_window_repeats",
    "user_name",
    "phrase_length",
    "reference_speed",
    "show_pitch_chart",
    "show_energy_chart",
    "show_face",
    "silence_timeout",
    "silence_threshold",
}
for _key in _USER:
    if not _key.startswith("_") and _key not in _KNOWN_USER_KEYS:
        print(f"[config] settings.json: unknown key {_key!r} ignored",
              file=sys.stderr)


# Validated accessors for settings.json values. The loader functions are pure
# (they take the parsed dict as their first argument); binding _USER — and
# BASE_DIR for path resolution — here keeps every call site below short.
_num = partial(loader.user_number, _USER)
_path = partial(loader.user_path, _USER, BASE_DIR)
_bool = partial(loader.user_bool, _USER)


def save_user_setting(key: str, value) -> bool:
    """Write one setting back to settings.json, keeping every other key.

    Thin wrapper that binds the settings path and the in-memory _USER view to
    loader.save_setting (which preserves hand-edited and "_" comment keys, and
    reports failures instead of raising). Returns True on success.
    """
    return loader.save_setting(CONFIG_DIR / "settings.json", key, value, _USER)


# Display name of the practicing user; shown in the Name field of the UI and
# written back via save_user_setting("user_name", ...) when edited.
USER_NAME = _USER.get("user_name", "")
if not isinstance(USER_NAME, str):
    print(f"[config] settings.json: user_name must be a string, got "
          f"{USER_NAME!r}; using ''", file=sys.stderr)
    USER_NAME = ""

# =====================================================================
# Local model cache (HuggingFace) — download once, then load offline
# =====================================================================
# Kokoro and Wav2Vec2 are pulled from the HuggingFace Hub. We pin
# their cache to a project-local folder so the weights live next to the code and
# survive a cleared home-directory cache. Once every model is present we flip the
# Hub into offline mode, which skips the per-start network round-trip HF normally
# makes to re-validate file revisions — that check is what makes startup feel like
# it "re-downloads" on every launch.
#
# IMPORTANT: huggingface_hub / transformers read these env vars at *import* time,
# so this block must run before those libraries are imported. config is the first
# project module imported by main.py (before stt/tts/pronunciation), so it is early
# enough. We use setdefault() so an externally set HF_HOME is respected.
MODEL_CACHE_DIR = BASE_DIR / "model_cache"
loader.ensure_dir(MODEL_CACHE_DIR)
os.environ.setdefault("HF_HOME", str(MODEL_CACHE_DIR))

# Repo IDs whose presence in the cache means "fully downloaded".
# Whisper (Systran/faster-whisper-small) is intentionally absent: STT is
# disabled (stt.py is kept but never loaded), so its cache must not gate
# offline mode.
_CACHED_REPOS = (
    "hexgrad/Kokoro-82M",             # Kokoro TTS (model + voice files)
    "facebook/wav2vec2-large-960h",   # Wav2Vec2 pronunciation model
)


# Auto offline: the first run downloads online; once everything is cached, every
# later run loads straight from disk with no network access. Delete model_cache/
# to force a fresh download. HF_HOME is read from os.environ (not MODEL_CACHE_DIR)
# so an externally set cache location is honored.
if loader.models_cached(Path(os.environ["HF_HOME"]) / "hub", _CACHED_REPOS):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# =====================================================================
# Language Configuration
# =====================================================================
TARGET_LANGUAGE = "English"
TARGET_LANG_CODE = "en"  # ISO code used for Whisper transcription routing

# =====================================================================
# Controls
# =====================================================================
# Safety threshold to prevent infinite recording loops if a key gets physically stuck
MAX_RECORD_SECONDS = _num("max_record_seconds", 20, minimum=1)

# Recording now starts on a single press and stops on its own after the speaker
# falls silent (recorder.py runs the VAD on the capture thread). These two tune
# that auto-stop; both are read from settings.json so they can be adjusted to a
# room/mic without code changes.
#   silence_timeout   — seconds of continuous silence (after speech has begun)
#                       before the take is finalized automatically.
#   silence_threshold — RMS level (0..1) at or above which a chunk counts as
#                       speech rather than silence. Kept low: silence is near-zero
#                       RMS, while quiet speech may only reach ~0.04, and the level
#                       is averaged over a whole capture chunk (which dilutes brief
#                       bursts) — too high a value never arms the silence timer.
#                       Raise it only if a noisy room keeps the take from stopping.
SILENCE_TIMEOUT = _num("silence_timeout", 3.0, minimum=0.5)
SILENCE_THRESHOLD = _num("silence_threshold", 0.01, minimum=0.0)

# Hardware Acceleration setup — the value detected by hwconfig wins; otherwise
# loader.detect_device probes torch directly (and does not import torch at all
# when hwconfig already supplied a valid device, so this stays cheap).
DEVICE = loader.detect_device(_HW.get("DEVICE"))

# =====================================================================
# LLM Backend Settings
# =====================================================================
# Backend selection, read from settings.json ("llm_backend"):
#   "lm-studio"    — external LM Studio app (must be running separately)
#   "local_server" — llm_server.py started automatically as a subprocess
LLM_BACKEND = _USER.get("llm_backend", "local_server")
if LLM_BACKEND not in ("local_server", "lm-studio"):
    print(f"[config] settings.json: unknown llm_backend {LLM_BACKEND!r} "
          f"(expected 'local_server' or 'lm-studio'); using 'local_server'",
          file=sys.stderr)
    LLM_BACKEND = "local_server"

# =====================================================================
# LM Studio backend (for "lm-studio" backend)
# =====================================================================
LM_STUDIO_URL = "http://localhost:1234/v1"
LM_STUDIO_API_KEY = "lm-studio"
LM_STUDIO_MODEL = "local-model"

# =====================================================================
# Local LLM Server (for "local_server" backend)
# =====================================================================
LOCAL_SERVER_HOST = "127.0.0.1"
LOCAL_SERVER_PORT = 8765
LOCAL_SERVER_URL = f"http://{LOCAL_SERVER_HOST}:{LOCAL_SERVER_PORT}/v1"
LOCAL_SERVER_API_KEY = "local"
LOCAL_SERVER_MODEL = "local-model"

# How long (seconds) to wait for the server to become ready after launching
LOCAL_SERVER_STARTUP_TIMEOUT = 60

# =====================================================================
# GGUF Model Settings (shared by "local_server" and "local_gguf" backends)
# =====================================================================
# GGUF file, read from settings.json ("external_model_path"); a relative path
# resolves against the project root.
EXTERNAL_MODEL_PATH = _path(
    "external_model_path",
    BASE_DIR / "models" / "llama-3.2-3b-instruct-q4_k_m.gguf",
)
# GPU offload and context size: hwconfig picks values matched to the detected
# VRAM/RAM; the literals here are conservative fallbacks for unknown hardware.
EXTERNAL_N_GPU_LAYERS = _HW.get("EXTERNAL_N_GPU_LAYERS", 20)  # layers on GPU (-1 = all)
# Context window size (n_ctx). settings.json ("external_n_ctx") wins over the
# hwconfig-detected value, per the usual layering. int() because the value is
# passed to the server as a command-line argument (a float would break it).
# minimum=256: anything below breaks generation outright (the system prompt
# alone would not fit), so treat it as a typo rather than passing it through.
EXTERNAL_N_CTX = int(_num("external_n_ctx",
                                  _HW.get("EXTERNAL_N_CTX", 2048), minimum=256))

# =====================================================================
# Speech-to-Text (Whisper) Settings
# =====================================================================
# NOTE: Whisper STT is currently disabled — main.py neither loads nor warms up
# STTManager (transcription is done by Wav2Vec2 in pronunciation/acoustic/). The settings
# below are kept so stt.py can be re-enabled without changes.
WHISPER_MODEL = "small"
WHISPER_BEAM_SIZE = 1         # Beam size 1 provides optimal speed at temperature 0.0
WHISPER_NO_SPEECH_THRESHOLD = 0.45
WHISPER_CPU_THREADS = _HW.get("WHISPER_CPU_THREADS", 4)  # CPU inference threads

# Context conditioning instruction guiding Whisper's spelling logic
WHISPER_INITIAL_PROMPT = (
    f"This is a conversation with a {TARGET_LANGUAGE} tutor. "
    f"The speaker is practicing simple phrases."
)

# =====================================================================
# English Accent
# =====================================================================
# Target English accent, read from settings.json ("english_accent") at startup.
# Changing it requires a restart: the Kokoro pipeline language, the
# default/selectable voices, and the espeak language used to build the
# reference phonemes are all wired from this profile at load time (there is no
# runtime switch).
#   "american" — General American
#   "british"  — British (Received Pronunciation)
ENGLISH_ACCENT = _USER.get("english_accent", "american")

# Per-accent settings. Voice prefixes: 'af_'/'bf_' = female, 'am_'/'bm_' = male
# (a = American, b = British). A voice's data is downloaded on first use and
# cached locally. The espeak_language must match the accent so pronunciation is
# scored against phonemes of the same dialect (see pronunciation/acoustic/speech.py).
_ACCENT_PROFILES = {
    "american": {
        "kokoro_lang_code": "a",
        "espeak_language": "en-us",
        "default_voice": "af_heart",
        "voices": [
            "af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky",
            "am_adam", "am_michael", "am_echo", "am_eric", "am_liam",
        ],
    },
    "british": {
        "kokoro_lang_code": "b",
        "espeak_language": "en-gb",
        "default_voice": "bf_emma",
        "voices": [
            "bf_emma", "bf_alice", "bf_isabella", "bf_lily",
            "bm_george", "bm_daniel", "bm_fable", "bm_lewis",
        ],
    },
}

# A typo in the hand-edited settings.json must not crash startup — warn and
# fall back to the default accent instead.
# (isinstance guards the dict lookup: an unhashable value such as a list
# would otherwise raise TypeError instead of falling back.)
if not isinstance(ENGLISH_ACCENT, str) or ENGLISH_ACCENT not in _ACCENT_PROFILES:
    print(f"[config] settings.json: unknown english_accent {ENGLISH_ACCENT!r} "
          f"(expected one of {sorted(_ACCENT_PROFILES)}); using 'american'",
          file=sys.stderr)
    ENGLISH_ACCENT = "american"
_ACCENT = _ACCENT_PROFILES[ENGLISH_ACCENT]

# espeak language code ("en-us"/"en-gb") used by the pronunciation analyzer to
# phonemize the reference text. Read by pronunciation/acoustic/speech.py.
ESPEAK_LANGUAGE = _ACCENT["espeak_language"]

# =====================================================================
# Text-to-Speech (Kokoro) Settings
# =====================================================================
# Derived from the selected accent above ('a' = American, 'b' = British).
KOKORO_LANG_CODE = _ACCENT["kokoro_lang_code"]

# Default voice: the accent's default, unless settings.json names another voice
# of the same accent ("voice": null keeps the accent default). A voice from the
# other accent is rejected — it would not match KOKORO_LANG_CODE.
KOKORO_VOICE = _ACCENT["default_voice"]
_user_voice = _USER.get("voice")
if _user_voice is not None:
    if _user_voice in _ACCENT["voices"]:
        KOKORO_VOICE = _user_voice
    else:
        print(f"[config] settings.json: voice {_user_voice!r} is not a known "
              f"{ENGLISH_ACCENT} voice; using {KOKORO_VOICE!r}", file=sys.stderr)

# Voices the user can pick from in the UI. All belong to the same lang_code, so
# switching between them needs no pipeline reload.
KOKORO_VOICES = _ACCENT["voices"]

# Reference playback speed ("reference_speed"), selected in the UI and
# persisted on change. The UI selector is built from these choices, so the
# valid values and the visible options can never drift apart.
REFERENCE_SPEED_CHOICES = (1.0, 0.9, 0.8)
REFERENCE_SPEED = float(_num("reference_speed", 1.0))
if REFERENCE_SPEED not in REFERENCE_SPEED_CHOICES:
    print(f"[config] settings.json: reference_speed must be one of "
          f"{REFERENCE_SPEED_CHOICES}, got {REFERENCE_SPEED!r}; using 1.0",
          file=sys.stderr)
    REFERENCE_SPEED = 1.0

# =====================================================================
# Pronunciation Analysis (Wav2Vec2) Settings
# =====================================================================
# Pronunciation engine selection. Set here in code (not exposed in settings.json),
# since it's a developer/build choice rather than an end-user preference:
#   "acoustic" -> pronunciation/acoustic/  (Wav2Vec2 embeddings + cosine-DTW)
#   "phoneme"  -> pronunciation/phoneme/   (espeak reference + phoneme ASR + edit distance)
# The dispatcher (mimora/engine.py) binds one backend at startup; only that engine's
# models are loaded.
ENGINE = "phoneme"

# =====================================================================
# Acoustic + transcription model used by the pronunciation/acoustic/ module.
WAV2VEC2_MODEL_NAME = "facebook/wav2vec2-large-960h"
# Phoneme-ASR model used by the pronunciation/phoneme/ module: a wav2vec2 CTC model that
# emits espeak-style IPA, so its phone inventory matches the espeak reference.
WAV2VEC2_PHONEME_MODEL_NAME = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
# Device for Wav2Vec2. Defaults to the shared DEVICE; hwconfig may pin it to
# "cpu" to avoid VRAM contention with llama_cpp / Kokoro on a single GPU.
WAV2VEC2_DEVICE = _HW.get("WAV2VEC2_DEVICE") or DEVICE
# Score (0-100) at or above which a repetition is accepted; below it the learner
# is asked to repeat the same phrase.
PRONUNCIATION_SCORE_THRESHOLD = float(
    _num("pronunciation_score_threshold", 70.0, minimum=0, maximum=100)
)
# Acoustic floor: typical per-step cosine DTW distance of a *good* attempt
# (the user's voice never matches the TTS voice exactly, so this is > 0). This
# is only the pre-calibration default — after a practice session run
# ``python pronunciation/acoustic/calibrate.py`` and the value it writes to
# pronunciation/acoustic/calibration.json takes precedence (tuned to your voice and mic).
PRONUNCIATION_ACOUSTIC_GOOD = 0.20

# =====================================================================
# Practice Text & Phrase Generation
# =====================================================================
# Source text shown in the input panel at startup. The LLM builds practice
# phrases from whatever text the user has in that panel. Read from
# settings.json ("practice_text_file"); a relative path resolves against the
# project root.
PRACTICE_TEXT_FILE = _path("practice_text_file",
                                BASE_DIR / "texts" / "practice_text.txt")

# One short phrase is generated per request (non-streaming).
PHRASE_GEN_TEMPERATURE = 0.7
PHRASE_GEN_MAX_TOKENS = 40
PHRASE_GEN_SYSTEM_PROMPT = (
    "You generate short English sentences for pronunciation practice. "
    "Reply with exactly ONE natural spoken sentence of 4 to 8 words, easy to read aloud. "
    "Use only plain words and a single final period — no quotation marks, no numbering, "
    "no extra commentary. Base the sentence on the topic and vocabulary of the text the user provides."
)
# Sliding window over the source text: instead of sending the whole text with
# every request (which makes a small model converge on one "most likely"
# sentence), only PHRASE_GEN_WINDOW_SENTENCES consecutive sentences are sent.
# After PHRASE_GEN_WINDOW_REPEATS generations the window advances by half its
# size (so consecutive windows overlap and the topic shifts gradually) and
# wraps around at the end of the text. int() because the values are used for
# list slicing and modular arithmetic.
PHRASE_GEN_WINDOW_SENTENCES = int(_num("phrase_gen_window_sentences", 5,
                                               minimum=1))
PHRASE_GEN_WINDOW_REPEATS = int(_num("phrase_gen_window_repeats", 5,
                                             minimum=1))

# "Few words" mode: generate a short 2-4 word fragment instead of a complete
# sentence (e.g. "give me", "on the table", "where are you from"). Uses its own
# system prompt and a tighter token budget.
PHRASE_GEN_FRAGMENT_SYSTEM_PROMPT = (
    "You generate very short English fragments for pronunciation practice. "
    "Reply with exactly ONE natural fragment of 2 to 4 words that is NOT a complete "
    "sentence — such as a sentence opening, a prepositional phrase, or the start of a "
    "question. Examples: give me; on the table; where are you from. "
    "Use only plain lowercase words — no final period, no quotation marks, no numbering, "
    "no extra commentary. Base the fragment on the topic and vocabulary of the text the user provides."
)
PHRASE_GEN_FRAGMENT_MAX_TOKENS = 16

# Phrase length selected in the UI ("phrase_length"), persisted on change:
# "full" → complete sentence, "fragment" → 2-4 word fragment (see
# LLMManager.generate_phrase).
PHRASE_LENGTH = _USER.get("phrase_length", "full")
if PHRASE_LENGTH not in ("full", "fragment"):
    print(f"[config] settings.json: phrase_length must be 'full' or "
          f"'fragment', got {PHRASE_LENGTH!r}; using 'full'", file=sys.stderr)
    PHRASE_LENGTH = "full"

# =====================================================================
# Prosody panel (UI state)
# =====================================================================
# Visibility of the two prosody charts, toggled by their title checkboxes in
# the app window and persisted on change.
SHOW_PITCH_CHART = _bool("show_pitch_chart", True)
SHOW_ENERGY_CHART = _bool("show_energy_chart", True)

# Visibility of the articulation face shown beside the prosody charts, toggled
# by the "Face" checkbox in the prosody header and persisted on change.
SHOW_FACE = _bool("show_face", True)

# =====================================================================
# Color Theme (UI palette)
# =====================================================================
# UI colors, read from settings.json ("color_theme") at startup; changing the
# theme requires a restart. Each theme lives in themes/<name>_schema.json as a
# flat map of semantic color names to hex values, so adding a theme is just
# adding a file. The built-in palette below doubles as the complete list of
# valid keys and as the fallback: a missing or broken schema file, or a missing
# key inside one, falls back to these values, so the app always starts with a
# usable (dark) palette.
_DARK_THEME = {
    # Surfaces
    "bg_main": "#121214",           # window background
    "bg_panel": "#1a1a1e",          # panels, text inputs, status bar
    "bg_accent": "#1f1430",         # accent-tinted controls (buttons, combobox)
    "bg_accent_active": "#2a1a45",  # hovered/active accent controls
    "border": "#25252a",
    "accent": "#8a2be2",            # brand purple: titles, focus highlights
    # Text
    "text": "#f8f8f2",
    "text_emph": "#f1f1f6",         # emphasized feedback text
    "text_bright": "#ffffff",       # selection fg, text cursor
    "text_dim": "#a0a0a5",          # secondary labels
    "text_muted": "#6272a4",        # tertiary/system text
    "text_accent": "#d6c2ff",       # text on accent-tinted controls
    "text_disabled": "#555560",
    "text_disabled_dim": "#3a3a40",
    # Status / feedback ("reference" marks everything tied to the reference
    # audio: status texts and the reference prosody curve)
    "good": "#50fa7b",
    "ready": "#00e676",
    "bad": "#ff5555",
    "warn": "#ffb86c",
    "info": "#8be9fd",
    "reference": "#ff79c6",
    # Mic button per-state inner fill (outlines reuse accent/bad/warn/good)
    "mic_loading_bg": "#1e1e24",
    "mic_loading_outline": "#44475a",
    "mic_recording_bg": "#3a0c10",
    "mic_processing_bg": "#36220f",
    "mic_speaking_bg": "#0f2c1d",
}

COLOR_THEME = _USER.get("color_theme", "dark")
if not isinstance(COLOR_THEME, str) or not COLOR_THEME.strip():
    print(f"[config] settings.json: color_theme must be a non-empty string, "
          f"got {COLOR_THEME!r}; using 'dark'", file=sys.stderr)
    COLOR_THEME = "dark"

# Resolved palette consumed by ui.py / main.py. Starts as a copy of the
# built-in dark palette so every key is always present.
THEME = dict(_DARK_THEME)

_THEME_FILE = CONFIG_DIR / "themes" / f"{COLOR_THEME}_schema.json"
_SCHEMA = loader.read_json(_THEME_FILE)
if not _SCHEMA:
    # For "dark" a missing file is fine — the built-in palette IS dark.
    if COLOR_THEME != "dark":
        print(f"[config] theme file {_THEME_FILE.name} is missing or invalid; "
              f"using the built-in dark palette", file=sys.stderr)
else:
    for _key, _value in _SCHEMA.items():
        if _key.startswith("_"):
            continue  # comment keys, same convention as settings.json
        if _key not in _DARK_THEME:
            print(f"[config] {_THEME_FILE.name}: unknown color {_key!r} ignored",
                  file=sys.stderr)
        elif isinstance(_value, str) and _value.strip():
            THEME[_key] = _value
        else:
            print(f"[config] {_THEME_FILE.name}: {_key} must be a color string, "
                  f"got {_value!r}; using {_DARK_THEME[_key]!r}", file=sys.stderr)
    _missing = sorted(set(_DARK_THEME) - set(_SCHEMA))
    if _missing:
        print(f"[config] {_THEME_FILE.name}: missing colors filled from the "
              f"built-in dark palette: {', '.join(_missing)}", file=sys.stderr)

# =====================================================================
# Shared Audio Device Settings
# =====================================================================
# Single lock coordinates PortAudio access between the mic (main.py) and
# speaker (tts.py) streams. Both modules import this object — do not create
# separate Lock instances or they will not mutually exclude each other.
AUDIO_LOCK = threading.Lock()

# Pipeline sample rate: mic captures are downsampled to this rate, and it is
# what playback of recordings and Wav2Vec2 pronunciation analysis expect.
# (Equals the 16 kHz Whisper would also require if STT were re-enabled.)
AUDIO_SAMPLE_RATE = 16_000

AUDIO_CHANNELS = 1           # Mono for both recording and playback
AUDIO_LATENCY = None         # None → OS default shared-mode latency
# Device indices come from hwconfig; None → OS default microphone/speaker.
AUDIO_INPUT_DEVICE = _HW.get("AUDIO_INPUT_DEVICE")
AUDIO_OUTPUT_DEVICE = _HW.get("AUDIO_OUTPUT_DEVICE")

# =====================================================================
# Logging Settings
# =====================================================================
# All log files live in a dedicated logs/ directory (created on import so the
# handlers can open their files immediately).
LOG_DIR = BASE_DIR / "logs"
loader.ensure_dir(LOG_DIR)
LOG_FILE = str(LOG_DIR / "main.log")
# Log file for the auto-started local LLM server subprocess (see main.py).
LLM_SERVER_LOG_FILE = str(LOG_DIR / "llm_server.log")
