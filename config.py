import json
import os
import sys
import threading
from pathlib import Path

# Base directory — always absolute, regardless of working directory at launch.
BASE_DIR = Path(__file__).parent

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


def _read_json(path: Path) -> dict:
    """Parse a JSON object from *path*; returns {} when absent or invalid."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[config] cannot read {path.name} ({exc}); using defaults",
              file=sys.stderr)
        return {}
    if not isinstance(data, dict):
        print(f"[config] {path.name} must contain a JSON object; using defaults",
              file=sys.stderr)
        return {}
    return data


# Machine-derived overrides. Only the "config" section is consumed here;
# the "hardware" section is diagnostics for humans.
_HW = _read_json(BASE_DIR / "hwconfig" / "hardware_config.json").get("config")
if not isinstance(_HW, dict):
    _HW = {}

# User preferences. Keys starting with "_" are skipped so they can serve as
# comments (plain JSON has no comment syntax).
_USER = _read_json(BASE_DIR / "settings.json")

_KNOWN_USER_KEYS = {
    "english_accent",
    "voice",
    "color_theme",
    "pronunciation_score_threshold",
    "max_record_seconds",
    "llm_backend",
    "external_model_path",
    "practice_text_file",
    "phrase_gen_recent_memory",
}
for _key in _USER:
    if not _key.startswith("_") and _key not in _KNOWN_USER_KEYS:
        print(f"[config] settings.json: unknown key {_key!r} ignored",
              file=sys.stderr)


def _user_number(key: str, default):
    """Numeric setting from settings.json; *default* on a non-numeric value."""
    value = _USER.get(key, default)
    # bool is a subclass of int — exclude it so `true` is not accepted silently.
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    print(f"[config] settings.json: {key} must be a number, got {value!r}; "
          f"using {default}", file=sys.stderr)
    return default


def _user_path(key: str, default: Path) -> str:
    """Path setting from settings.json; *default* on a non-string value.

    A relative value is resolved against BASE_DIR (pathlib keeps an absolute
    value as-is when joined), so settings.json works regardless of the working
    directory at launch.
    """
    value = _USER.get(key)
    if value is None:
        return str(default)
    if isinstance(value, str) and value.strip():
        return str(BASE_DIR / value)
    print(f"[config] settings.json: {key} must be a non-empty string, got "
          f"{value!r}; using {default}", file=sys.stderr)
    return str(default)

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
# project module imported by main.py (before stt/tts/pronounce), so it is early
# enough. We use setdefault() so an externally set HF_HOME is respected.
MODEL_CACHE_DIR = BASE_DIR / "model_cache"
MODEL_CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("HF_HOME", str(MODEL_CACHE_DIR))

# Repo IDs whose presence in the cache means "fully downloaded".
# Whisper (Systran/faster-whisper-small) is intentionally absent: STT is
# disabled (stt.py is kept but never loaded), so its cache must not gate
# offline mode.
_CACHED_REPOS = (
    "hexgrad/Kokoro-82M",             # Kokoro TTS (model + voice files)
    "facebook/wav2vec2-large-960h",   # Wav2Vec2 pronunciation model
)


def _all_models_cached() -> bool:
    """True only when every networked model already exists in the local cache.

    Besides a non-empty snapshots dir, the blobs dir must hold no *.incomplete
    files — those are partial downloads left by an interrupted first run, and
    flipping to offline mode with one present would crash model loading.
    """
    hub_dir = Path(os.environ["HF_HOME"]) / "hub"
    for repo in _CACHED_REPOS:
        repo_dir = hub_dir / ("models--" + repo.replace("/", "--"))
        snapshots = repo_dir / "snapshots"
        if not snapshots.is_dir() or not any(snapshots.iterdir()):
            return False
        if any(repo_dir.glob("blobs/*.incomplete")):
            return False
    return True


# Auto offline: the first run downloads online; once everything is cached, every
# later run loads straight from disk with no network access. Delete model_cache/
# to force a fresh download.
if _all_models_cached():
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
MAX_RECORD_SECONDS = _user_number("max_record_seconds", 20)

# Hardware Acceleration setup — the value detected by hwconfig wins; otherwise
# probe torch directly (wrapped so startup does not crash if torch is absent).
DEVICE = _HW.get("DEVICE")
if DEVICE not in ("cuda", "cpu"):
    try:
        import torch
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        DEVICE = "cpu"

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
EXTERNAL_MODEL_PATH = _user_path(
    "external_model_path",
    BASE_DIR / "models" / "llama-3.2-3b-instruct-q4_k_m.gguf",
)
# GPU offload and context size: hwconfig picks values matched to the detected
# VRAM/RAM; the literals here are conservative fallbacks for unknown hardware.
EXTERNAL_N_GPU_LAYERS = _HW.get("EXTERNAL_N_GPU_LAYERS", 20)  # layers on GPU (-1 = all)
EXTERNAL_N_CTX = _HW.get("EXTERNAL_N_CTX", 2048)              # context window size

# =====================================================================
# Speech-to-Text (Whisper) Settings
# =====================================================================
# NOTE: Whisper STT is currently disabled — main.py neither loads nor warms up
# STTManager (transcription is done by Wav2Vec2 in pronounce/). The settings
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
# scored against phonemes of the same dialect (see pronounce/speech.py).
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
# phonemize the reference text. Read by pronounce/speech.py.
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

# =====================================================================
# Pronunciation Analysis (Wav2Vec2) Settings
# =====================================================================
# Acoustic + transcription model used by the pronounce/ module.
WAV2VEC2_MODEL_NAME = "facebook/wav2vec2-large-960h"
# Device for Wav2Vec2. Defaults to the shared DEVICE; hwconfig may pin it to
# "cpu" to avoid VRAM contention with llama_cpp / Kokoro on a single GPU.
WAV2VEC2_DEVICE = _HW.get("WAV2VEC2_DEVICE") or DEVICE
# Score (0-100) at or above which a repetition is accepted; below it the learner
# is asked to repeat the same phrase.
PRONUNCIATION_SCORE_THRESHOLD = float(
    _user_number("pronunciation_score_threshold", 70.0)
)
# Acoustic floor: typical per-step cosine DTW distance of a *good* attempt
# (the user's voice never matches the TTS voice exactly, so this is > 0). This
# is only the pre-calibration default — after a practice session run
# ``python pronounce/calibrate.py`` and the value it writes to
# pronounce/calibration.json takes precedence (tuned to your voice and mic).
PRONUNCIATION_ACOUSTIC_GOOD = 0.20

# =====================================================================
# Practice Text & Phrase Generation
# =====================================================================
# Source text shown in the input panel at startup. The LLM builds practice
# phrases from whatever text the user has in that panel. Read from
# settings.json ("practice_text_file"); a relative path resolves against the
# project root.
PRACTICE_TEXT_FILE = _user_path("practice_text_file", BASE_DIR / "practice_text.txt")

# One short phrase is generated per request (non-streaming).
PHRASE_GEN_TEMPERATURE = 0.7
PHRASE_GEN_MAX_TOKENS = 40
PHRASE_GEN_SYSTEM_PROMPT = (
    "You generate short English sentences for pronunciation practice. "
    "Reply with exactly ONE natural spoken sentence of 4 to 8 words, easy to read aloud. "
    "Use only plain words and a single final period — no quotation marks, no numbering, "
    "no extra commentary. Base the sentence on the topic and vocabulary of the text the user provides."
)
# How many recently used phrases to send back to the model so it avoids
# repeats. int() because the value is used for list slicing.
PHRASE_GEN_RECENT_MEMORY = int(_user_number("phrase_gen_recent_memory", 5))

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

_THEME_FILE = BASE_DIR / "themes" / f"{COLOR_THEME}_schema.json"
_SCHEMA = _read_json(_THEME_FILE)
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
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = str(LOG_DIR / "main.log")
# Log file for the auto-started local LLM server subprocess (see main.py).
LLM_SERVER_LOG_FILE = str(LOG_DIR / "llm_server.log")
