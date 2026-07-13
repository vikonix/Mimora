# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev

import os
import sys
import threading
from functools import partial
from pathlib import Path

from . import loader

# Project root - always absolute, regardless of working directory at launch.
# This file lives in mimora/, so the root is one level up.
BASE_DIR = Path(__file__).resolve().parent.parent
# Hand-edited configuration data (settings.json, themes/) lives here.
CONFIG_DIR = BASE_DIR / "config"

# =====================================================================
# Settings files (optional overrides)
# =====================================================================
# Values in this module are layered, lowest priority first:
#   1. built-in defaults - the literals in this file;
#   2. config/hardware_config.json, "config" section - machine-derived
#      values written by `python tools/detect_hardware.py`;
#   3. settings.json - hand-edited user preferences.
# A missing or broken file simply leaves the lower layers in effect: problems
# are reported to stderr instead of crashing startup, because both files are
# optional and settings.json is edited by hand.


# Machine-derived overrides. Only the "config" section is consumed here;
# the "hardware" section is diagnostics for humans.
_HW = loader.read_json(CONFIG_DIR / "hardware_config.json").get("config")
if not isinstance(_HW, dict):
    _HW = {}

# User preferences. Keys starting with "_" are skipped so they can serve as
# comments (plain JSON has no comment syntax).
_USER = loader.read_json(CONFIG_DIR / "settings.json")

_KNOWN_USER_KEYS = {
    "engine",
    "practice_language",
    "accent",
    # Legacy: superseded by practice_language + accent. Still honored as a read
    # fallback for the variant so an existing settings.json keeps working; the
    # settings window migrates it to the new keys in a later stage.
    "english_accent",
    "voice",
    "color_theme",
    "pronunciation_score_threshold",
    "phoneme_good_mode",
    "max_record_seconds",
    "llm_backend",
    "external_model_path",
    "external_n_ctx",
    "practice_text_file",
    "practice_text_collapsed",
    "phrase_gen_window_sentences",
    "phrase_gen_window_repeats",
    "user_name",
    "phrase_length",
    "reference_speed",
    "random_voice",
    "playback_own_recording",
    "save_recordings",
    "show_prosody",
    "show_face",
    "silence_timeout",
    "silence_threshold",
    "translation_language",
    "warm_up",
}
for _key in _USER:
    if not _key.startswith("_") and _key not in _KNOWN_USER_KEYS:
        print(f"[config] settings.json: unknown key {_key!r} ignored",
              file=sys.stderr)

# Built-in default for every user key, mirroring the fallback literals used
# throughout this module (keep the two in sync - tests/test_settings_fields.py
# pins the key set). Consumed by the settings window's "Default" reset, which
# removes all overrides from settings.json and applies these live. None means
# "no fixed literal": voice follows the accent's default voice, and
# external_n_ctx is machine-derived by hardware detection (restart applies it).
USER_SETTING_DEFAULTS = {
    "engine": "phoneme",
    "practice_language": "english",
    "accent": "american",
    # Legacy alias of "accent" (see _KNOWN_USER_KEYS); kept in sync with it while
    # the settings window still writes the old key.
    "english_accent": "american",
    "voice": None,
    "color_theme": "dark",
    "pronunciation_score_threshold": 70.0,
    "phoneme_good_mode": "global",
    "max_record_seconds": 20,
    "llm_backend": "local_server",
    "external_model_path": "models/llama-3.2-3b-instruct-q4_k_m.gguf",
    "external_n_ctx": None,
    "practice_text_file": "texts/practice_text.txt",
    "practice_text_collapsed": True,
    "phrase_gen_window_sentences": 5,
    "phrase_gen_window_repeats": 5,
    "user_name": "",
    "phrase_length": "full",
    "reference_speed": 1.0,
    "random_voice": False,
    "playback_own_recording": True,
    "save_recordings": False,
    "show_prosody": False,
    "show_face": True,
    "silence_timeout": 3.0,
    "silence_threshold": 0.01,
    "translation_language": "",
    "warm_up": False,
}


# Validated accessors for settings.json values. The loader functions are pure
# (they take the parsed dict as their first argument); binding _USER - and
# BASE_DIR for path resolution - here keeps every call site below short.
_num = partial(loader.user_number, _USER)
_path = partial(loader.user_path, _USER, BASE_DIR)
_bool = partial(loader.user_bool, _USER)


def user_setting(key: str, fallback):
    """The value currently stored in settings.json for *key*, else *fallback*.

    save_user_setting keeps the in-memory _USER view current, so this reflects
    every change made during the session - including restart-only settings the
    running constants do not pick up. The settings window reads through this so
    a reopened window shows what is saved, not what this run started with.

    Falls back on None ("voice": null means "accent default") and on a value
    whose type does not match the fallback's - a hand-edited settings.json can
    hold anything, and the caller expects the fallback's type (mirrors the
    per-type checks the loader helpers apply to the constants).
    """
    value = _USER.get(key)
    if value is None:
        return fallback
    if isinstance(fallback, bool):
        return value if isinstance(value, bool) else fallback
    if isinstance(fallback, (int, float)):
        # bool is a subclass of int - exclude it, as loader.user_number does.
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value
        return fallback
    if isinstance(fallback, str):
        return value if isinstance(value, str) else fallback
    return value


def reset_user_settings() -> bool:
    """Remove every known user key from settings.json ("Default" reset).

    The overrides are gone from both the file and the in-memory _USER view, so
    user_setting() immediately starts reporting fallbacks and the next start
    runs on the built-in defaults plus hardware detection. Returns True on
    success.
    """
    return loader.reset_settings(CONFIG_DIR / "settings.json",
                                 _KNOWN_USER_KEYS, _USER)


def default_user_settings() -> dict:
    """Resolved built-in defaults, ready to apply live after a reset.

    Resolves the None placeholders in USER_SETTING_DEFAULTS where possible
    (voice -> the default variant's default voice) and makes the path defaults
    absolute; keys with no resolvable literal (external_n_ctx) are omitted -
    hardware detection supplies them on the next start.
    """
    values = {key: value for key, value in USER_SETTING_DEFAULTS.items()
              if value is not None}
    values["voice"] = accent_default_voice(USER_SETTING_DEFAULTS["accent"])
    values["practice_text_file"] = str(BASE_DIR / values["practice_text_file"])
    values["external_model_path"] = str(BASE_DIR / values["external_model_path"])
    return values


def save_user_setting(key: str, value) -> bool:
    """Write one setting back to settings.json, keeping every other key.

    Thin wrapper that binds the settings path and the in-memory _USER view to
    loader.save_setting (which preserves hand-edited and "_" comment keys, and
    reports failures instead of raising). Returns True on success.

    Writing either new language key also completes the one-time migration off
    the legacy "english_accent" key: it is only ever read as a fallback for
    "accent" (see the active-variant section), so once the new form is persisted
    the stale legacy key is dropped from the file to avoid it lingering.
    """
    path = CONFIG_DIR / "settings.json"
    ok = loader.save_setting(path, key, value, _USER)
    if ok and key in ("accent", "practice_language") and "english_accent" in _USER:
        loader.reset_settings(path, {"english_accent"}, _USER)
    return ok


# Display name of the practicing user; shown in the Name field of the UI and
# written back via save_user_setting("user_name", ...) when edited.
USER_NAME = _USER.get("user_name", "")
if not isinstance(USER_NAME, str):
    print(f"[config] settings.json: user_name must be a string, got "
          f"{USER_NAME!r}; using ''", file=sys.stderr)
    USER_NAME = ""

# =====================================================================
# Local model cache (HuggingFace) - download once, then load offline
# =====================================================================
# Kokoro and Wav2Vec2 are pulled from the HuggingFace Hub. We pin
# their cache to a project-local folder so the weights live next to the code and
# survive a cleared home-directory cache. Once every model the run actually needs
# is present we flip the Hub into offline mode, which skips the per-start network
# round-trip HF normally makes to re-validate file revisions - that check is what
# makes startup feel like it "re-downloads" on every launch.
#
# IMPORTANT: huggingface_hub / transformers read these env vars at *import* time,
# so they must be set before those libraries are imported. config is the first
# project module imported by main.py (before tts/pronunciation) and runs to the end
# during that import, so HF_HOME here and the offline flags set in the
# "Offline-mode gating" section below (after the active engine and its model name
# are known) all land early enough. We use setdefault() so an externally set
# HF_HOME is respected.
MODEL_CACHE_DIR = BASE_DIR / "model_cache"
loader.ensure_dir(MODEL_CACHE_DIR)
os.environ.setdefault("HF_HOME", str(MODEL_CACHE_DIR))

# Language configuration (LANGUAGE_PROFILES, the selected practice language and
# variant, and the derived TARGET_LANGUAGE / ESPEAK_LANGUAGE / KOKORO_* / voice
# constants) lives below, after the settings-file accessors it depends on.

# =====================================================================
# Controls
# =====================================================================
# Safety threshold to prevent infinite recording loops if a key gets physically stuck
MAX_RECORD_SECONDS = _num("max_record_seconds", 20, minimum=1)

# Recording now starts on a single press and stops on its own after the speaker
# falls silent (recorder.py runs the VAD on the capture thread). These two tune
# that auto-stop; both are read from settings.json so they can be adjusted to a
# room/mic without code changes.
#   silence_timeout   - seconds of continuous silence (after speech has begun)
#                       before the take is finalized automatically.
#   silence_threshold - RMS level (0..1) strictly above which a chunk counts as
#                       speech rather than silence. Kept low: silence is near-zero
#                       RMS, while quiet speech may only reach ~0.04, and the level
#                       is averaged over a whole capture chunk (which dilutes brief
#                       bursts) - too high a value never arms the silence timer.
#                       Raise it only if a noisy room keeps the take from stopping.
#                       The minimum is above zero: at 0 any real mic noise floor
#                       (rms > 0) would count as speech and the take would never
#                       auto-stop, silently pinning every recording to
#                       MAX_RECORD_SECONDS.
SILENCE_TIMEOUT = _num("silence_timeout", 3.0, minimum=0.5)
SILENCE_THRESHOLD = _num("silence_threshold", 0.01, minimum=0.001)

# Diagnostic recording dumps ("save_recordings"): when true, every take writes
# the spoken reference (model.wav), the raw mic capture (raw.wav), the
# normalized signal (normalized.wav) and the phrase text (phrase.txt) to
# records/, each overwritten per take so only the latest is kept. Off by
# default: the dumps put the user's voice on disk on every attempt, so they
# are opt-in for debugging rather than always-on.
SAVE_RECORDINGS = _bool("save_recordings", False)

# Random reference voice ("random_voice"): when true, every new phrase picks a
# fresh voice from the active accent's list (never the one just heard), instead
# of the fixed "voice" setting. The pick is ephemeral - it is never written to
# settings.json, so the chosen "voice" survives turning this off. Off by
# default. Note: a voice's data is downloaded on its first use, so the first
# pick of an unused voice delays that one generation.
RANDOM_VOICE = _bool("random_voice", False)

# Play the just-recorded take back to the user before analysis
# ("playback_own_recording"): the learner hears their own attempt right away,
# then the score follows. On by default. Turning it off skips straight to
# analysis (useful on slow machines, or when the extra playback is unwanted);
# the manual "listen to your recording" control is unaffected either way.
PLAYBACK_OWN_RECORDING = _bool("playback_own_recording", True)

# Model warm-up at startup ("warm_up"): when true every loaded model runs one
# dummy pass right after loading, so the first real take pays no first-call
# latency. Off by default: the passes lengthen startup, and on slow machines
# the wait up front hurts more than a slower first take - the cost simply
# moves to the first take (and, for the phoneme engine, to the first
# panphon/espeak use), it is never paid twice.
WARM_UP = _bool("warm_up", False)

# Hardware Acceleration setup - the value detected by hardware detection wins;
# otherwise loader.detect_device probes torch directly (and does not import torch
# at all when hardware detection already supplied a valid device, so this stays
# cheap).
DEVICE = loader.detect_device(_HW.get("DEVICE"))

# =====================================================================
# LLM Backend Settings
# =====================================================================
# Backend selection, read from settings.json ("llm_backend"):
#   "lm-studio"    - external LM Studio app (must be running separately)
#   "local_server" - llm_server.py started automatically as a subprocess
LLM_BACKEND_CHOICES = ("local_server", "lm-studio")
LLM_BACKEND = _USER.get("llm_backend", "local_server")
if LLM_BACKEND not in LLM_BACKEND_CHOICES:
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
# GGUF Model Settings (used by the "local_server" backend)
# =====================================================================
# GGUF file, read from settings.json ("external_model_path"); a relative path
# resolves against the project root.
EXTERNAL_MODEL_PATH = _path(
    "external_model_path",
    BASE_DIR / "models" / "llama-3.2-3b-instruct-q4_k_m.gguf",
)
# GPU offload and context size: hardware detection picks values matched to the
# detected VRAM/RAM; the literals here are conservative fallbacks for unknown
# hardware.
EXTERNAL_N_GPU_LAYERS = _HW.get("EXTERNAL_N_GPU_LAYERS", 20)  # layers on GPU (-1 = all)
# Context window size (n_ctx). settings.json ("external_n_ctx") wins over the
# detected value, per the usual layering. int() because the value is
# passed to the server as a command-line argument (a float would break it).
# minimum=256: anything below breaks generation outright (the system prompt
# alone would not fit), so treat it as a typo rather than passing it through.
EXTERNAL_N_CTX = int(_num("external_n_ctx",
                                  _HW.get("EXTERNAL_N_CTX", 2048), minimum=256))

# =====================================================================
# Language & variant profiles
# =====================================================================
# The practice language is DATA, not an assumption spread through the code.
# Every language is one entry in LANGUAGE_PROFILES; adding a language is adding
# a record here (plus engine calibration), never a new `if language == ...`.
#
# A profile describes:
#   display_name       - shown in the window title and settings;
#   flores_code        - FLORES-200 source code for the NLLB translator
#                        (mimora/translator.py), e.g. "eng_Latn";
#   default_variant    - the variant used when settings.json names none;
#   engines            - pronunciation engines available for this language.
#                        Availability is a language property: the acoustic
#                        engine is English-only ASR, so a non-English profile
#                        omits it (see available_engines);
#   practice_text_file - default source text, relative to the project root;
#   variants           - the former "accents": a display key -> Kokoro/espeak
#                        wiring. Voice prefixes: 'af_'/'bf_' = female,
#                        'am_'/'bm_' = male (a = American, b = British). A
#                        voice's data is downloaded on first use and cached
#                        locally; espeak_language must match the variant so
#                        pronunciation is scored against phonemes of the same
#                        dialect (see pronunciation/phoneme/speech.py).
#
# Besides the wiring above, a profile also carries the language-specific text
# the app used to hardcode: the phrase-generation prompts and asks, the
# voice-preview phrase and the translator warm-up text (see the english entry).
# Only English is defined today (variants american/british); Spanish and the
# code-ready future languages arrive as further entries.
LANGUAGE_PROFILES = {
    "english": {
        "display_name": "English",
        "flores_code": "eng_Latn",
        "default_variant": "american",
        "engines": ("phoneme", "acoustic", "none"),
        "practice_text_file": "texts/practice_text.txt",
        # Phrase-generation prompts (mimora/llm.py). The instructions stay in
        # English - they steer the model - while the TARGET language of the
        # generated phrase is named inside each string, so a new language ships
        # its own prompts here rather than a code branch. "system" is the full
        # sentence prompt, "fragment_system" the 2-4 word fragment prompt;
        # "full_ask"/"fragment_ask" are the per-request user asks.
        "phrase_gen": {
            "system": (
                "You generate short English sentences for pronunciation practice. "
                "Reply with exactly ONE natural spoken sentence of 4 to 8 words, easy to read aloud. "
                "Use only plain words and a single final period - no quotation marks, no numbering, "
                "no extra commentary. Output ONLY the sentence itself, with nothing before or after it: "
                "do not add a lead-in such as 'Here's a sentence' or 'Sure', and never put a colon before "
                "the sentence. Base the sentence on the topic and vocabulary of the text the user provides."
            ),
            "fragment_system": (
                "You generate very short English fragments for pronunciation practice. "
                "Reply with exactly ONE natural fragment of 2 to 4 words that is NOT a complete "
                "sentence - such as a sentence opening, a prepositional phrase, or the start of a "
                "question. Examples: give me; on the table; where are you from. "
                "Use only plain lowercase words - no final period, no quotation marks, no numbering, "
                "no extra commentary. Output ONLY the fragment itself, with nothing before or after it: "
                "do not add a lead-in such as 'Here's a fragment' or 'Sure', and never put a colon before "
                "the fragment. Base the fragment on the topic and vocabulary of the text the user provides."
            ),
            "full_ask": (
                "Give me ONE short English sentence to practice pronunciation, "
                "based on this text."
            ),
            "fragment_ask": (
                "Give me ONE short English fragment of 2 to 4 words (NOT a complete "
                "sentence) to practice pronunciation, based on this text."
            ),
        },
        # Spoken by the voice-preview button (settings_window.py); short and
        # phonetically varied, in the language being practiced.
        "preview_phrase": "Hello! This is how I sound. Let's practice together.",
        # Throwaway text to prime the NLLB translator's source tokenizer
        # (mimora/translator.py warm_up); any short phrase in the source language.
        "translator_warmup": "Hello.",
        "variants": {
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
        },
    },
}


def language_choices() -> tuple:
    """Practice-language keys selectable in the UI, in definition order."""
    return tuple(LANGUAGE_PROFILES)


def _variants_of(language: str) -> dict:
    """Variant map of *language*, or {} for an unknown language."""
    profile = LANGUAGE_PROFILES.get(language)
    return profile["variants"] if profile else {}


def available_engines(language: str = None) -> tuple:
    """Pronunciation engines available for *language* (default: the active one).

    Engine availability is a property of the language profile: a non-English
    language omits the English-only acoustic engine, so the settings window can
    offer only the engines that actually work for the chosen language.
    """
    profile = LANGUAGE_PROFILES.get(language or PRACTICE_LANGUAGE)
    return tuple(profile["engines"]) if profile else ()


def accent_choices(language: str = None) -> tuple:
    """Variant names of *language* (default: the active one), in definition order.

    The optional *language* lets the settings window list the variants of the
    language currently selected in its Language field - which may differ from
    the running one until a restart. Called with no argument it answers for the
    active language, keeping the existing single-argument call sites working.
    """
    return tuple(_variants_of(language or PRACTICE_LANGUAGE))


def default_accent(language: str = None) -> str:
    """Default variant of *language* (default: the active one), or "" if unknown.

    Used by the settings window when the language selector changes: the variant
    and voice must be reset to the newly selected language's defaults.
    """
    profile = LANGUAGE_PROFILES.get(language or PRACTICE_LANGUAGE)
    return profile["default_variant"] if profile else ""


def accent_voices(accent: str, language: str = None) -> tuple:
    """Voices of *accent* within *language* (default: active), or () if unknown.

    Public accessor for the settings window: it lets the voice list follow the
    variant (and language) selector without reaching into LANGUAGE_PROFILES.
    """
    variant = _variants_of(language or PRACTICE_LANGUAGE).get(accent)
    return tuple(variant["voices"]) if variant else ()


def accent_default_voice(accent: str, language: str = None) -> str:
    """Default voice of *accent* within *language* (default: active), or "".

    Used when the variant or language selector changes: the persisted voice
    must belong to the persisted variant, so it is reset to that variant's
    default.
    """
    variant = _variants_of(language or PRACTICE_LANGUAGE).get(accent)
    return variant["default_voice"] if variant else ""


# --- Active practice language ----------------------------------------
# Practice language, read from settings.json ("practice_language"). A typo must
# not crash startup - warn and fall back to the default language instead.
PRACTICE_LANGUAGE = _USER.get("practice_language", "english")
if not isinstance(PRACTICE_LANGUAGE, str) \
        or PRACTICE_LANGUAGE not in LANGUAGE_PROFILES:
    print(f"[config] settings.json: unknown practice_language "
          f"{PRACTICE_LANGUAGE!r} (expected one of {sorted(LANGUAGE_PROFILES)}); "
          f"using 'english'", file=sys.stderr)
    PRACTICE_LANGUAGE = "english"
_LANG_PROFILE = LANGUAGE_PROFILES[PRACTICE_LANGUAGE]

# Display name shown in the window title and settings.
TARGET_LANGUAGE = _LANG_PROFILE["display_name"]

# --- Active variant (formerly "accent") ------------------------------
# Variant within the language, read from settings.json ("accent"); the legacy
# key "english_accent" is honored as a fallback so an existing settings.json
# keeps working. Changing the variant requires a restart: the Kokoro pipeline
# language, the default/selectable voices, and the espeak language used to build
# the reference phonemes are all wired from the variant at load time.
#   English variants: "american" (General American), "british" (RP).
_variant_map = _LANG_PROFILE["variants"]
ACCENT = _USER.get("accent", _USER.get("english_accent"))
if ACCENT is None:
    ACCENT = _LANG_PROFILE["default_variant"]
elif not isinstance(ACCENT, str) or ACCENT not in _variant_map:
    # (isinstance guards the dict lookup: an unhashable value such as a list
    # would otherwise raise TypeError instead of falling back.)
    print(f"[config] settings.json: unknown accent {ACCENT!r} for "
          f"{PRACTICE_LANGUAGE} (expected one of {sorted(_variant_map)}); "
          f"using {_LANG_PROFILE['default_variant']!r}", file=sys.stderr)
    ACCENT = _LANG_PROFILE["default_variant"]
_VARIANT = _variant_map[ACCENT]

# espeak language code ("en-us"/"en-gb"/...) used by the pronunciation analyzer
# to phonemize the reference text. Read by the pronunciation engines.
ESPEAK_LANGUAGE = _VARIANT["espeak_language"]

# =====================================================================
# Text-to-Speech (Kokoro) Settings
# =====================================================================
# Kokoro pipeline language of the active variant ('a' = American, 'b' = British).
KOKORO_LANG_CODE = _VARIANT["kokoro_lang_code"]

# Default voice: the variant's default, unless settings.json names another voice
# of the same variant ("voice": null keeps the variant default). A voice from a
# different variant is rejected - it would not match KOKORO_LANG_CODE.
KOKORO_VOICE = _VARIANT["default_voice"]
_user_voice = _USER.get("voice")
if _user_voice is not None:
    if _user_voice in _VARIANT["voices"]:
        KOKORO_VOICE = _user_voice
    else:
        print(f"[config] settings.json: voice {_user_voice!r} is not a known "
              f"{ACCENT} voice; using {KOKORO_VOICE!r}", file=sys.stderr)

# Voices the user can pick from in the UI. All belong to the same lang_code, so
# switching between them needs no pipeline reload.
KOKORO_VOICES = _VARIANT["voices"]

# Reference playback speed ("reference_speed"), chosen in Settings and
# persisted on change. The Settings control is a slider over a continuous
# range (rather than a few fixed steps), so REFERENCE_SPEED_MIN/MAX define
# both the valid bounds and the slider extent - they can never drift apart.
# REFERENCE_SPEED_STEP is the slider's snap granularity.
REFERENCE_SPEED_MIN = 0.7
REFERENCE_SPEED_MAX = 1.4
REFERENCE_SPEED_STEP = 0.05
REFERENCE_SPEED = float(_num("reference_speed", 1.0,
                             minimum=REFERENCE_SPEED_MIN,
                             maximum=REFERENCE_SPEED_MAX))

# The one-tap slow replay (the "Slow ▶" button next to Reference) plays one
# step slower than the normal reference: REFERENCE_SPEED - REFERENCE_SLOW_DELTA.
# Relative rather than fixed, so it always feels "a bit slower" whatever the
# configured speed. Floored by REFERENCE_SLOW_MIN so a very low setting cannot
# drive the effective sample rate to zero. Not user settings.
REFERENCE_SLOW_DELTA = 0.1
REFERENCE_SLOW_MIN = 0.5

# =====================================================================
# Pronunciation Analysis (Wav2Vec2) Settings
# =====================================================================
# Pronunciation engine selection, read from settings.json ("engine"). Exposed as a
# user setting so scoring can be turned off on slow machines without code edits:
#   "phoneme"  -> pronunciation/phoneme/   (espeak reference + phoneme ASR + edit
#                 distance; the default)
#   "acoustic" -> pronunciation/acoustic/  (Wav2Vec2 embeddings + cosine-DTW)
#   "none"     -> pronunciation/none/      (scoring disabled: no recognizer model is
#                 downloaded or loaded and analysis returns instantly; the learner
#                 compares the takes by ear)
# The dispatcher (mimora/engine.py) binds one backend at startup; only that engine's
# models are loaded. Changing the engine requires a restart.
ENGINE_CHOICES = ("phoneme", "acoustic", "none")
ENGINE = _USER.get("engine", "phoneme")
if not isinstance(ENGINE, str) or ENGINE not in ENGINE_CHOICES:
    print(f"[config] settings.json: unknown engine {ENGINE!r} "
          f"(expected one of {ENGINE_CHOICES}); using 'phoneme'", file=sys.stderr)
    ENGINE = "phoneme"
# Engine availability is per-language (see available_engines): an engine valid
# in general but not offered for the active language falls back to that
# language's first available engine. For English all three are available, so
# this is a no-op there.
_available_engines = available_engines(PRACTICE_LANGUAGE)
if _available_engines and ENGINE not in _available_engines:
    print(f"[config] settings.json: engine {ENGINE!r} is not available for "
          f"{PRACTICE_LANGUAGE} (available: {_available_engines}); "
          f"using {_available_engines[0]!r}", file=sys.stderr)
    ENGINE = _available_engines[0]

# GOOD-anchor mode for the phoneme engine (see pronunciation/phoneme/config.py),
# read from settings.json ("phoneme_good_mode"); only affects ENGINE == "phoneme":
#   "global"  - one calibrated PHONEME_GOOD anchor for every phrase; the 0-5
#               bucket cutpoints were fit under this anchor, so scores and
#               buckets stay consistent. The default: it needs no extra
#               recognizer pass, so analysis is ~2x faster on the first take of
#               each phrase (notably on CPU-only Intel Macs).
#   "ceiling" - per-phrase anchor = the TTS reference's own recognized per-phone
#               distance, so a flawless read maps to 100 for each phrase. Costs
#               an extra recognizer pass over the reference per phrase and shifts
#               scores away from the global anchor the buckets were calibrated for.
PHONEME_GOOD_MODE_CHOICES = ("global", "ceiling")
PHONEME_GOOD_MODE = _USER.get("phoneme_good_mode", "global")
if PHONEME_GOOD_MODE not in PHONEME_GOOD_MODE_CHOICES:
    print(f"[config] settings.json: phoneme_good_mode must be 'global' or "
          f"'ceiling', got {PHONEME_GOOD_MODE!r}; using 'global'", file=sys.stderr)
    PHONEME_GOOD_MODE = "global"

# =====================================================================
# Acoustic + transcription model used by the pronunciation/acoustic/ module.
WAV2VEC2_MODEL_NAME = "facebook/wav2vec2-large-960h"
# Phoneme-ASR model used by the pronunciation/phoneme/ module: a wav2vec2 CTC model that
# emits espeak-style IPA, so its phone inventory matches the espeak reference.
WAV2VEC2_PHONEME_MODEL_NAME = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
# Device for Wav2Vec2. Defaults to the shared DEVICE; hardware detection may pin
# it to "cpu" to avoid VRAM contention with llama_cpp / Kokoro on a single GPU.
WAV2VEC2_DEVICE = _HW.get("WAV2VEC2_DEVICE") or DEVICE
# Score (0-100) at or above which a repetition is accepted; below it the learner
# is asked to repeat the same phrase.
PRONUNCIATION_SCORE_THRESHOLD = float(
    _num("pronunciation_score_threshold", 70.0, minimum=0, maximum=100)
)
# Acoustic floor: typical per-step cosine DTW distance of a *good* attempt
# (the user's voice never matches the TTS voice exactly, so this is > 0). This
# is only the pre-calibration default - after a practice session run
# ``python pronunciation/acoustic/calibrate.py`` and the value it writes to
# pronunciation/acoustic/calibration.json takes precedence (tuned to your voice and mic).
PRONUNCIATION_ACOUSTIC_GOOD = 0.20

# =====================================================================
# Practice Text & Phrase Generation
# =====================================================================
# Source text shown in the input panel at startup. The LLM builds practice
# phrases from whatever text the user has in that panel. Read from
# settings.json ("practice_text_file"); a relative path resolves against the
# project root. The default is the active language's profile text, so each
# language ships its own starter text without a code change.
PRACTICE_TEXT_FILE = _path("practice_text_file",
                                BASE_DIR / _LANG_PROFILE["practice_text_file"])

# One short phrase is generated per request (non-streaming). Temperature and the
# token budgets are language-independent tuning; the prompts and asks are
# language text, so they come from the active profile (see LANGUAGE_PROFILES).
PHRASE_GEN_TEMPERATURE = 0.7
PHRASE_GEN_MAX_TOKENS = 40
_PHRASE_GEN = _LANG_PROFILE["phrase_gen"]
PHRASE_GEN_SYSTEM_PROMPT = _PHRASE_GEN["system"]
# Per-request user asks handed to the LLM (mimora/llm.py) for each phrase length.
PHRASE_GEN_FULL_ASK = _PHRASE_GEN["full_ask"]
PHRASE_GEN_FRAGMENT_ASK = _PHRASE_GEN["fragment_ask"]
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
# system prompt (from the profile) and a tighter token budget.
PHRASE_GEN_FRAGMENT_SYSTEM_PROMPT = _PHRASE_GEN["fragment_system"]
PHRASE_GEN_FRAGMENT_MAX_TOKENS = 16

# Voice-preview phrase (settings_window.py "Listen" button) in the practiced
# language, from the active profile.
PREVIEW_PHRASE = _LANG_PROFILE["preview_phrase"]

# Phrase length selected in the UI ("phrase_length"), persisted on change:
# "full" → complete sentence, "fragment" → 2-4 word fragment (see
# LLMManager.generate_phrase).
PHRASE_LENGTH_CHOICES = ("full", "fragment")
PHRASE_LENGTH = _USER.get("phrase_length", "full")
if PHRASE_LENGTH not in PHRASE_LENGTH_CHOICES:
    print(f"[config] settings.json: phrase_length must be 'full' or "
          f"'fragment', got {PHRASE_LENGTH!r}; using 'full'", file=sys.stderr)
    PHRASE_LENGTH = "full"

# Offline translation engine: a dedicated NLLB-200 model (mimora/translator.py),
# NOT the chat LLM - the local 3B LLM produced unusable translations (empty CJK,
# leaked English). NLLB is a 200-language MT model that is small and CPU-friendly.
# The repo is also part of _CACHED_REPOS (see the offline-mode gating below) so
# offline-mode gating waits for it; the installer pre-fetches it into model_cache/
# (see install.py).
NLLB_TRANSLATOR_MODEL_NAME = "facebook/nllb-200-distilled-600M"
# Device for the translator. Defaults to CPU on purpose: translation is
# latency-tolerant (it runs in the background after the phrase is shown and the
# reference has played), and keeping NLLB off the GPU avoids VRAM contention
# with Kokoro / Wav2Vec2 / llama_cpp - matching the translator's RAM (not VRAM) budget.
# hardware detection may pin it to "cuda" on a machine with VRAM to spare.
TRANSLATOR_DEVICE = _HW.get("TRANSLATOR_DEVICE") or "cpu"
# The translator's source language is the language being practiced: NLLB
# tokenizes the source with this FLORES-200 prefix (mimora/translator.py), and
# it comes from the active profile so it follows the practice language.
SOURCE_FLORES_CODE = _LANG_PROFILE["flores_code"]
# Text used to prime the translator's tokenizer on warm-up, in the source
# language (from the profile).
TRANSLATOR_WARMUP = _LANG_PROFILE["translator_warmup"]

# =====================================================================
# Offline-mode gating
# =====================================================================
# Auto offline: the first run downloads online; once every model the run actually
# needs is cached, every later run loads straight from disk with no network access.
# Delete model_cache/ to force a fresh download.
#
# The set of required repos is engine-aware: the always-used shared models (Kokoro
# TTS, the NLLB translator) plus the ACTIVE engine's Wav2Vec2 model only. The
# dispatcher never loads the inactive engine's weights, so requiring them would
# needlessly keep the Hub online (and waste ~1.2 GB the run never touches). The
# "none" engine has no recognizer model at all, so nothing extra is required then.
_ENGINE_MODEL_REPO = {
    "phoneme": WAV2VEC2_PHONEME_MODEL_NAME,   # default engine
    "acoustic": WAV2VEC2_MODEL_NAME,
}
# ENGINE was validated above, so a missing entry can only mean "none".
_engine_repo = _ENGINE_MODEL_REPO.get(ENGINE)
_CACHED_REPOS = (
    "hexgrad/Kokoro-82M",                     # Kokoro TTS (model + voice files)
    NLLB_TRANSLATOR_MODEL_NAME,               # NLLB-200 offline translator
) + ((_engine_repo,) if _engine_repo else ())  # active engine's recognizer, if any
# HF_HOME is read from os.environ (not MODEL_CACHE_DIR) so an externally set cache
# location is honored.
if loader.models_cached(Path(os.environ["HF_HOME"]) / "hub", _CACHED_REPOS):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Translation panel: language the practice phrase is translated into and shown
# under the phrase card. The first ("") choice means "translation off" - it is
# the default, so the panel and the extra translation work stay opt-in. The
# label is mapped to a FLORES-200 code for NLLB (see mimora/translator.py) and
# persisted via save_user_setting("translation_language", …).
TRANSLATION_LANGUAGES = ("", "Russian", "Ukrainian", "Spanish", "French",
                         "German", "Italian", "Chinese", "Japanese")


def translation_targets(language: str = None) -> tuple:
    """Translation-panel choices with the practiced language removed.

    Translating a phrase into the language it is already in is pointless, so the
    active language's display name is dropped from TRANSLATION_LANGUAGES (the
    "off" choice and every other language stay). *language* defaults to the
    active practice language. English is not in the base list today, so for
    English practice this returns the list unchanged; a later stage adds English
    as a target for practicing other languages.
    """
    profile = LANGUAGE_PROFILES.get(language or PRACTICE_LANGUAGE)
    own = profile["display_name"] if profile else None
    return tuple(label for label in TRANSLATION_LANGUAGES if label != own)


TRANSLATION_LANGUAGE = _USER.get("translation_language", "")
if TRANSLATION_LANGUAGE not in TRANSLATION_LANGUAGES:
    print(f"[config] settings.json: translation_language must be one of "
          f"{TRANSLATION_LANGUAGES}, got {TRANSLATION_LANGUAGE!r}; using '' "
          f"(translation off)", file=sys.stderr)
    TRANSLATION_LANGUAGE = ""

# =====================================================================
# Prosody panel (UI state)
# =====================================================================
# Expanded/visible state of the prosody block (pitch + energy charts), toggled
# by the "Intonation & stress" collapse header in the app and by the settings
# window, and persisted on change. False = collapsed: the charts are hidden and
# their (expensive) computation is skipped. Defaults to collapsed for a calmer
# first view.
SHOW_PROSODY = _bool("show_prosody", False)

# Visibility of the articulation face shown in the hero card's score row,
# toggled from the settings window and persisted on change.
SHOW_FACE = _bool("show_face", True)

# Collapsed state of the editable practice-text box, toggled by clicking the
# "Practice text:" caption in the app and persisted on change. The selector
# row below the box stays visible either way. Collapsed by default.
PRACTICE_TEXT_COLLAPSED = _bool("practice_text_collapsed", True)

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
    "bg_main": "#121214",           # window background (darkest surface)
    "bg_panel": "#1a1a1e",          # panels, text inputs, status bar
    "bg_card": "#222836",           # hero-card surface, lightest of the three
                                    # (bg_main < bg_panel < bg_card)
    "bg_accent": "#1f1430",         # accent-tinted controls (combobox, small buttons)
    "bg_accent_active": "#2a1a45",  # hovered/active accent controls
    "bg_button": "#4d2f87",         # primary action buttons (muted brand purple)
    "bg_button_active": "#5e3aa6",  # hovered/active primary buttons
    "text_button": "#ffffff",       # label on primary action buttons
    "border": "#25252a",
    "accent": "#8a2be2",            # brand purple: titles, focus highlights
    # Text
    "text": "#f8f8f2",
    "text_emph": "#f1f1f6",         # emphasized feedback text
    "text_bright": "#ffffff",       # selection fg, text cursor
    "text_dim": "#a0a0a5",          # secondary labels
    "text_muted": "#6272a4",        # tertiary/system text
    "text_accent": "#d6c2ff",       # text on accent-tinted controls
    "phrase": "#d8c4ff",            # hero practice phrase (overridden per theme to
                                    # match the cyan "you" prosody curve)
    "text_disabled": "#555560",
    "text_disabled_dim": "#3a3a40",
    # Articulation face (intentionally identical in every theme; the disc is
    # white with dark-blue eyes and a red-brown mouth, see face_widget.py)
    "face": "#ffffff",
    "eyes": "#1e2a44",
    "mouth": "#8a3a2c",
    # Status / feedback ("reference" marks everything tied to the reference
    # audio: status texts and the reference prosody curve)
    "good": "#5cd98a",
    "ready": "#00e676",
    "bad": "#ef6b6b",
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

def available_themes() -> tuple:
    """Theme names selectable in the UI, discovered from config/themes/.

    Every ``<name>_schema.json`` file is one theme; the built-in "dark" palette
    is always included even without a schema file (see the fallback logic
    below). Sorted for a stable selector order.
    """
    names = {"dark"}
    try:
        for schema in (CONFIG_DIR / "themes").glob("*_schema.json"):
            names.add(schema.name[: -len("_schema.json")])
    except OSError:
        pass  # unreadable themes dir - the built-in dark theme still works
    return tuple(sorted(names))


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
    # For "dark" a missing file is fine - the built-in palette IS dark.
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
# speaker (tts.py) streams. Both modules import this object - do not create
# separate Lock instances or they will not mutually exclude each other.
AUDIO_LOCK = threading.Lock()

# Pipeline sample rate: mic captures are downsampled to this rate, and it is
# what playback of recordings and Wav2Vec2 pronunciation analysis expect.
AUDIO_SAMPLE_RATE = 16_000

AUDIO_CHANNELS = 1           # Mono for both recording and playback
AUDIO_LATENCY = None         # None → OS default shared-mode latency
# Device indices come from hardware detection; None → OS default microphone/speaker.
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
