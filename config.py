import os
import threading
from pathlib import Path

# Base directory — always absolute, regardless of working directory at launch.
BASE_DIR = Path(__file__).parent

# =====================================================================
# Local model cache (HuggingFace) — download once, then load offline
# =====================================================================
# Whisper, Kokoro and Wav2Vec2 are all pulled from the HuggingFace Hub. We pin
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
_CACHED_REPOS = (
    "Systran/faster-whisper-small",   # faster-whisper "small"
    "hexgrad/Kokoro-82M",             # Kokoro TTS (model + voice files)
    "facebook/wav2vec2-large-960h",   # Wav2Vec2 pronunciation model
)


def _all_models_cached() -> bool:
    """True only when every networked model already exists in the local cache."""
    hub_dir = Path(os.environ["HF_HOME"]) / "hub"
    for repo in _CACHED_REPOS:
        snapshots = hub_dir / ("models--" + repo.replace("/", "--")) / "snapshots"
        if not snapshots.is_dir() or not any(snapshots.iterdir()):
            return False
    return True


# Auto offline: the first run downloads online; once everything is cached, every
# later run loads straight from disk with no network access. Delete model_cache/
# to force a fresh download.
if _all_models_cached():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# =====================================================================
# Language Pair & Persona Configuration
# =====================================================================
NATIVE_LANGUAGE = "Russian"
TARGET_LANGUAGE = "English"
TARGET_LANG_CODE = "en"  # ISO code used for Whisper transcription routing

# System prompt shaping the LLM behavior into a specific educational persona
SYSTEM_PROMPT = (
    f"You are a friendly {TARGET_LANGUAGE} tutor named Emma. "
    f"The user's native language is {NATIVE_LANGUAGE}, but you should talk to them in simple {TARGET_LANGUAGE}. "
    "Keep responses very short. Use simple spoken sentences. "
    "Avoid idioms, abbreviations, complex punctuation, and compressed phrases."
)

# =====================================================================
# Controls
# =====================================================================
# Safety threshold to prevent infinite recording loops if a key gets physically stuck
MAX_RECORD_SECONDS = 20

# Hardware Acceleration setup — wrapped so startup does not crash if torch is absent
try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"

# =====================================================================
# LLM Backend Settings
# =====================================================================
# Backend selection:
#   "lm-studio"    — external LM Studio app (must be running separately)
#   "local_server" — llm_server.py started automatically as a subprocess
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
# Absolute path — safe regardless of the working directory at launch
EXTERNAL_MODEL_PATH = str(BASE_DIR / "models" / "llama-3.2-3b-instruct-q4_k_m.gguf")
EXTERNAL_N_GPU_LAYERS = 20  # Number of layers to offload to GPU
EXTERNAL_N_CTX = 2048       # Context window size

# Generation tuning parameters
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 50
LLM_TOP_P = 0.9

# Context buffer constraints
LLM_HISTORY_MAX_PAIRS = 4  # Number of full conversation turns kept in short-term memory

# =====================================================================
# Speech-to-Text (Whisper) Settings
# =====================================================================
WHISPER_MODEL = "small"
WHISPER_BEAM_SIZE = 1         # Beam size 1 provides optimal speed at temperature 0.0
WHISPER_NO_SPEECH_THRESHOLD = 0.45
WHISPER_CPU_THREADS = 4       # CPU inference threads (tune to available core count)

# Context conditioning instruction guiding Whisper's spelling logic
WHISPER_INITIAL_PROMPT = (
    f"This is a conversation with a {TARGET_LANGUAGE} tutor. "
    f"The speaker is practicing simple phrases."
)

# =====================================================================
# Text-to-Speech (Kokoro) Settings
# =====================================================================
KOKORO_LANG_CODE = "a"        # 'a' = American English, 'b' = British English, etc.
KOKORO_VOICE = "af_heart"     # Default voice model identifier

# Voices the user can pick from in the UI. All belong to lang_code 'a'
# (American English), so switching between them needs no pipeline reload.
# Prefix convention: 'af_' = American female, 'am_' = American male.
# A voice's data is downloaded on first use and then cached locally.
KOKORO_VOICES = [
    "af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky",
    "am_adam", "am_michael", "am_echo", "am_eric", "am_liam",
]

# =====================================================================
# Pronunciation Analysis (Wav2Vec2) Settings
# =====================================================================
# Acoustic + transcription model used by the pronounce/ module.
WAV2VEC2_MODEL_NAME = "facebook/wav2vec2-large-960h"
# Device for Wav2Vec2. Defaults to the shared DEVICE; set to "cpu" explicitly to
# avoid VRAM contention with llama_cpp / Kokoro on a single GPU.
WAV2VEC2_DEVICE = DEVICE
# Score (0-100) at or above which a repetition is accepted; below it the learner
# is asked to repeat the same phrase.
PRONUNCE_SCORE_THRESHOLD = 70.0

# =====================================================================
# Practice Text & Phrase Generation
# =====================================================================
# Source text shown in the input panel at startup. The LLM builds practice
# phrases from whatever text the user has in that panel.
PRACTICE_TEXT_FILE = str(BASE_DIR / "practice_text.txt")

# One short phrase is generated per request (non-streaming).
PHRASE_GEN_TEMPERATURE = 0.7
PHRASE_GEN_MAX_TOKENS = 40
PHRASE_GEN_SYSTEM_PROMPT = (
    "You generate short English sentences for pronunciation practice. "
    "Reply with exactly ONE natural spoken sentence of 4 to 8 words, easy to read aloud. "
    "Use only plain words and a single final period — no quotation marks, no numbering, "
    "no extra commentary. Base the sentence on the topic and vocabulary of the text the user provides."
)
# How many recently used phrases to send back to the model so it avoids repeats.
PHRASE_GEN_RECENT_MEMORY = 5

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
# Shared Audio Device Settings
# =====================================================================
# Single lock coordinates PortAudio access between the mic (main.py) and
# speaker (tts.py) streams. Both modules import this object — do not create
# separate Lock instances or they will not mutually exclude each other.
AUDIO_LOCK = threading.Lock()

AUDIO_CHANNELS = 1           # Mono for both recording and playback
AUDIO_LATENCY = None         # None → OS default shared-mode latency
AUDIO_INPUT_DEVICE = None    # None → OS default microphone
AUDIO_OUTPUT_DEVICE = None   # None → OS default speaker

# =====================================================================
# Logging Settings
# =====================================================================
# All log files live in a dedicated logs/ directory (created on import so the
# handlers can open their files immediately).
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = str(LOG_DIR / "voice_tutor.log")
# Log file for the auto-started local LLM server subprocess (see main.py).
LLM_SERVER_LOG_FILE = str(LOG_DIR / "llm_server.log")
