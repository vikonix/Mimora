import threading
from pathlib import Path

# Base directory — always absolute, regardless of working directory at launch.
BASE_DIR = Path(__file__).parent

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
KOKORO_VOICE = "af_heart"     # Voice model identifier

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
LOG_FILE = str(BASE_DIR / "voice_tutor.log")
