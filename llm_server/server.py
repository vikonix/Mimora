"""
Minimal OpenAI-compatible HTTP server for local GGUF models.

Runs as a separate process to keep llama_cpp in its own CUDA context,
preventing GPU contention with Kokoro TTS in the main application.

Usage:
    python server.py --model ../models/llama-3.2-3b-instruct-q4_k_m.gguf
    python server.py --model path/to/model.gguf --port 8765 --n-gpu-layers 20 --n-ctx 2048

Endpoints:
    GET  /health                  — model status
    GET  /v1/models               — OpenAI-compatible model list
    POST /v1/chat/completions     — streaming or non-streaming chat
    POST /v1/model/load           — hot-reload model without restarting server
"""

import argparse
import json
import logging
import os
import site
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Iterator, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel


def _register_nvidia_dll_dirs() -> None:
    """Make CUDA runtime DLLs from pip-installed nvidia-* packages findable.

    The CUDA wheel of llama-cpp-python needs cudart64_12.dll / cublas64_12.dll
    but only searches its own lib/ dir and CUDA_PATH. With no system CUDA
    Toolkit those DLLs come from the nvidia-cuda-runtime-cu12 /
    nvidia-cublas-cu12 pip packages, which place them under
    site-packages/nvidia/*/bin — register those dirs before importing
    llama_cpp. No-op on non-Windows or when the packages are absent.

    llama_cpp loads llama.dll with winmode=RTLD_GLOBAL (0), which selects the
    legacy Windows DLL search: PATH is consulted, os.add_dll_directory() dirs
    are not — so the dirs must go on PATH (add_dll_directory is kept in case
    a future llama_cpp switches to the default winmode).
    """
    if sys.platform != "win32":
        return
    for site_dir in site.getsitepackages():
        for bin_dir in Path(site_dir).glob("nvidia/*/bin"):
            os.add_dll_directory(str(bin_dir))
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ["PATH"]


_register_nvidia_dll_dirs()

try:
    from llama_cpp import Llama
except ImportError:
    print(
        "ERROR: llama_cpp is not installed.\n"
        "Install it with: pip install llama-cpp-python\n"
        "For CUDA support see llm_server/README.md",
        file=sys.stderr,
    )
    sys.exit(1)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

app = FastAPI(title="EchoLoop LLM Server", version="1.0.0")

# Global model state.
# _inference_lock serialises all inference calls — llama_cpp is not thread-safe.
# It is held for the entire streaming duration, so model reload blocks until
# the current generation finishes. This is intentional for a single-user app.
_model: Optional[Llama] = None
_model_path: Optional[str] = None
_model_params: dict = {}
_inference_lock = threading.Lock()


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "local-model"
    messages: list[Message]
    temperature: float = 0.7
    max_tokens: int = 256
    top_p: float = 0.9
    stream: bool = False


class ModelLoadRequest(BaseModel):
    model_path: str
    n_gpu_layers: int = 20
    n_ctx: int = 2048


# ── Model management ──────────────────────────────────────────────────────────

def _load_model(model_path: str, n_gpu_layers: int, n_ctx: int, verbose: bool = False) -> None:
    """Load (or reload) the GGUF model. Must be called with _inference_lock held."""
    global _model, _model_path, _model_params

    if _model is not None:
        logging.info("Unloading current model before reload...")
        if hasattr(_model, "close"):
            _model.close()
        _model = None

    logging.info(
        f"Loading model: {model_path} | "
        f"n_gpu_layers={n_gpu_layers} | n_ctx={n_ctx}"
    )
    _model = Llama(
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        verbose=verbose,
    )
    _model_path = model_path
    _model_params = {"n_gpu_layers": n_gpu_layers, "n_ctx": n_ctx}
    logging.info("Model loaded successfully.")


# ── SSE helpers ───────────────────────────────────────────────────────────────

def _make_sse_chunk(chunk_id: str, delta: dict, finish_reason: Optional[str] = None) -> str:
    """Format a single SSE data line in OpenAI streaming format.

    ``chunk_id`` is generated once per stream — OpenAI clients expect every
    chunk of one completion to share the same id. ``delta`` is passed through
    as-is so non-content deltas (the leading ``{"role": "assistant"}`` chunk)
    are not lost; rebuilding it from the content alone used to drop them.
    """
    payload = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason,
        }],
    }
    return f"data: {json.dumps(payload)}\n\n"


def _stream_chat(request: ChatCompletionRequest) -> Iterator[str]:
    """
    Generator that holds _inference_lock for the full streaming duration.
    This prevents concurrent inference calls and model reloads mid-stream.
    """
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"  # one id for the whole stream

    with _inference_lock:
        if _model is None:
            yield _make_sse_chunk(chunk_id, {}, finish_reason="stop")
            yield "data: [DONE]\n\n"
            return

        stream = _model.create_chat_completion(
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
            stream=True,
        )

        for chunk in stream:
            try:
                choice = chunk["choices"][0]
                yield _make_sse_chunk(chunk_id, choice.get("delta") or {},
                                      choice.get("finish_reason"))
            except (KeyError, IndexError):
                continue

    yield "data: [DONE]\n\n"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Returns model load status and basic params.

    Deliberately lock-free: a health probe must answer immediately even while
    a long generation (or model load) holds _inference_lock. The globals may
    be read mid-reload, which is acceptable for a diagnostic endpoint.
    """
    loaded = _model is not None
    return {
        "status": "ok" if loaded else "no_model",
        "model_path": _model_path,
        "params": dict(_model_params) if loaded else {},
    }


@app.get("/v1/models")
def list_models():
    """
    OpenAI-compatible model list endpoint.
    Required for LLMManager.check_connection() to work.
    """
    return {
        "object": "list",
        "data": [{
            "id": "local-model",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "local",
        }],
    }


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions with optional streaming."""
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. POST /v1/model/load first.",
        )

    if request.stream:
        return StreamingResponse(
            _stream_chat(request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",  # disables nginx buffering if proxied
            },
        )

    # Non-streaming: acquire lock and return full response.
    # The second _model is None check inside the lock is intentional (double-checked locking):
    # the model could be unloaded between the early check above and acquiring the lock.
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    with _inference_lock:
        if _model is None:
            raise HTTPException(status_code=503, detail="Model not loaded.")
        result = _model.create_chat_completion(
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
            stream=False,
        )
    return result


@app.post("/v1/model/load")
def reload_model(req: ModelLoadRequest):
    """
    Hot-reload the model without restarting the server.
    Blocks until any in-progress inference finishes.
    """
    if not os.path.exists(req.model_path):
        raise HTTPException(
            status_code=404,
            detail=f"Model file not found: {req.model_path}",
        )
    try:
        with _inference_lock:
            _load_model(req.model_path, req.n_gpu_layers, req.n_ctx)
        return {"status": "loaded", "model_path": req.model_path}
    except Exception as exc:
        # _load_model unloads the old model before loading the new one (two
        # models may not fit in VRAM simultaneously), so a failed reload leaves
        # the server with no model at all — say so explicitly.
        raise HTTPException(
            status_code=500,
            detail=f"Reload failed and the server now has NO model loaded "
                   f"(POST /v1/model/load again): {exc}",
        )


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="EchoLoop local LLM server (OpenAI-compatible)"
    )
    parser.add_argument("--model", type=str, default=None,
                        help="Path to GGUF model file (optional at startup)")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--n-gpu-layers", type=int, default=20,
                        help="Number of model layers to offload to GPU")
    parser.add_argument("--n-ctx", type=int, default=2048,
                        help="Context window size in tokens")
    parser.add_argument("--verbose", action="store_true",
                        help="Show llama.cpp model loading output (layer counts, VRAM usage, etc.)")
    args = parser.parse_args()

    if args.model:
        with _inference_lock:
            _load_model(args.model, args.n_gpu_layers, args.n_ctx, verbose=args.verbose)
    else:
        logging.warning(
            "No --model provided. Server will start without a model. "
            "POST /v1/model/load to load one."
        )

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
