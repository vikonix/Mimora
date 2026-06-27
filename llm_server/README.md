# LLM Server

A standalone HTTP server for running local GGUF models. Runs as a separate process so that llama_cpp and Kokoro TTS operate in independent CUDA contexts — no GPU contention.

Fully compatible with the OpenAI Chat Completions API, so the main application communicates with it through the standard `openai` client.

## Installing Dependencies

```bash
cd llm_server
pip install -r requirements.txt
```

### llama-cpp-python with CUDA Support (Recommended)

The default `pip install llama-cpp-python` gives a CPU-only build: recent versions have no
prebuilt wheels on PyPI, so pip silently compiles from source without CUDA, and
`--n-gpu-layers` is then ignored. For GPU support you need a CUDA build.

**Option A — prebuilt CUDA wheel (no compiler or CUDA Toolkit needed):**

```powershell
python -m pip install llama-cpp-python==0.3.4 --force-reinstall --no-cache-dir --only-binary=:all: --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
python -m pip install nvidia-cuda-runtime-cu12==12.1.105 nvidia-cublas-cu12==12.1.3.1
```

Notes:
- `0.3.4` is the newest version with prebuilt CUDA wheels (Python 3.11, Windows) on that
  index. `--only-binary=:all:` makes pip fail loudly instead of silently falling back to a
  CPU-only source build.
- The `nvidia-*` packages provide the CUDA runtime DLLs (`cudart64_12.dll`,
  `cublas64_12.dll`) that the wheel does not bundle. `server.py` registers their
  `site-packages/nvidia/*/bin` directories via `os.add_dll_directory` at startup, so no
  system CUDA Toolkit and no manual DLL copying is required.
- The only system requirement is an NVIDIA driver with CUDA 12.1 support.

**Option B — build from source** (for a version newer than the prebuilt wheels):

Prerequisites (Windows): Visual Studio Community with the **"Desktop development with C++"**
workload and a CUDA Toolkit compatible with your GPU (12.x or 11.x).

```powershell
# PowerShell
$env:CMAKE_ARGS="-DGGML_CUDA=on"
python -m pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
```

```cmd
rem Command Prompt
set CMAKE_ARGS=-DGGML_CUDA=on
python -m pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
```

**Verify:** `python hwconfig/detect_hardware.py` (from the project root) must report
`llama offload: yes`; with a CPU-only build it reports `llama offload: NO`.

## Starting the Server Manually

```bash
python server.py --model ../models/llama-3.2-3b-instruct-q4_k_m.gguf
```

All parameters:

| Parameter | Default | Description |
|---|---|---|
| `--model` | — | Path to the GGUF file (can be omitted at startup) |
| `--host` | `127.0.0.1` | Bind address |
| `--port` | `8765` | Port |
| `--n-gpu-layers` | `20` | Number of layers to offload to GPU |
| `--n-ctx` | `2048` | Context window size in tokens |

## Automatic Startup from the App

When the LLM backend is `local_server` (the default; set via `"llm_backend"` in `config/settings.json`, read by `config.py`), the main application launches the server automatically and waits for it to become ready (up to `LOCAL_SERVER_STARTUP_TIMEOUT` seconds).

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Model status and load parameters |
| `GET` | `/v1/models` | Model list (OpenAI-compatible) |
| `POST` | `/v1/chat/completions` | Chat completion (streaming and non-streaming) |
| `POST` | `/v1/model/load` | Hot-swap model without restarting the server |

### Hot-Swapping the Model

```bash
curl -X POST http://127.0.0.1:8765/v1/model/load \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "../models/qwen2.5-3b-instruct-q4_k_m.gguf",
    "n_gpu_layers": 20,
    "n_ctx": 2048
  }'
```

## Choosing `n_gpu_layers`

`n_gpu_layers` controls how many model layers are offloaded to the GPU. More layers = faster generation, but more VRAM required.

| VRAM | Recommended value |
|---|---|
| 4 GB | 10–15 |
| 6 GB | 15–20 |
| 8 GB | 20–25 |
| 12 GB+ | 25+ (full offload) |

If VRAM runs out, the remaining layers fall back to CPU automatically.

## Recommended Models

- `Llama-3.2-3B-Instruct-Q4_K_M.gguf` — good balance of quality and speed
- `Qwen2.5-3B-Instruct-Q4_K_M.gguf` — strong multilingual alternative
