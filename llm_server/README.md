# LLM Server

A standalone HTTP server for running local GGUF models. Runs as a separate process so that llama_cpp and Kokoro TTS operate in independent CUDA contexts — no GPU contention.

Fully compatible with the OpenAI Chat Completions API, so the main application communicates with it through the standard `openai` client.

## Installing Dependencies

```bash
cd llm_server
pip install -r requirements.txt
```

### llama-cpp-python with CUDA Support (Recommended)

The default `pip install llama-cpp-python` builds a CPU-only version. For GPU support you need a CUDA build.

**Prerequisites (Windows):**
- Visual Studio Community with the **"Desktop development with C++"** workload
- CUDA Toolkit compatible with your GPU (12.x or 11.x)

**Installation:**
```powershell
# PowerShell
$env:CMAKE_ARGS="-DGGML_CUDA=on"
pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
```

```cmd
rem Command Prompt
set CMAKE_ARGS=-DGGML_CUDA=on
pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
```

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

When `LLM_BACKEND = "local_server"` is set in `config.py`, the main application launches the server automatically and waits for it to become ready (up to `LOCAL_SERVER_STARTUP_TIMEOUT` seconds).

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
