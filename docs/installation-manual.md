# Manual installation

This file carries the from-source install and the platform-specific steps behind
it. It is the third of three ways to install Mimora, and the one to read when
you want to run each command yourself, or to fix one piece of a setup that the
other two left incomplete:

1. **From the published wheel** (recommended) - see
   [Installation](../README.md#installation) in README.
2. **From a clone with `install.py`** - the same steps as here, automated and
   logged; see [Install from a clone with
   `install.py`](../README.md#install-from-a-clone-with-installpy).
3. **By hand** - this file.

The sections below are independent of each other. The Python dependencies are
the first one; everything after it is a native piece that pip cannot supply
(espeak-ng, PortAudio), a platform quirk (emoji fonts, the CUDA build of
PyTorch), or a component that is downloaded rather than installed (the
llama-server binary, the GGUF chat model). The last two arrive on the first
start of the app as well, which is why they are steps and not requirements.

Hardware, Python version and the model list stay in README, under
[Requirements](../README.md#requirements).

---

## Python dependencies

```bash
# 1. Clone
git clone https://github.com/vikonix/Mimora.git Mimora
cd Mimora

# 2. All dependencies in one step
#    The list lives in [project.dependencies] in pyproject.toml and covers both
#    pronunciation engines (panphon included, for the default phoneme engine).
#    No separate per-engine install is needed.
#
#    Editable (-e), because this is a clone: a plain `pip install .` would copy
#    the code into site-packages, leaving a second copy that your edits do not
#    reach. Editable also keeps Mimora in "source tree" mode, so config/, models/
#    and logs/ stay in the project directory rather than moving to the OS
#    user-data directory (see mimora/paths.py).
pip install -e .
```

The offline translator (NLLB-200) needs no extra step - its dependencies
(`transformers`, `sentencepiece`) are in that same list.

On Windows with an NVIDIA card, add the CUDA build of `torch` afterwards (see
[GPU support](#gpu-support)): PyPI's `torch` is CPU-only there. On
Linux PyPI already serves a CUDA build, and macOS has no CUDA at all.

## espeak-ng (no separate install needed)

Both pronunciation engines phonemize the target text with **espeak-ng**, through
`phonemizer`. It arrives with the dependencies: the `espeakng-loader` wheel
carries the espeak-ng shared library and its data, and Mimora registers both
with `phonemizer` before the first phonemization. Nothing to install by hand,
and nothing to put on `PATH`.

To see which one your environment will use:

```bash
python -m pronunciation.common.espeak
```

Run it **with the virtual environment activated**, or call the interpreter by
path (`.venv\Scripts\python.exe -m ...`). It answers for the interpreter that
runs it, so a bare `python` outside the environment reports on a different
Python than the one that runs Mimora. `install.py` reports the same thing in
its espeak-ng step and is immune to this, because it invokes its own
interpreter explicitly.

Note also that `phonemizer` loads a shared **library**, not the `espeak-ng`
executable, so `espeak-ng --version` answering on the command line says nothing
about whether Mimora can use it.

**A system espeak-ng is still a valid setup** - it is the fallback when the
wheel is missing, which is how a standalone install of `pronunciation/phoneme/`
or `pronunciation/acoustic/` can work:

- **macOS** - `brew install espeak-ng`
- **Linux** - `sudo apt-get install espeak-ng`
- **Windows** - download and run the installer from the [espeak-ng releases](https://github.com/espeak-ng/espeak-ng/releases), then set **both** of these environment variables. One is not enough: `phonemizer`'s own search looks for `espeak-ng.dll` while the installer writes `libespeak-ng.dll`, and the data directory is never found beside the library, because `phonemizer` copies the DLL to a temporary directory before loading it.

  ```
  PHONEMIZER_ESPEAK_LIBRARY   = C:\Program Files\eSpeak NG\libespeak-ng.dll
  PHONEMIZER_ESPEAK_DATA_PATH = C:\Program Files\eSpeak NG\espeak-ng-data
  ```

Switching espeak-ng versions is not free: the scoring calibration was fitted
against the transcription of the bundled build, which is why `espeakng-loader`
is pinned to a minor in `pyproject.toml`.

## Audio on Linux (PortAudio)

`sounddevice` is a wrapper around the native **PortAudio** library. Its Windows
and macOS wheels ship that library inside; its Linux wheels do not, so without
the system package every import fails with
`OSError: PortAudio library not found` - including the one in
`mimora/detect_hardware.py`, which is why `install.py` checks for it up front:

```bash
sudo apt-get install libportaudio2     # Debian / Ubuntu
sudo dnf install portaudio             # Fedora
```

If the library is installed but the app finds **no audio devices** (the
installer prints `0 input / 0 output`), the usual cause is the backend rather
than the library: Debian and Ubuntu build PortAudio with the ALSA backend only,
while WSL and most desktop setups route audio through PulseAudio. Check it with

```bash
ldd "$(ldconfig -p | grep -m1 portaudio | awk '{print $NF}')" | grep pulse
```

An empty answer means no PulseAudio backend. Two known ways out: install
`libasound2-plugins` and point ALSA's default device at pulse
(`pcm.!default pulse` in `~/.asoundrc`), or rebuild PortAudio from source with
`./configure --with-pulseaudio`.

## Emoji icons on Linux (mic button shows a blank box)

The mic/record button (`mimora/ui.py` `draw_mic_button`) draws its state icons
(`🎤` `🔴` `🔊` `⌛` `⚡`) as text on the Tk canvas, using the platform font
(`mimora/ui_theme.py`, `"DejaVu Sans"` on Linux). DejaVu Sans covers `⚡`/`⌛`
(older BMP symbols) but not `🎤`/`🔴`/`🔊` (astral-plane emoji), so on a fresh
Linux install those three render as a blank/tofu box instead of the icon -
it can look like the mic and speaker icons are simply missing.

Fix: install a **monochrome** emoji font so Tk can render the glyphs as normal
vector outlines (Tk canvas text cannot render color/bitmap emoji fonts like
`fonts-noto-color-emoji`, which is the one `apt` installs by default):

```bash
sudo apt install fonts-symbola   # in Ubuntu's universe repo; enable it first if missing:
                                  # sudo add-apt-repository universe && sudo apt update
fc-cache -f -v
```

Then restart Mimora.

## GPU support

Whether you need this step depends on your platform. PyPI serves **CPU-only**
`torch` on Windows and macOS and a **CUDA-enabled** build on Linux, and macOS has
no CUDA at all - so this section is about **Windows with an NVIDIA card**, which
is the one combination that gets a CPU wheel it did not want. Mimora says so at
startup if it happens: the app still works, Wav2Vec2 and speech synthesis are
just several times slower.

- **PyTorch** - install a CUDA build (other CUDA versions: see [pytorch.org](https://pytorch.org/get-started/locally/)):
  ```powershell
  python -m pip install torch --index-url https://download.pytorch.org/whl/cu124 --force-reinstall
  ```

  With **uv** this is one flag instead - it reads the installed driver and picks
  the matching PyTorch index itself, falling back to CPU when there is no GPU:
  ```powershell
  uv tool install mimora --python 3.12 --torch-backend auto
  # or, equivalently:
  # UV_TORCH_BACKEND=auto uv tool install mimora --python 3.12
  ```
  Needs uv 0.9.20 or newer. There is no equivalent for `pipx`: an index cannot be
  named in a published package's metadata, so a pipx install gets the CPU wheel
  and the manual step above.

  **Do not add this flag on Linux or macOS.** PyPI's `torch` is already a CUDA
  build on Linux, so the flag changes nothing you wanted; what it does change is
  the index every other package in the resolution is taken from, and the result
  is not even stable between runs. macOS has no CUDA at all.

  `--python 3.12` is not optional. A tool environment "will ignore non-global
  Python version requests like `.python-version` files and the `requires-python`
  value" ([uv docs](https://docs.astral.sh/uv/concepts/tools/#python-versions)),
  so without the flag uv installs into whatever interpreter it finds first. On a
  machine where that is 3.14 the install compiles `editdistance` and
  `curated-tokenizers` from source, because neither publishes a cp314 wheel - and
  fails outright without a C++ compiler. uv downloads a 3.12 itself if there is
  none.
- **The LLM** needs no pip package at all: it runs in the official llama.cpp
  binary, and the fetcher below picks the GPU build for your platform
  automatically - CUDA on Windows, Vulkan on Linux, where llama.cpp publishes
  no CUDA binaries at all (see the next section).

## Get the llama-server binary

The default LLM backend runs the official **llama.cpp** server as a subprocess.
`install.py` installs it, and the app offers to fetch it on the first start; to
do it separately, or to change the build:

```bash
python -m mimora.llama_server_fetch
```

This downloads a pinned llama.cpp release into `bin/llama/`, verifies the
checksum of every asset, and then confirms that the binary really runs on the
GPU backend it advertises - a CUDA build with missing runtime DLLs otherwise
falls back to the CPU **silently** and just runs about three times slower.
`--list` shows the available builds, `--variant` picks one explicitly, and
`--dry-run` prints the plan without downloading. If you already manage your own
`llama-server`, put it on `PATH` or name it in `settings.json`
(`"llama_server_path"`) instead.

Builds are pinned for **Windows x64** (CUDA, falling back to CPU when the
driver is too old), **Linux x64** (Vulkan, falling back to CPU) and **macOS**
(Metal on Apple Silicon, CPU-only on Intel - llama.cpp builds its Intel asset
with Metal switched off).

The macOS builds come with the release's own limits, and neither can be worked
around from here. Both carry a minimum macOS version stamped into the binary at
build time, read out of the current pin and checked **before** anything is
downloaded: **13.3** for the Intel build (llama.cpp sets that deployment target
explicitly) and **26.0** for the Apple Silicon one (llama.cpp sets none, so the
build inherits the macOS of the runner it was compiled on). A Mac below its
build's minimum is told so and left alone - the installer records it as a manual
step and moves on, rather than failing.

The Intel build has a second limit that no header states and nothing can check
in advance: it is compiled for the CPU of llama.cpp's CI runner, so an older
Intel Mac stops with an illegal instruction the first time the binary runs.

In either case the way out is the same: build llama.cpp on the machine and name
the result in `"llama_server_path"`, or switch `"llm_backend"` to `lm-studio`.

The Linux fallback is worth a word. llama.cpp ships no CUDA binary for Linux,
so the GPU build there is the Vulkan one, and whether it can see the GPU cannot
be known before downloading it - it needs a Vulkan loader (`libvulkan1`), an
ICD manifest published by the driver, and a device reachable through both.
Under WSL2 the NVIDIA driver publishes no Vulkan ICD at all, so the check comes
back empty. The fetcher therefore tries Vulkan, verifies it with
`--list-devices`, and installs the CPU build instead when no device appears,
saying so in the log. If you fix the Vulkan side later, `--force` re-runs the
whole selection.

## Get a GGUF model

`install.py` already downloads `llama-3.2-3b-instruct-q4_k_m.gguf` into `models/`,
the first-run window offers the same download, and `python -m mimora.gguf_fetch`
does it on its own (`--list` shows the target path and whether the file is there).
To use a different model instead, download a small instruct model (e.g. `Llama-3.2-3B-Instruct-Q4_K_M.gguf`) and place it at the path set by `EXTERNAL_MODEL_PATH` in `mimora/config.py` (default: `models/llama-3.2-3b-instruct-q4_k_m.gguf`).
