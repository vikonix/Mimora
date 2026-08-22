# Mimora

**A local, offline pronunciation trainer.** Mimora speaks a phrase, you repeat it, and it scores how close you were, highlighting the words to work on. Practice the same phrase until you are happy with the score, then move on to the next one. Everything runs on your machine: speech synthesis, speech recognition, phrase generation and scoring.

<p align="center">
  <img src="https://raw.githubusercontent.com/vikonix/Mimora/main/docs/mimora1.gif" width="400"
       alt="Mimora demo: one full practice loop - new phrase, speak, score, highlighted words, replay">
</p>

| Dark theme | Light theme |
|:---:|:---:|
| ![Mimora pronunciation trainer, dark theme](https://raw.githubusercontent.com/vikonix/Mimora/main/docs/mimora-dark.png) | ![Mimora pronunciation trainer, light theme](https://raw.githubusercontent.com/vikonix/Mimora/main/docs/mimora-light.png) |

*Both themes ship with the app; drop your own `<name>_schema.json` into `config/themes/` to add or replace one.*

---

## Why Mimora?

- 🔒 **100% offline after install** - your voice never leaves your computer. No cloud, no accounts, no API keys.
- ♾️ **Unlimited, varied practice** - a local LLM writes fresh phrases from your own text, so you never drill the same canned sentences.
- 💸 **No subscription** - free and open source. The only cost is disk space for the models.
- 💻 **Runs on a normal PC** - **16 GB RAM and no graphics card**. A GPU is optional and only makes it faster.
- 🎯 **Instant, word-level feedback** - a 0-100 score and the exact words to work on, with reference-vs-you replay.

## Who is it for?

Anyone working on clearer speech in a new language: language learners, accent reduction and shadowing practice. You bring the text (a paragraph, a song, a script), and Mimora turns it into spoken drills.

## Supported languages

The practice language is set in the settings window (**Language**, applies after a restart). A language can offer regional variants (**Accent**).

| Language | Variants | Scoring engines |
|---|---|---|
| **English** | American, British | `phoneme` (default, calibrated), `acoustic`, `none` |
| **Spanish** (Peninsular / Castilian) | Castilian | `phoneme` (**experimental**, see below), `none` |

**English** is fully calibrated in both variants. **Spanish** is an experiment while multilingual support is in development. The scoring engine itself is language-independent, but its calibration is not, and a Spanish calibration needs labeled non-native speech that is not available yet. Until it lands, Spanish takes are scored with the English calibration: usable, but not tuned. The `acoustic` engine uses an English recognizer and is offered for English only.

The translation panel covers 200+ languages, but that is the translation shown beside the phrase, not the practice language.

---

## How it works

For each practice phrase Mimora runs a simple loop:

1. **Prompt** - the local LLM writes a phrase from your *practice text*, and the TTS voice of the language speaks it (Kokoro for English, Supertonic for Spanish). The same audio is the reference for scoring.
2. **Record** - press `SPACE` or click the mic, then repeat the phrase. The take stops on its own once you fall silent, or when you press `SPACE` again.
3. **Analyze** - your recording is compared with the reference, and prosody (pitch and energy) is added to the result. The default **phoneme** engine matches espeak reference phonemes against a wav2vec2 phoneme recognizer. The **acoustic** engine is the alternative, selected in the settings window (**Engine**; settings.json `"engine"`).
4. **Feedback** - you get a score out of 100, what was recognized, and the words to improve.
5. **Loop** - repeat the same phrase until you are happy with the score, then generate the next one.

You can play the **reference** and **your own recording** back to back to hear the difference.

> **No-LLM mode for slow machines.** Set the **LLM backend** to `off` and the language model is never loaded or started. Each phrase is then a sentence of your practice text, taken verbatim and in order. This saves the GGUF download, the server subprocess, RAM and VRAM, and it starts faster. It also suits drilling a text exactly as written. Phrase length is fixed in this mode: sentences are never shortened. Together with the `none` scoring engine, Mimora then runs with no LLM and no recognizer model at all.

### Example

```text
Reference:  The weather is getting colder.
You said:   The weather is getting color.
Score:      82 / 100
Work on:    colder
```

---

## Features

- 🎙️ **One-press recording** - press once, speak, and it stops by itself when you go quiet.
- 🗣️ **One consistent reference voice** - the prompt and the scored reference are the same voice, so the target never changes. Prefer variety? Turn on **Random voice per phrase** in the settings.
- 🧠 **Practice your own material** - paste a paragraph, a song or a few sentences into the *practice text* panel, and the local LLM turns them into an endless stream of phrases.
- ⚙️ **Settings window** (the gear button) - the practice **language** and its **accent** (both apply after a restart), the TTS **voice** and playback **speed**, **Random voice per phrase** (needs at least two voices of the current language), the **phrase length** (full phrase or a few words), the **translation language** shown under the phrase, and a **user name** that selects the per-user scoring calibration.
- 📊 **Two interchangeable scoring engines**, selected in the settings window (**Engine**; settings.json `"engine"`). The default **phoneme** engine compares the phonemes of your take with the phonemes of the reference text. The **acoustic** engine compares the sound of the two recordings instead. Both are length-invariant, and both can be calibrated to your voice (`python pronunciation/<engine>/calibrate.py`).
- 🔁 **Replay reference vs. your recording** to hear the difference.
- 😀 **Articulation face** - a schematic mouth moves with the speech while audio plays, and shows a smiley for your score while idle.
- 🧵 **Responsive window** - scoring and model loading never block the interface.

---

## Requirements

- **Hardware** - a typical laptop or desktop: **16 GB RAM and no GPU required**. On CPU the first phrases are slower. An NVIDIA GPU is optional, needs a CUDA build of PyTorch, and speeds up scoring and phrase generation.
- **Python 3.11 or 3.12.** Python 3.13 and newer install only with a C++ compiler, because two dependencies publish no wheels for them. `pip` refuses such a version by itself; `uv` does not, so pass `--python 3.12` to it (see below).
- **Windows** is the primary target. macOS and Linux are supported as well.
- A microphone and speakers.
- **espeak-ng** is needed by `phonemizer`, but it is **not a separate install**: the library and its data ship in the `espeakng-loader` wheel, which comes with the dependencies. A system install stays a valid fallback, see [espeak-ng](docs/installation-manual.md#espeak-ng-no-separate-install-needed).
- **PortAudio** (Linux only) - the native library behind recording and playback (`libportaudio2` on Debian/Ubuntu). The Windows and macOS wheels bundle it, the Linux ones do not. See [Audio on Linux](docs/installation-manual.md#audio-on-linux-portaudio).

### macOS notes

**Apple Silicon** Macs run the same pinned stack as Windows and Linux. **Intel
Macs (x86_64)** are supported through an automatic fallback: PyTorch publishes
no macOS x86_64 wheel newer than **torch 2.2.2**, and that torch cannot run
`transformers 5`. The dependency markers therefore install an older stack there
(**torch 2.2.2, transformers 4.x, NumPy 1.x**), with no manual steps. That stack
does without the `transformers 5.3` fix for CVE-2026-4372, which is acceptable
for a local app that loads only fixed, trusted models. Every other platform
keeps the hardened pins.

The **local chat model** is the one piece that depends on the macOS version
itself, because Mimora runs llama.cpp's own `llama-server` binary. llama.cpp
builds its **Apple Silicon** asset on a current macOS runner and sets no
deployment target, so the binary inherits that runner's minimum: **macOS 26**.
The Intel asset sets **13.3** explicitly. Below its minimum Mimora says so
before it downloads anything and carries on without the local model, so
everything else still works. To get phrase generation back on such a Mac, set
the **LLM backend** to LM Studio, or build llama.cpp yourself and name the
result in `"llama_server_path"`. See
[llama-server](docs/installation-manual.md#get-the-llama-server-binary).

`tkinter` comes with the python.org installer, but not with Homebrew Python.
`install.py` adds the matching `python-tk@<version>` formula for the interpreter
it runs in. If you set things up by hand on Homebrew Python, match your version,
for example `brew install python-tk@3.12`.

### Models

`install.py` downloads all of these in one run (see [Install from a clone](#install-from-a-clone-with-installpy)).
You do not have to use it: on the first start Mimora checks what is missing,
names the exact volume, and asks before anything is downloaded. Each model also
has its own command, listed per row below.

The first-run window asks in three groups. The models a session cannot run
without (the engine's recognizer and the language's TTS) leave you **Download**
or **Quit**. The local chat model is a real choice: declining it writes
`"llm_backend": "off"` into `config/settings.json`, and the settings window turns
it back on. The offline translator appears only when a translation language is
selected, and declining it writes `"translation_language": ""` the same way. Turning
translation on later restarts Mimora into the same window, because with
translation off nothing about the translator is downloaded or checked over the
network.

| Model | Used by | Download | Notes |
|---|---|---|---|
| `facebook/wav2vec2-xlsr-53-espeak-cv-ft` | pronunciation analysis (**phoneme** engine, default) | 1264 MB | espeak IPA phoneme recognizer; `python -m mimora.model_fetch --hf` |
| `facebook/wav2vec2-large-960h` | pronunciation analysis (**acoustic** engine) | 1262 MB | `python -m mimora.model_fetch --hf` |
| Kokoro-82M (`hexgrad/Kokoro-82M`) | text-to-speech (English) | 363 MB | `python -m mimora.model_fetch --hf` |
| Supertonic 3 (`Supertone/supertonic-3`) | text-to-speech (Spanish) | 404 MB | `python -m mimora.model_fetch --supertonic`. Weights are **OpenRAIL-M** licensed (code MIT), so they are downloaded, never bundled |
| `facebook/nllb-200-distilled-600M` | offline translation (translation panel) | 2483 MB | downloaded only once a translation language is selected |
| A GGUF chat model (e.g. `Llama-3.2-3B-Instruct-Q4_K_M`) | phrase generation | 2019 MB | `python -m mimora.gguf_fetch`. Not needed with `"llm_backend": "off"` |
| llama-server binary (pinned llama.cpp release) | phrase generation | 641 MB CUDA, 18 MB CPU | most of the CUDA figure is NVIDIA's runtime; `python -m mimora.llama_server_fetch` |

---

## Installation

Three ways in, in the order most people should try them:

1. **From the published wheel** ([below](#install-from-pypi-recommended)) - one
   command, no clone, no build toolchain. This is the recommended way.
2. **From a clone with [`install.py`](#install-from-a-clone-with-installpy)** -
   the guided setup for a checkout. It also downloads every model up front, and
   it is the way to go if you intend to change the code.
3. **By hand** - every command run yourself, plus the platform-specific pieces
   behind them:
   [`docs/installation-manual.md`](docs/installation-manual.md).

Every way installs **prebuilt packages (wheels), so no compilation toolchain is
needed**. You need neither a C/C++ compiler nor a CUDA toolkit.

### Install from PyPI (recommended)

Mimora is an application rather than a library, so install it with a tool
installer instead of adding it to an everyday environment.
**[uv](https://docs.astral.sh/uv/) is the recommended one**: it is the only
installer that supplies both the Python version and the CUDA build of PyTorch by
itself.

uv is a single binary and needs no Python of its own:

```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

`brew install uv`, `winget install --id=astral-sh.uv -e`, `scoop install main/uv`
and `pipx install uv` install it as well. If you already have uv, check the
version (`uv --version`) and update it with `uv self update`: the GPU flag below
needs 0.9.20 or newer.

Then install Mimora and start it:

```bash
uv tool install mimora --python 3.12
mimora
```

`--python 3.12` is not optional. A uv tool environment ignores the package's
`requires-python`, so without the flag uv installs into whatever interpreter it
finds first. uv downloads a 3.12 itself if the machine has none.

**Windows with an NVIDIA GPU** - add `--torch-backend auto`, so uv resolves the
CUDA build of `torch` instead of PyPI's CPU-only wheel. **Do not add this flag on
Linux or macOS** - see [GPU support](docs/installation-manual.md#gpu-support) for
what it does there.

```powershell
uv tool install mimora --python 3.12 --torch-backend auto
```

**[pipx](https://pipx.pypa.io/stable/installation/)** gives the same result
except for that index: an index cannot be named in a published package's
metadata, so on Windows with an NVIDIA card the CUDA build is a second command.

```powershell
pipx install --python 3.12 mimora
pipx runpip mimora install torch --index-url https://download.pytorch.org/whl/cu124 --force-reinstall
```

**pip** works too, into a virtual environment of Python 3.11 or 3.12:

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux

pip install mimora
mimora
```

**On Linux** one native library comes from the system rather than from a wheel:
`sudo apt-get install libportaudio2` (see [Audio on
Linux](docs/installation-manual.md#audio-on-linux-portaudio)). Add a monochrome
emoji font as well, or the mic button draws blank boxes (see [Emoji icons on
Linux](docs/installation-manual.md#emoji-icons-on-linux-mic-button-shows-a-blank-box)).

The install itself downloads no models. The first start does, and it asks first
(see [Models](#models)). Those downloads and your settings live in the operating
system's user-data directory (see [Where Mimora keeps your
files](#where-mimora-keeps-your-files)), which is outside the installed package
and therefore survives both of these:

```bash
uv tool upgrade mimora            # pipx: pipx upgrade mimora
uv tool uninstall mimora          # pipx: pipx uninstall mimora
```

Uninstalling removes the program, not the models. Delete the data directory by
hand to reclaim that space.

### Install from a clone with `install.py`

Use this one to work on Mimora itself, or to have every model in place before
the first start.

`install.py` does the whole setup: it checks the native pieces pip cannot supply
(`tkinter` and PortAudio on Linux, the MSVC runtime on Windows), installs the
Python dependencies, reports which `espeak-ng` library the engines will use,
detects an NVIDIA GPU and installs the matching CUDA build of `torch`, downloads
the Hugging Face models into `model_cache/`, installs the pinned llama-server
binary into `bin/llama/`, and downloads the GGUF chat model into `models/`.

```bash
git clone https://github.com/vikonix/Mimora.git Mimora
cd Mimora

# Create and activate a virtual environment, then run the installer INSIDE it
# (the script installs into whatever interpreter runs it):
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux

python install.py
```

The installer prints each step and its exact command, then asks before it runs
it (answer `Y` to run, `n` to abort, `s` to skip). Anything already installed is
offered as reinstall-or-skip rather than blindly redone. The full run is logged
to `logs/install.log`. Expect **several minutes**, mostly downloads, and roughly
**10 GB** of disk once all packages and models are in place.

Useful flags:

- `--yes` - run non-interactively (skips already-installed steps; add `--reinstall` to force them)
- `--dry-run` - print the steps and commands without executing anything
- `--cpu` - skip the GPU (CUDA) installs
- `--skip-models` / `--skip-gguf` - skip the model / GGUF downloads

The espeak-ng step reports the library the engines resolve to, which on a normal
install is the bundled one. It offers a system install only if none is found, and
on Windows it just prints
[instructions](docs/installation-manual.md#espeak-ng-no-separate-install-needed).
On Windows, **Developer Mode** lets the model cache use symlinks; without it the
installer copies files instead, which always works but uses more disk.

### Manual installation

Every step run by hand, plus the native pieces behind it - espeak-ng, PortAudio
on Linux, the emoji font, the CUDA build of PyTorch, the llama-server binary and
the GGUF chat model: [`docs/installation-manual.md`](docs/installation-manual.md).

---

## Usage

Installed from PyPI, Mimora is a console script:

```bash
mimora
```

From a clone, run it in the **same virtual environment** you installed into, so
that the app uses the interpreter that has the dependencies:

```bash
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux

python main.py
```

On the first launch the app loads the TTS and recognizer models and starts the
LLM server. Cached models are just a load that takes a moment; a missing model is
downloaded first, which takes a while. Once the app shows **Ready**:

1. Edit the **Practice text** panel, or keep the default.
2. Click **Next phrase ▶**. Mimora generates a phrase and speaks it.
3. **Press `SPACE`** (or click the mic button) and repeat the phrase. The take stops on silence, or on a second `SPACE`.
4. Read your **score** and verdict on the phrase card. Mispronounced words are underlined, and a click plays one slowly. The **WORK ON** badges name the sounds to fix, and a click gives an example word. Earlier takes stay in the attempt history below.
5. Use **Reference ▶** (or **Slow ▶**) and **My recording ▶** to compare, then repeat the phrase or generate the next one.

The **first few phrases run noticeably slowly** while the models warm up. This is
normal, and the speed settles after the first few requests.

Press `ESC` or close the window to quit. The LLM server subprocess is terminated
cleanly.

### Where Mimora keeps your files

Running from a clone, everything stays inside the project directory: `config/`,
`models/`, `model_cache/`, `bin/llama/` and `logs/`.

Installed as a package, the same layout moves under the operating system's
user-data directory, because the package's own directory is rebuilt on the next
upgrade:

| OS | Location |
|---|---|
| Windows | `%APPDATA%\Mimora\` |
| macOS | `~/Library/Application Support/Mimora/` |
| Linux | `$XDG_DATA_HOME/mimora/` (default `~/.local/share/mimora/`) |

The `MIMORA_HOME` environment variable puts that directory anywhere else, and it
overrides both cases. Use it to move several gigabytes of downloads to another
drive, to escape a roaming Windows profile that would copy them over the network
at every login, or to run two configurations side by side.

**Paths inside `config/settings.json` resolve against that file's own
directory.** The settings window writes absolute paths, which work everywhere. A
relative path is read as relative to the settings file you are editing. Keys you
leave out follow their own defaults instead, which are resolved for you.

Most settings are editable in the settings window. For the rest,
[`config/settings.example.json`](config/settings.example.json) documents every
key next to its default. Copy it to `config/settings.json` to start, or edit that
file directly. The file and every key in it are optional, and anything missing
falls back to the built-in default.

---

## GPU / CPU notes

On a single GPU the recognizer, the TTS voice, the translator and llama.cpp can
compete for VRAM. Mimora keeps them apart: the LLM runs in a separate process,
the practice loop runs its phases one after another, and the translator defaults
to the CPU. If VRAM is still tight, set `WAV2VEC2_DEVICE = "cpu"` in
`mimora/config.py`; short phrases score acceptably on the CPU.

---

## Known limitations

- **Spanish scoring is experimental**, because the calibration is per-language and no Spanish one exists yet (see [Supported languages](#supported-languages)). The app logs a startup warning, and the settings window shows a notice.
- **Only mistakes the recognizer hears become word errors.** A distorted word that is still recognized may not appear in the list. The phoneme distance, the DTW of the acoustic engine and the prosody part compensate for this in the score.
- **Scoring is heuristic** and depends on your voice and microphone. After a practice session you can re-anchor the active engine to your own data: `python pronunciation/phoneme/calibrate.py` or `python pronunciation/acoustic/calibrate.py`, where `--dry-run` previews the change. Every take's raw components are logged to `logs/phoneme_samples.jsonl` (or `logs/acoustic_samples.jsonl`) and to `logs/main.log`.

---

## Credits

- **[OpenPronounce](https://github.com/Halleck45/OpenPronounce)** (MIT) - the pronunciation-scoring core reused in `pronunciation/acoustic/`.
- **[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)** - text-to-speech (English variants).
- **[Supertonic 3](https://huggingface.co/Supertone/supertonic-3)** ([supertonic-py](https://github.com/supertone-inc/supertonic-py), code MIT, weights OpenRAIL-M) - text-to-speech (Spanish variant), ONNX runtime.
- **[wav2vec2-xlsr-53-espeak-cv-ft](https://huggingface.co/facebook/wav2vec2-xlsr-53-espeak-cv-ft)** (Hugging Face Transformers) - espeak-style IPA phoneme recognizer for the default `phoneme` engine.
- **[Wav2Vec2](https://huggingface.co/facebook/wav2vec2-large-960h)** (Hugging Face Transformers) - acoustic embeddings and transcription (`acoustic` engine).
- **[NLLB-200](https://huggingface.co/facebook/nllb-200-distilled-600M)** (Hugging Face Transformers) - offline translation for the translation panel.
- **[espeak-ng](https://github.com/espeak-ng/espeak-ng)** / **[phonemizer-fork](https://github.com/bootphon/phonemizer)** - reference phonemization (espeak IPA).
- **[panphon](https://github.com/dmort27/panphon)** - articulatory feature distance used by the phoneme edit-distance scoring.
- **[llama.cpp](https://github.com/ggerganov/llama.cpp)** - local LLM inference; the official `llama-server` binary is run as a subprocess.

## License

See [`LICENSE`](https://github.com/vikonix/Mimora/blob/main/LICENSE). The reused OpenPronounce components are MIT-licensed; their attribution is retained in `pronunciation/acoustic/speech.py`.

Model weights have their own licenses. The Supertonic 3 TTS weights in particular
are licensed under [OpenRAIL-M](https://huggingface.co/Supertone/supertonic-3/blob/main/LICENSE)
(the `supertonic` package code is MIT), so Mimora never bundles them: they are
downloaded into `model_cache/supertonic3/` by `install.py` or on the first run.
