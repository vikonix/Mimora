# Technical details

The details README leaves out: the languages, how the score is made, the
native pieces behind the dependencies, the macOS specifics, the models, where
Mimora keeps its files, and GPU memory. How to install is in
[Installation](../README.md#installation) in README.

---

## Languages

The table of languages and variants is in [Supported
languages](../README.md#supported-languages) in README.

**English** is fully calibrated in both variants. **Spanish** is an experiment while multilingual support is in development. The scoring engine itself is language-independent, but its calibration is not, and a Spanish calibration needs labeled non-native speech that is not available yet. Until it lands, Spanish takes are scored with the English calibration: usable, but not tuned. The `acoustic` engine uses an English recognizer and is offered for English only.

The translation panel covers 200+ languages, but that is the translation shown beside the phrase, not the practice language.

---

## How scoring works

Two short overviews, ideas rather than code:

- [Phoneme engine](engine-phoneme-multilingual.md) - the default. Compares the
  sounds you produced with the sounds the text expects, for any language.
- [Acoustic engine](engine-openpronounce-base.md) - compares your recording with
  the reference recording. English only.

---

## Native components

- **espeak-ng** is needed by `phonemizer`, but it is **not a separate install**: the library and its data ship in the `espeakng-loader` wheel, which comes with the dependencies. A system install stays a valid fallback, see [espeak-ng](installation-manual.md#espeak-ng-no-separate-install-needed).
- **tkinter** (Linux only) - the Tk GUI toolkit, packaged apart from the interpreter (`python3-tk` on Debian/Ubuntu). It is not on PyPI, so no installer can supply it; Windows and python.org macOS builds carry it already. See [Tk on Linux](installation-manual.md#tk-on-linux-tkinter).
- **PortAudio** (Linux only) - the native library behind recording and playback (`libportaudio2` on Debian/Ubuntu). The Windows and macOS wheels bundle it, the Linux ones do not. See [Audio on Linux](installation-manual.md#audio-on-linux-portaudio).

## macOS notes

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
deployment target, so the binary inherits that runner's minimum: **macOS 26
(Tahoe)**. The Intel asset sets **13.3 (Ventura)** explicitly. Below its minimum
Mimora says so before it downloads anything and carries on without the local
model, so everything else still works. To get phrase generation back on such a
Mac, set the **LLM backend** to LM Studio, or build llama.cpp yourself and name
the result in `"llama_server_path"`. See
[llama-server](installation-manual.md#get-the-llama-server-binary).

`tkinter` comes with the python.org installer, but not with Homebrew Python.
`install.py` adds the matching `python-tk@<version>` formula for the interpreter
it runs in. If you set things up by hand on Homebrew Python, match your version,
for example `brew install python-tk@3.12`.

## Models

`install.py` downloads all of these in one run (see [Installation from a clone](installation-from-clone.md)).
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

## Where Mimora keeps your files

Installed as a package, the layout of a clone (see [Installation from a
clone](installation-from-clone.md#where-the-files-are)) moves under the
operating system's user-data directory, because the package's own directory is
rebuilt on the next upgrade:

| OS | Location |
|---|---|
| Windows | `%APPDATA%\Mimora\` |
| macOS | `~/Library/Application Support/Mimora/` |
| Linux | `$XDG_DATA_HOME/mimora/` (default `~/.local/share/mimora/`) |

The `MIMORA_HOME` environment variable puts that directory anywhere else, and it
overrides the project directory of a clone as well. Use it to move several gigabytes of downloads to another
drive, to escape a roaming Windows profile that would copy them over the network
at every login, or to run two configurations side by side.

**Paths inside `config/settings.json` resolve against that file's own
directory.** The settings window writes absolute paths, which work everywhere. A
relative path is read as relative to the settings file you are editing. Keys you
leave out follow their own defaults instead, which are resolved for you.

Most settings are editable in the settings window. For the rest,
[`config/settings.example.json`](../config/settings.example.json) documents every
key next to its default. Copy it to `config/settings.json` to start, or edit that
file directly. The file and every key in it are optional, and anything missing
falls back to the built-in default.

---

## GPU / CPU notes

On a single GPU the recognizer, the TTS voice, the translator and llama.cpp can
compete for VRAM. Mimora keeps them apart: the LLM runs in a separate process,
the practice loop runs its phases one after another, and the translator defaults
to the CPU. If VRAM is still tight, set `"WAV2VEC2_DEVICE": "cpu"` in the
`"config"` section of `config/hardware_config.json`; short phrases score
acceptably on the CPU.
