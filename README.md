<p align="center">
  <img src="https://raw.githubusercontent.com/vikonix/Mimora/main/docs/mimora-icon.png" width="64" alt="Mimora icon">
</p>

<h1 align="center">Mimora</h1>

<p align="center">
  <a href="https://pypi.org/project/mimora/"><img src="https://img.shields.io/pypi/v/mimora" alt="PyPI version"></a>
  <a href="https://pypi.org/project/mimora/"><img src="https://img.shields.io/pypi/pyversions/mimora" alt="Python 3.11 | 3.12"></a>
  <a href="https://github.com/vikonix/Mimora/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="License: MIT"></a>
</p>

**Free, offline pronunciation trainer for English and Spanish.** Mimora says a
phrase, you repeat it, and you get a grade from 0 to 5 (like 4+) and the exact
words to work on. Speech synthesis, speech recognition, phrase generation and
scoring all run on your computer, so your voice never leaves it.

<p align="center">
  <img src="https://raw.githubusercontent.com/vikonix/Mimora/main/docs/mimora1.gif" width="400"
       alt="Mimora demo: one full practice loop - new phrase, speak, score, highlighted words, replay">
</p>

## Download

**Windows** - download the Setup file and run it. It installs for your user,
without administrator rights.

<p>
  <a href="https://github.com/vikonix/Mimora/releases/latest"><img src="https://img.shields.io/badge/Download_for_Windows-Setup_.exe-2ea44f?style=for-the-badge" alt="Download for Windows (.exe)"></a>
</p>

**macOS, Linux, or Windows from a terminal** - one command: see [Quick
install](#quick-install-recommended).

Needs 16 GB RAM and no graphics card. The models take about 4 GB (up to about
8.5 GB with every model), and Mimora asks before it downloads them. The Setup
file is not code-signed yet, so the first time Windows may show "Windows
protected your PC": click **More info**, then **Run anyway**. Other ways to
install (uv, pipx, pip, from a clone): [Installation](#installation).

---

## Why Mimora?

- 🔒 **100% offline after install** - your voice never leaves your computer. No cloud, no accounts, no API keys.
- ♾️ **Unlimited, varied practice** - a local LLM writes fresh phrases from your own text, so you never drill the same canned sentences.
- 💸 **No subscription** - free and open source. The only cost is disk space for the models.
- 💻 **Runs on a normal PC** - **16 GB RAM and no graphics card**. A GPU is optional and only makes it faster.
- 🎯 **Instant, word-level feedback** - a grade from 0 to 5 and the exact words to work on, with reference-vs-you replay.

## Who is it for?

Anyone working on clearer speech in a new language: language learners, accent reduction and shadowing practice. You bring the text (a paragraph, a song, a script), and Mimora turns it into spoken drills.

## Supported languages

The practice language is set in the settings window (**Language**, applies after a restart). A language can offer regional variants (**Accent**).

| Language | Variants | Scoring engines |
|---|---|---|
| **English** | American, British | `phoneme` (default, calibrated), `acoustic`, `none` |
| **Spanish** (Peninsular / Castilian) | Castilian | `phoneme` (**experimental**), `none` |

Calibration, the status of Spanish and the translation languages: [Languages](https://github.com/vikonix/Mimora/blob/main/docs/technical-details.md#languages).

---

## How it works

For each practice phrase Mimora runs a simple loop:

1. **Prompt** - the local LLM writes a phrase from your *practice text*, and the TTS voice of the language speaks it (Kokoro for English, Supertonic for Spanish). The same audio is the reference for scoring.
2. **Record** - press `SPACE` or click the mic, then repeat the phrase. The take stops on its own once you fall silent, or when you press `SPACE` again.
3. **Analyze** - your recording is compared with the reference, and prosody (pitch and energy) is added to the result. The default **phoneme** engine matches espeak reference phonemes against a wav2vec2 phoneme recognizer. The **acoustic** engine is the alternative, selected in the settings window (**Engine**; settings.json `"engine"`). How each engine scores: [How scoring works](https://github.com/vikonix/Mimora/blob/main/docs/technical-details.md#how-scoring-works).
4. **Feedback** - you get a grade from 0 to 5 with a verdict (for example **4+ Good**), what was recognized, and the words to improve. The acoustic engine shows a score out of 100 instead.
5. **Loop** - repeat the same phrase until you are happy with the score, then generate the next one.

You can play the **reference** and **your own recording** back to back to hear the difference.

> **No-LLM mode for slow machines.** Set the **LLM backend** to `off` and the language model is never loaded or started. Each phrase is then a sentence of your practice text, taken verbatim and in order. This saves the GGUF download, the server subprocess, RAM and VRAM, and it starts faster. It also suits drilling a text exactly as written. Phrase length is fixed in this mode: sentences are never shortened. Together with the `none` scoring engine, Mimora then runs with no LLM and no recognizer model at all.

### Example

```text
Reference:  The weather is getting colder.
You said:   The weather is getting color.
Grade:      4-  Good
Work on:    colder
```

---

## Features

- 🎙️ **One-press recording** - press once, speak, and it stops by itself when you go quiet.
- 🗣️ **One consistent reference voice** - the prompt and the scored reference are the same voice, so the target never changes. Prefer variety? Turn on **Random voice per phrase** in the settings.
- 🧠 **Practice your own material** - paste a paragraph, a song or a few sentences into the *practice text* panel, and the local LLM turns them into an endless stream of phrases.
- ⚙️ **Settings window** (the gear button) - the practice **language** and its **accent** (both apply after a restart), the TTS **voice** and playback **speed**, **Random voice per phrase** (needs at least two voices of the current language), the **phrase length** (full phrase or a few words), the **translation language** shown under the phrase, and a **user name** that selects the per-user scoring calibration.
- 📊 **Two interchangeable scoring engines**, selected in the settings window (**Engine**; settings.json `"engine"`). The default **phoneme** engine compares the phonemes of your take with the phonemes of the reference text. The **acoustic** engine compares the sound of the two recordings instead. Both are length-invariant, and both can be calibrated to your voice from a clone (`python pronunciation/<engine>/calibrate.py`).
- 🔁 **Replay reference vs. your recording** to hear the difference.
- 😀 **Articulation face** - a schematic mouth moves with the speech while audio plays, and shows a smiley for your score while idle.
- 🧵 **Responsive window** - scoring and model loading never block the interface.

## Screenshots

| Dark theme | Light theme |
|:---:|:---:|
| ![Mimora pronunciation trainer, dark theme](https://raw.githubusercontent.com/vikonix/Mimora/main/docs/mimora-dark.png) | ![Mimora pronunciation trainer, light theme](https://raw.githubusercontent.com/vikonix/Mimora/main/docs/mimora-light.png) |

*Both themes ship with the app; drop your own `<name>_schema.json` into `config/themes/` to add or replace one.*

---

## Installation

Two ways in, in the order most people should try them:

1. **[Quick install](#quick-install-recommended)** - one command on Windows,
   macOS and Linux, or a Setup file on Windows. It installs uv and Mimora and
   adds Mimora to the Start menu or the Linux application menu. This is the
   recommended way.
2. **[From PyPI](#install-from-pypi)** - the same package, installed with uv,
   pipx or pip, with every command and option in your hands.

To work on the code, or to have every model in place before the first start,
install from a clone: [Installation from a clone](https://github.com/vikonix/Mimora/blob/main/docs/installation-from-clone.md).
To run each command yourself: [Manual installation](https://github.com/vikonix/Mimora/blob/main/docs/installation-manual.md).

Every way installs **prebuilt packages (wheels), so no compilation toolchain is
needed**. You need neither a C/C++ compiler nor a CUDA toolkit.

### Requirements

- **Hardware** - a typical laptop or desktop: **16 GB RAM and no GPU required**. On CPU the first phrases are slower. An NVIDIA GPU is optional, needs a CUDA build of PyTorch, and speeds up scoring and phrase generation.
- **Windows** is the primary target. macOS and Linux are supported as well.
- A microphone and speakers.

Platform details (espeak-ng, tkinter and PortAudio on Linux, macOS) and the
model list: [Technical details](https://github.com/vikonix/Mimora/blob/main/docs/technical-details.md).

### Quick install (recommended)

One command installs [uv](https://docs.astral.sh/uv/) if it is missing, then
Mimora with Python 3.12, and adds Mimora to the Start menu (Windows) or the
application menu (Linux). On macOS, start it with `mimora` in a terminal. Copy
the command into a terminal. Do not download the script with a browser and run
it as a file.

```bash
# macOS
curl -LsSf https://github.com/vikonix/Mimora/releases/latest/download/install.sh | sh

# Linux (Ubuntu Desktop has wget but not curl)
wget -qO- https://github.com/vikonix/Mimora/releases/latest/download/install.sh | sh
```

```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://github.com/vikonix/Mimora/releases/latest/download/install.ps1 | iex"
```

<details>
<summary>What the script does</summary>

`-ExecutionPolicy ByPass` applies to this one command only, the system policy
does not change. On Windows the script also installs the Visual C++ runtime if
it is missing, and on a computer with an NVIDIA GPU it gets the CUDA build of
PyTorch.

**On Linux** the script does not run `sudo`. When system packages are missing,
it prints one `apt` command for them at the end: run it before the first start.

The models are not part of the installation. The first start names the
download size (about 4 GB with the default settings) and asks before it
downloads.

</details>

<details>
<summary>Upgrade and uninstall</summary>

To upgrade, run the same command again. To uninstall:

```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "& ([scriptblock]::Create((irm https://github.com/vikonix/Mimora/releases/latest/download/install.ps1))) -Uninstall"
```

```bash
# macOS
curl -LsSf https://github.com/vikonix/Mimora/releases/latest/download/install.sh | sh -s -- --uninstall

# Linux
wget -qO- https://github.com/vikonix/Mimora/releases/latest/download/install.sh | sh -s -- --uninstall
```

Uninstalling keeps the models, settings and logs. The script prints the
directory, so that you can delete it yourself.

</details>

#### Windows Setup file

If you prefer not to use a terminal, download
`Mimora-<version>-windows-x64-setup.exe` from the [latest
release](https://github.com/vikonix/Mimora/releases/latest) and run it. It
installs for the current user, without administrator rights, and it also
installs the Visual C++ runtime if it is missing. It downloads Mimora from
PyPI, so it needs an internet connection. The Setup file is not signed yet:
Windows shows "Windows protected your PC" with an unknown publisher. Click
**More info**, then **Run anyway**. To uninstall, use **Settings > Apps >
Installed apps**. The uninstaller asks whether to delete the models and
settings too.

### Install from PyPI

Mimora is an application rather than a library, so install it with a tool
installer instead of adding it to an everyday environment.
**[uv](https://docs.astral.sh/uv/) is the recommended one**: it is the only
installer that supplies both the Python version and the CUDA build of PyTorch by
itself.

Mimora needs **Python 3.11 or 3.12**. Python 3.13 and newer install only with a
C++ compiler, because two dependencies publish no wheels for them. `pip` refuses
such a version by itself; `uv` does not, so pass `--python 3.12` to it (see
below).

uv is a single binary and needs no Python of its own:

```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Linux (Ubuntu Desktop has wget but not curl)
wget -qO- https://astral.sh/uv/install.sh | sh
```

The terminal that ran this command does not see `uv` yet. Open a new
terminal, or on macOS and Linux run `source ~/.local/bin/env`.

`brew install uv`, `winget install --id=astral-sh.uv -e`, `scoop install main/uv`
and `pipx install uv` install it as well. If you already have uv, check the
version (`uv --version`) and update it with `uv self update`: the GPU flag below
needs 0.9.20 or newer.

<details>
<summary><b>Linux:</b> install <code>python3-tk</code> and <code>libportaudio2</code> first</summary>

Neither is on PyPI, so no installer can supply them, and each one stops the
app at a different import: `tkinter` when the window is built, PortAudio when
audio starts.

```bash
# Debian / Ubuntu
sudo apt install python3-tk libportaudio2
```

Fedora: `sudo dnf install python3-tkinter portaudio`. Arch: `sudo pacman -S tk
portaudio`. Details in [Tk on
Linux](https://github.com/vikonix/Mimora/blob/main/docs/installation-manual.md#tk-on-linux-tkinter) and [Audio on
Linux](https://github.com/vikonix/Mimora/blob/main/docs/installation-manual.md#audio-on-linux-portaudio). Add a monochrome
emoji font as well, or the mic button draws blank boxes. The font helps only
when uv uses the system Python; with a Python that uv downloads itself the icons
stay blank (see [Emoji icons on
Linux](https://github.com/vikonix/Mimora/blob/main/docs/installation-manual.md#emoji-icons-on-linux-mic-button-shows-a-blank-box)).
On a minimal system (WSL, a container, a server) also install `libgomp1`: the
phrase generator links against it, and a desktop system usually has it already
(see [Get the llama-server
binary](https://github.com/vikonix/Mimora/blob/main/docs/installation-manual.md#get-the-llama-server-binary)).

</details>

Then install Mimora and start it:

```bash
uv tool install mimora --python 3.12
mimora
```

`--python 3.12` is not optional. A uv tool environment ignores the package's
`requires-python`, so without the flag uv installs into whatever interpreter it
finds first. uv downloads a 3.12 itself if the machine has none.

<details>
<summary><b>Windows with an NVIDIA GPU:</b> add <code>--torch-backend auto</code></summary>

Add `--torch-backend auto`, so uv resolves the CUDA build of `torch` instead of
PyPI's CPU-only wheel. **Do not add this flag on Linux or macOS** - see [GPU support](https://github.com/vikonix/Mimora/blob/main/docs/installation-manual.md#gpu-support) for
what it does there.

```powershell
uv tool install mimora --python 3.12 --torch-backend auto
```

</details>

**[pipx and pip](https://github.com/vikonix/Mimora/blob/main/docs/installation-manual.md#pipx-and-pip)**
work as well.

The install itself downloads no models. The first start does, and it asks first
(see [Models](https://github.com/vikonix/Mimora/blob/main/docs/technical-details.md#models)). Those downloads and your settings live in the operating
system's user-data directory (see [Where Mimora keeps your
files](#where-mimora-keeps-your-files)), which is outside the installed package
and therefore survives both of these:

```bash
uv tool upgrade mimora
uv tool uninstall mimora
```

Uninstalling removes the program, not the models. Delete the data directory by
hand to reclaim that space.

---

## Usage

Installed with the Setup file or the quick install, start Mimora from the Start
menu (Windows) or the application menu (Linux). In a terminal, and after an
install from PyPI, it is a console script:

```bash
mimora
```

From a clone, see [Run from a
clone](https://github.com/vikonix/Mimora/blob/main/docs/installation-from-clone.md#run-from-a-clone).

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

Press `ESC` or close the window to quit. A llama-server that Mimora started is
stopped; one that was already running when Mimora started is left running.

### Where Mimora keeps your files

Installed, Mimora keeps the models, settings and logs in the operating
system's user-data directory, outside the package, so an upgrade keeps them:

| OS | Location |
|---|---|
| Windows | `%APPDATA%\Mimora\` |
| macOS | `~/Library/Application Support/Mimora/` |
| Linux | `$XDG_DATA_HOME/mimora/` (default `~/.local/share/mimora/`) |

To move it elsewhere, and for the settings file, see [Where Mimora keeps your
files](https://github.com/vikonix/Mimora/blob/main/docs/technical-details.md#where-mimora-keeps-your-files).

---

## Known limitations

- **Spanish scoring is experimental**, because the calibration is per-language and no Spanish one exists yet (see [Languages](https://github.com/vikonix/Mimora/blob/main/docs/technical-details.md#languages)). The app logs a startup warning, and the settings window shows a notice.
- **No local phrase generator on older Macs.** It needs macOS 26 (Tahoe) on Apple Silicon and macOS 13.3 (Ventura) on an Intel Mac. Below that, Mimora says so before any download and works on without it. See [macOS notes](https://github.com/vikonix/Mimora/blob/main/docs/technical-details.md#macos-notes).
- **Only mistakes the recognizer hears become word errors.** A distorted word that is still recognized may not appear in the list. The phoneme distance, the DTW of the acoustic engine and the prosody part compensate for this in the score.
- **Scoring is heuristic** and depends on your voice and microphone. Every take's raw components are logged to `phoneme_samples.jsonl` (or `acoustic_samples.jsonl`) and to `main.log` in the `logs/` folder of the [data directory](#where-mimora-keeps-your-files). From a clone you can re-anchor the active engine to that data after a practice session: `python pronunciation/phoneme/calibrate.py` or `python pronunciation/acoustic/calibrate.py`, where `--dry-run` previews the change.

---

## Feedback

Found a bug, or have a question or an idea? Open an issue on
[GitHub](https://github.com/vikonix/Mimora/issues). When something fails, attach
`main.log` and `install.log` from the `logs/` folder of the [data
directory](#where-mimora-keeps-your-files), and say which operating system and
which way of installing you used. `main.log` is rewritten at every start, so
copy it before you start Mimora again. `install.log` keeps every installation
and first-start download, and is there once Mimora has downloaded something.

## Contributing

Pull requests are welcome. Before the first one is merged, a bot asks you to
sign the [Contributor License Agreement](https://github.com/vikonix/Mimora/blob/main/CLA.md):
you reply on the pull request with the sentence it gives you. One signature
covers all your later contributions.

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
(the `supertonic` package code is MIT), so Mimora never bundles them:
`install.py` or the first start downloads them.
