# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Model weights are not part of a release. They are downloaded on the first start
and stay in the user data directory across upgrades.

## [1.2.0] - 2026-10-30

Mimora installs without a terminal on Windows, and with one command on Windows,
macOS and Linux. The practice loop, both scoring engines, both languages and
both voices are unchanged.

### Added

- A Windows Setup file, `Mimora-1.2.0-windows-x64-setup.exe`, on the GitHub
  release. It installs for the current user without administrator rights, adds
  the Visual C++ runtime when it is missing, and installs Mimora from PyPI. The
  uninstaller asks whether to delete the models and settings too.
- One-command install: `install.ps1` for Windows, `install.sh` for macOS and
  Linux, on the same release. It installs uv when it is missing, then Mimora,
  and adds a Start menu shortcut or a Linux menu entry. Run it again to upgrade;
  `-Uninstall` / `--uninstall` removes Mimora and keeps the models.
- Both installers ask to close a running Mimora before they change any file.
- `mimora-gui`: the same program without a console window. The Start menu
  shortcut starts it.
- A startup splash, shown until the main window is open.
- The Mimora icon on every window, in the taskbar and in the Dock, instead of
  the Python one.
- Mimora uses a llama-server that already answers on its port (left by a
  crashed session, or started by another local app) and leaves it running on
  exit. A thinking model on such a server no longer returns empty phrases. A
  port held by a program that does not answer is refused with a message.
- Linux: when `tkinter` or PortAudio is missing, the error names the package to
  install instead of showing a traceback.
- `logs/install.log` records what the first-run window downloaded and is
  appended to, not overwritten. `logs/hwdetect.log` keeps one section per
  hardware probe.
- The fast test suite runs on GitHub Actions: Python 3.12 on Windows, macOS
  and Ubuntu, and Python 3.11 on Ubuntu.
- `docs/technical-details.md` and `docs/installation-from-clone.md`. README is
  now the short version.

### Changed

- Apple Silicon: the chat model runs on the GPU (Metal) instead of the CPU. An
  existing `config/hardware_config.json` keeps the old values until you run
  `mimora --detect-hardware`.
- `torch>=2.7` everywhere except Intel macOS. transformers 5 needs it, and an
  older torch already in the environment was kept before.
- `install.py` takes the CUDA build of torch only from series that have torch
  2.7 or newer (`cu132` down to `cu118`), and removes a `torchaudio` left over
  from an older install.
- The scroll bar of the practice text follows the theme. With Tk 9 (macOS) the
  scroll bars are narrow and have no arrows.

### Fixed

- On a new Windows installation the first start could not download the spaCy
  model or llama-server (`CERTIFICATE_VERIFY_FAILED`). Downloads now use the
  system certificate store plus certifi.
- Linux with the Python build of uv (Tk 9, the one `install.sh` installs): the
  main window stayed at "Starting" forever. Worker threads now update the
  window only through the main thread.
- A new install received `preshed 3.0.14`, which breaks loading spaCy, and
  Mimora stopped at "Initialization Failed". `preshed` is now held below 3.0.14.
- Linux without `libgomp1`: the llama-server check names the missing package,
  and the first-run window shows the whole message instead of the first 300
  characters.
- A Mac with no llama-server build for its macOS version: the error message
  advises setting the LLM backend to `off` or `lm-studio`.
- The pipx command for an NVIDIA card used the `cu124` index, whose newest torch
  is too old for transformers 5. It uses `cu128`.

## [1.1.0] - 2026-08-23

Mimora becomes an installable application: one command to install, one word to
start, and it downloads what it needs on the first start after it asks. The
practice loop, both scoring engines, both languages and both voices are
unchanged.

### Added

- A `mimora` command, and `python -m mimora`. `uv tool install mimora --python
  3.12` is the recommended install.
- A first run window that names the exact volume and asks before downloading.
  Refusing the chat model sets `"llm_backend": "off"`; refusing the translator
  clears the translation language.
- Automatic downloads for every model and binary, each also its own command
  (`python -m mimora.model_fetch`, `.gguf_fetch`, `.llama_server_fetch`). The
  llama.cpp assets are checked against a sha256.
- A user data directory outside the package, so an upgrade keeps your models,
  settings and calibration. `MIMORA_HOME` moves it.
- espeak-ng as part of the install, from the `espeakng-loader` wheel. A system
  install is now only the fallback.
- A hardware probe (`python -m mimora.detect_hardware`) and a startup warning
  when a GPU is present but unused.
- `docs/installation-manual.md`, sixteen test files, and the release tools
  `check_wheel.py`, `check_tqdm_hook.py` and `preview_first_run.py`.

### Changed

- The LLM backend is the official llama.cpp `llama-server` binary, pinned per
  platform and installed by Mimora, with a device check before launch. It
  replaces the Python server in `llm_server/`.
- Dependencies moved into `pyproject.toml`. One list instead of a list and a
  copy.
- Python 3.11 and 3.12 only. Two dependencies publish no wheels above that and
  would need a C++ compiler.
- The application moved into the package (`mimora/app.py`), together with the
  practice texts and theme schemas, so the wheel carries them.
- `install.py` rewritten: native preflight checks, one confirmation per step,
  full log.
- Documentation reorganized into `AGENTS.md` and `docs/installation-manual.md`.

### Fixed

- The first run downloaded nothing on any platform: the progress bar stand-in
  was missing the class level lock huggingface_hub asks for.
- The first run failed on the chat model on Intel macOS, where the resolved
  huggingface_hub takes no progress hook.
- The default scoring engine did not load on Linux, because `torchaudio` was
  declared but never imported and resolved to a different CUDA series than
  torch. It is no longer a dependency.
- Intel macOS: a missing `click`, and an `llvmlite` build failure now avoided by
  capping `numba`.
- A machine with no usable GPU re-downloaded the GPU build of llama-server on
  every install run.
- A Mac below the pinned build's minimum macOS version is told so before the
  download rather than after it.

### Removed

- `llm_server/`, the Python LLM server.
- The root `requirements.txt`.
- `tools/check_wheels.py`, `smoke_test_llama.py` and `sweep_llama_versions.py`.

## [1.0.0] - 2026-07-21

First public release. Phoneme and acoustic scoring engines, English and Spanish,
Kokoro and Supertonic text to speech, offline NLLB translation, and a guided
`install.py` for a clone.

[1.2.0]: https://github.com/vikonix/Mimora/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/vikonix/Mimora/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/vikonix/Mimora/releases/tag/v1.0.0
