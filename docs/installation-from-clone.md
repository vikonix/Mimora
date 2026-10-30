# Installation from a clone

For most people the [quick install or the published
package](../README.md#installation) in README is the better way. Every step of
the setup below can also be run by hand: [Manual
installation](installation-manual.md).

---

## Set up with `install.py`

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
[instructions](installation-manual.md#espeak-ng-no-separate-install-needed).
On Windows, **Developer Mode** lets the model cache use symlinks; without it the
installer copies files instead, which always works but uses more disk.

---

## Run from a clone

Run Mimora in the **same virtual environment** you installed into, so that the
app uses the interpreter that has the dependencies:

```bash
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux

python main.py
```

What to do once the app shows **Ready** is in [Usage](../README.md#usage) in
README.

## Where the files are

Running from a clone, everything stays inside the project directory: `config/`,
`models/`, `model_cache/`, `bin/llama/` and `logs/`.

Installed as a package, they move to the user-data directory instead: see
[Where Mimora keeps your files](technical-details.md#where-mimora-keeps-your-files).
