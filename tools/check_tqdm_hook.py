# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""Check the first-run progress hook against the INSTALLED huggingface_hub.

    python tools/check_tqdm_hook.py

The fast suite downloads nothing, so it describes the hub with fakes: it can
say what the code does when tqdm_class is missing, never whether it is missing
here. This asks the library that is actually resolved in this environment and
then fetches through it. Run it after raising the huggingface_hub pin, and once
on every platform where that pin does not apply, before a release.

The download is README.md of the GGUF repo rather than the 2 GB model:
gguf_fetch.ensure_gguf takes the filename from the target it is given, so a few
kilobytes exercise the identical call. Signature and network path both.

The fetch runs inside first_run_download's own online context, and the
huggingface_hub import with it (see main). Reaching the library any earlier
would measure a transfer the app never performs, and on a machine that already
has its models it would not reach the network at all.

Where the pin applies (huggingface_hub >= 1.24 - Windows, Linux, Apple
Silicon):

    huggingface_hub 1.26.0
      hf_hub_download    declares=True  passed=True
      snapshot_download  declares=True  passed=True
    downloaded: ...README.md
    bytes recorded: 17238

Where it does not (Intel macOS: transformers 4.x caps the hub below 1.0, and
hf_hub_download gained the argument only in 1.24):

    huggingface_hub 0.36.2
      hf_hub_download    declares=False passed=False
      snapshot_download  declares=True  passed=True
    downloaded: ...README.md
    bytes recorded: 0

Two ways to read a bad result:

* a traceback instead of "downloaded:" means model_fetch.progress_kwargs let
  the keyword through to an entry point that does not take it - the first run
  dies on the required GGUF component, which is what that helper exists to
  prevent;
* "bytes recorded: 0" where the pin applies means the hub stopped driving our
  stand-in. Nothing raises, the download completes, and the first-run bar sits
  at zero for gigabytes. That silence is the reason this tool is worth running
  at all.
"""

from __future__ import annotations

import inspect
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# The tool runs as a script, so sys.path starts at tools/ and the package next
# door is not importable without this (same as tools/preview_first_run.py).
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# huggingface_hub is deliberately NOT imported here - main() imports it, and
# says what breaks when it is reached any sooner.
from mimora import first_run_download, gguf_fetch, model_fetch  # noqa: E402


def main() -> None:
    # The stand-in the first-run window uses, fed by a real download below. No
    # component is begun: advance() only touches the current counter, which is
    # what snapshot() reports, and complete() is never called.
    state = first_run_download.ProgressState(
        100 * first_run_download.BYTES_PER_MB)
    sink = first_run_download.make_tqdm_class(state)

    # Everything that touches huggingface_hub happens inside this context, the
    # import of the library included. The context puts the environment into the
    # state the app downloads in, and huggingface_hub freezes that state into
    # its constants module the moment it is imported - so importing it at the
    # top of this file would be wrong twice over:
    #
    # * prepare_hf_env() would not have run yet, leaving HF_HOME and the
    #   Windows / macOS fallbacks (HF_HUB_DISABLE_SYMLINKS, HF_HUB_DISABLE_XET)
    #   at their defaults. The xet path builds different bars, so the byte
    #   count printed at the end would describe a transfer no first run makes,
    #   and on Windows without Developer Mode the fetch can die with
    #   WinError 1314;
    # * the mimora import above pulls in config, which switches offline mode on
    #   wherever the run's models are already cached - which is exactly the
    #   machine a release is cut from. _hub_online() lifts it for the length of
    #   the download and restores it afterwards.
    #
    # The private name is used on purpose: this is the same package, and a
    # second copy of that ordering would be a second thing to get wrong.
    with first_run_download._hub_online():
        import huggingface_hub
        from huggingface_hub import hf_hub_download, snapshot_download

        print("huggingface_hub", huggingface_hub.__version__)
        for func in (hf_hub_download, snapshot_download):
            declares = "tqdm_class" in inspect.signature(func).parameters
            passed = bool(model_fetch.progress_kwargs(func, object))
            print(f"  {func.__name__:<18} declares={declares!s:<5} "
                  f"passed={passed}")

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "README.md"
            print("downloaded:",
                  gguf_fetch.ensure_gguf(target, tqdm_class=sink))

    print("bytes recorded:", state.snapshot().done_bytes)


if __name__ == "__main__":
    main()
