# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev

"""Runtime compatibility shims for the Intel-macOS fallback stack.

Intel Macs (x86_64) cap PyTorch at 2.2.2 - no newer wheel is published for that
platform - which in turn forces transformers<5 (see the "Intel macOS note" in
the root requirements.txt). transformers 4.5x refuses ``torch.load`` on torch
< 2.6 (CVE-2025-32434) unless the checkpoint is a safetensors file. Every model
Mimora loads through transformers (facebook/wav2vec2-large-960h,
facebook/wav2vec2-xlsr-53-espeak-cv-ft, facebook/nllb-200-distilled-600M)
publishes only ``pytorch_model.bin`` - no safetensors variant exists upstream -
so that gate blocks weight loading outright on Intel macOS.

Because these are fixed, pinned, trusted repositories (not user-supplied files),
we accept ``torch.load`` for them on the Intel-Mac fallback and disable that one
gate. The shim is guarded on torch < 2.6, so on every current stack (Windows /
Linux CUDA, Apple Silicon - all torch >= 2.6) it does nothing.
"""

from __future__ import annotations

import importlib
import logging

# Set once the gate has been handled (patched, or found unnecessary) so repeated
# load_models() calls stay cheap.
_handled = False


def _torch_below_2_6() -> bool:
    """True when the installed torch predates the 2.6 that transformers demands
    for ``torch.load``. Any parse failure returns False (assume a modern torch)."""
    try:
        import torch

        major, minor = (int(p) for p in torch.__version__.split("+")[0].split(".")[:2])
        return (major, minor) < (2, 6)
    except Exception:
        return False


def allow_torch_load_for_trusted_models() -> None:
    """Neutralise transformers' torch>=2.6 requirement for ``torch.load``.

    No-op unless torch < 2.6 (i.e. the Intel-macOS fallback). Call it once before
    any ``from_pretrained`` that loads a ``.bin`` checkpoint. Idempotent.
    """
    global _handled
    if _handled:
        return
    if not _torch_below_2_6():
        _handled = True  # modern torch - the gate never fires; nothing to do.
        return

    def _noop(*_args, **_kwargs):  # replacement for check_torch_load_is_safe
        return None

    patched = False
    # modeling_utils.load_state_dict calls check_torch_load_is_safe() by the name
    # bound in its own namespace, so patch that reference; also patch the origin
    # in import_utils for any other caller.
    for mod_name in ("transformers.modeling_utils", "transformers.utils.import_utils"):
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        if hasattr(mod, "check_torch_load_is_safe"):
            mod.check_torch_load_is_safe = _noop
            patched = True

    if patched:
        logging.getLogger(__name__).info(
            "Intel-macOS fallback: allowing torch.load for trusted pinned models "
            "(torch %s < 2.6; models ship only .bin, no safetensors).",
            _installed_torch_version(),
        )
    _handled = True


def _installed_torch_version() -> str:
    try:
        import torch

        return torch.__version__
    except Exception:
        return "unknown"
