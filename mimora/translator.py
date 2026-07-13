# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev

"""Offline phrase translator (NLLB-200).

A dedicated machine-translation model -- NOT the chat LLM -- turns each generated
practice phrase into the language chosen in the translation panel. The local 3B
chat model produced unusable translations (empty CJK,
leaked English words), so translation is delegated to
``facebook/nllb-200-distilled-600M``: a 200-language MT model that is small and
CPU-friendly.

``TranslatorManager`` mirrors ``TTSManager`` and the pronunciation engine:
``load_model()`` brings the network into memory, ``warm_up()`` pays the slow
first-call cost up front, and ``translate()`` returns one translated string.
Loading is lazy and idempotent: ``main.py`` preloads it at startup only when a
language is already selected, and ``translate()`` loads on demand the first time
a language is enabled at runtime -- so a session that never translates never
pays the model's RAM or startup cost.
"""

import logging
import threading

from mimora import config

# Display label (as listed in config.TRANSLATION_LANGUAGES) -> FLORES-200 code
# that NLLB expects as the target language. The source language is the one being
# practiced (config.SOURCE_FLORES_CODE). A label not in this map -- including the
# empty "translation off" choice -- yields no translation (translate() returns "").
_FLORES_CODES = {
    "Russian": "rus_Cyrl",
    "Ukrainian": "ukr_Cyrl",
    "Spanish": "spa_Latn",
    "French": "fra_Latn",
    "German": "deu_Latn",
    "Italian": "ita_Latn",
    "Chinese": "zho_Hans",
    # Japanese script is ISO 15924 "Jpan" (4 letters); "Jpn" is not a real code,
    # so it is not a vocab token and NLLB would silently pick a default language.
    "Japanese": "jpn_Jpan",
}

# Upper bound on generated translation length. Practice phrases are short (a few
# words), so this is only a safety cap that keeps a degenerate input from running
# generation away -- not a limit a real translation should ever reach.
_MAX_NEW_TOKENS = 128


class TranslatorManager:
    """Loads NLLB-200 once and translates practice phrases on demand."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._device = config.TRANSLATOR_DEVICE
        # Serializes the lazy load: translate() can be reached from the phrase
        # generation thread without an explicit preload, and two phrases could
        # overlap, so the first caller loads and the rest wait.
        self._load_lock = threading.Lock()
        # Serializes inference: translate() mutates the shared tokenizer
        # (src_lang) and runs the model, so concurrent calls must not interleave.
        self._infer_lock = threading.Lock()

    def load_model(self):
        """Bring the NLLB model + tokenizer into memory (idempotent, thread-safe).

        Safe to call repeatedly and from multiple threads: the first call loads,
        the rest return at once. transformers/torch are imported here rather than
        at module import, so an app run that never translates never pays the
        import cost. The model is assigned only after both it and the tokenizer
        load successfully, so a failed load leaves ``model`` as ``None`` and stays
        retryable (translate() keeps returning "").
        """
        if self.model is not None:
            return
        with self._load_lock:
            if self.model is not None:  # another thread loaded it while we waited
                return
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            from pronunciation.common.compat import allow_torch_load_for_trusted_models

            allow_torch_load_for_trusted_models()
            name = config.NLLB_TRANSLATOR_MODEL_NAME
            logging.info("Loading translator model %s on %s...", name, self._device)
            tokenizer = AutoTokenizer.from_pretrained(name)
            model = AutoModelForSeq2SeqLM.from_pretrained(name).to(self._device)
            model.eval()
            # NLLB ships a default max_length (=200) in its generation config.
            # We cap generation with max_new_tokens instead (see translate), and
            # transformers warns on every call when BOTH are set. Clearing the
            # baked-in default leaves max_new_tokens as the single length control
            # and silences that per-translation warning.
            model.generation_config.max_length = None
            self.tokenizer = tokenizer
            self.model = model
            logging.info("Translator model loaded.")

    def warm_up(self):
        """Run one throwaway translation so the first real call is not slow.

        Warms the *configured* target language (falling back to Russian when
        none is selected, e.g. when called directly in a test), so the first
        real translation pays nothing extra. Non-fatal: a warm-up failure is
        logged and swallowed. The model is still usable afterwards, and a
        transient hiccup here must not abort startup.
        """
        try:
            self.translate(config.TRANSLATOR_WARMUP,
                           config.TRANSLATION_LANGUAGE or "Russian")
        except Exception:
            logging.exception("Translator warm-up failed (continuing).")

    def translate(self, text: str, language_label: str) -> str:
        """Translate *text* (in the practiced language) into *language_label*.

        Returns the translated string, or "" when there is nothing to do: empty
        input, the "translation off" choice, or a label this engine does not
        support. The model is loaded on first use (see load_model). Any failure
        is logged and returns "" so a translation problem never breaks phrase
        generation.
        """
        text = (text or "").strip()
        target_code = _FLORES_CODES.get(language_label)
        if not text or not target_code:
            return ""
        try:
            self.load_model()
            import torch

            with self._infer_lock:
                # NLLB tokenizes the source with a language prefix taken from
                # src_lang, so it must be set before encoding. The source is the
                # practiced language (config.SOURCE_FLORES_CODE).
                self.tokenizer.src_lang = config.SOURCE_FLORES_CODE
                inputs = self.tokenizer(text, return_tensors="pt").to(self._device)
                # The output language is selected by forcing the first generated
                # token to the target language code. That code is an ordinary
                # vocab token, so convert_tokens_to_ids gives its id -- this works
                # for both the fast and slow tokenizer and across transformers
                # versions, unlike the removed lang_code_to_id mapping.
                forced_bos = self.tokenizer.convert_tokens_to_ids(target_code)
                # An unknown code maps to the <unk> id, which would make NLLB
                # silently translate into some default language. Refuse rather
                # than mislead: log it and return "" so the panel keeps its
                # placeholder instead of showing a wrong-language translation.
                if forced_bos is None or forced_bos == self.tokenizer.unk_token_id:
                    logging.error("Translator: unknown FLORES code %r for %s; "
                                  "skipping translation.", target_code, language_label)
                    return ""
                with torch.no_grad():
                    generated = self.model.generate(
                        **inputs,
                        forced_bos_token_id=forced_bos,
                        max_new_tokens=_MAX_NEW_TOKENS,
                    )
            return self.tokenizer.batch_decode(
                generated, skip_special_tokens=True)[0].strip()
        except Exception:
            logging.exception("Translation failed for %r -> %s.",
                              text, language_label)
            return ""
