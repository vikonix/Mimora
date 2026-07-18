# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev

"""Verbatim phrase source for the "off" LLM backend.

When config.LLM_BACKEND is "off", no language model is loaded or started:
practice phrases are the source text's own sentences, taken verbatim and in
order. :class:`SourceTextPhraseProvider` is a drop-in replacement for
``LLMManager`` in the phrase-generation path (main.py ``_generate_and_prompt``
calls only ``generate_phrase``); everything downstream (TTS, translation,
scoring) sees a plain text phrase and is unaffected by its origin.

Stdlib-only on purpose: this module must stay importable without the openai
package (which mimora/llm.py needs), so the sentence splitter shared with
``LLMManager`` lives HERE and llm.py delegates to it.
"""

import hashlib
import logging
import re
from typing import List, Optional


def split_sentences(text: str) -> List[str]:
    """Split text into sentences (and standalone lines, e.g. headings)."""
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p and p.strip()]


class SourceTextPhraseProvider:
    """Yields the source text's sentences as practice phrases, one per call.

    Sentences are returned sequentially with wraparound at the end of the
    text: predictable coverage, no immediate repeats. Editing the source text
    restarts the walk from the first sentence - detected by hashing the text,
    the same reset idea as ``LLMManager._current_window``.
    """

    def __init__(self):
        self._next_index = 0
        self._text_hash: Optional[str] = None

    def generate_phrase(self, source_text: str, length: str = "full") -> str:
        """Return the next sentence of ``source_text``, unmodified.

        ``length`` is accepted for interface compatibility with
        ``LLMManager.generate_phrase`` and ignored: a sentence is never
        shortened into a fragment (the phrase-length choice is disabled in
        the settings window while the LLM backend is "off").

        Returns "" when the text contains no sentences, which the caller
        already treats as a failed generation.
        """
        sentences = split_sentences(source_text)
        if not sentences:
            return ""

        text_hash = hashlib.md5(
            source_text.strip().encode("utf-8")).hexdigest()
        if text_hash != self._text_hash:
            self._text_hash = text_hash
            self._next_index = 0

        # Modulo guards against an index left beyond the end by an edit that
        # produced the same hash state but fewer sentences (defensive only).
        phrase = sentences[self._next_index % len(sentences)]
        self._next_index = (self._next_index + 1) % len(sentences)
        logging.info(f"Source-text practice phrase: {phrase!r}")
        return phrase
