# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev

import re
import random
import hashlib
import logging
from typing import List, Optional
from openai import OpenAI
from mimora import config

# Technical configuration parameters
LLM_TIMEOUT = 30.0

# Opening-style hints for "full" sentences. One is picked at random per
# request: a stateless prompt with a fixed text otherwise makes a small model
# converge on a single most-likely opening (e.g. every phrase starting with
# "The tourists...").
#
# A small model tends to copy the literal examples from the hint rather than
# treat them as a category (e.g. fixed "'Sometimes' or 'Usually'" made many
# phrases start with "Usually"). So hints with examples are built from a pool
# via _format_hint(): two examples are drawn at random per request — even when
# the model copies one, the openings still vary.
_EXAMPLE_POOLS = {
    "pronoun": ("I", "We", "They", "She", "He", "You"),
    "time": ("Every morning", "Last year", "On weekends", "In the evening",
             "Twice a week", "During the summer", "After lunch", "These days"),
    "adverb": ("Sometimes", "Usually", "Often", "Rarely", "Normally",
               "Occasionally", "Generally", "Slowly", "Quietly", "Luckily"),
}
# {pool} placeholders are filled by _format_hint with two random examples.
_OPENING_HINTS = (
    "Start the sentence with a pronoun such as {pronoun}.",
    "Start the sentence with a verb.",
    "Start the sentence with a time expression such as {time}.",
    "Start the sentence with a place or a location.",
    "Start the sentence with an adverb such as {adverb}.",
    "Make the sentence a simple question.",
)


def _format_hint(hint: str) -> str:
    """Fill {pool} placeholders in a hint with two random examples each."""
    for name, pool in _EXAMPLE_POOLS.items():
        if "{" + name + "}" in hint:
            first, second = random.sample(pool, 2)
            hint = hint.replace("{" + name + "}", f"'{first}' or '{second}'")
    return hint

# Common words skipped when picking a random focus word from the text window.
_STOPWORDS = frozenset(
    "the a an and or but of to in on at for from with about into over after "
    "is are was were be been being have has had do does did will would can "
    "could should this that these those there their they them then than it "
    "its his her she he you your we our not no yes very just also more most "
    "some any all each every one two when what where which who how why".split()
)


class LLMManager:
    def __init__(self, model: Optional[str] = None):
        self.client = None
        # Model name sent in API requests; defaults to LM Studio value from config
        self.model = model or config.LM_STUDIO_MODEL
        # Sliding-window state over the source text (see _current_window):
        # start index, how many phrases were generated at this position, and a
        # hash of the text the state belongs to (text edits reset the window).
        self._window_start = 0
        self._window_uses = 0
        self._window_text_hash: Optional[str] = None

    def init_client(self, base_url: Optional[str] = None,
                    api_key: Optional[str] = None):
        """
        Configure OpenAI-compatible client.

        Defaults to LM Studio settings from config when arguments are omitted,
        so existing "lm-studio" backend usage is unchanged.
        """
        url = base_url or config.LM_STUDIO_URL
        key = api_key or config.LM_STUDIO_API_KEY
        logging.info(f"Initializing LLM client → {url}")
        self.client = OpenAI(
            base_url=url,
            api_key=key,
            timeout=LLM_TIMEOUT,
        )

    def check_connection(self, silent: bool = False) -> bool:
        """
        Validates connectivity to the local LLM server.

        Pass silent=True during startup polling to suppress per-attempt error logs
        and avoid flooding the log with dozens of identical connection errors.
        """
        # A missing client is a programming error, not a connectivity problem —
        # raise it out instead of logging it as "server not available".
        if self.client is None:
            raise RuntimeError("LLM client not initialized. Call init_client() first.")
        try:
            self.client.models.list()
            logging.info("Successfully connected to LLM server.")
            return True
        except Exception as error:
            if silent:
                logging.debug(f"LLM server not yet available: {error}")
            else:
                # Connection failures are expected (e.g. LM Studio offline) —
                # log the message only, not the full traceback.
                logging.error(f"LLM server not available: {error}")
            return False

    def generate_phrase(self, source_text: str, length: str = "full") -> str:
        """Generate one short practice phrase derived from ``source_text``.

        This is a single, non-streaming completion. To keep output varied, the
        prompt changes from call to call: only a sliding window of the source
        text is sent (see _current_window), plus a randomly picked focus word
        and — for full sentences — a random opening-style hint. Listing
        previously used phrases in the prompt is deliberately avoided: small
        models tend to copy such "do not reuse" lists instead of avoiding them.

        ``length`` selects the output style:
          - "full"     → one complete sentence (the default).
          - "fragment" → a short 2-4 word fragment, not a complete sentence.
        """
        if self.client is None:
            raise RuntimeError("LLM client not initialized. Call init_client() first.")

        is_fragment = (length == "fragment")
        if is_fragment:
            system_prompt = config.PHRASE_GEN_FRAGMENT_SYSTEM_PROMPT
            max_tokens = config.PHRASE_GEN_FRAGMENT_MAX_TOKENS
            ask = ("Give me ONE short English fragment of 2 to 4 words (NOT a complete "
                   "sentence) to practice pronunciation, based on this text.")
        else:
            system_prompt = config.PHRASE_GEN_SYSTEM_PROMPT
            max_tokens = config.PHRASE_GEN_MAX_TOKENS
            ask = "Give me ONE short English sentence to practice pronunciation, based on this text."

        window_text = self._current_window(source_text)
        user_prompt = f"Source text:\n{window_text}\n\n{ask}"

        focus_word = self._pick_focus_word(window_text)
        if focus_word:
            user_prompt += f"\nTry to use the word '{focus_word}'."
        if not is_fragment:
            user_prompt += f"\n{_format_hint(random.choice(_OPENING_HINTS))}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=config.PHRASE_GEN_TEMPERATURE,
                max_tokens=max_tokens,
                stream=False,
                timeout=LLM_TIMEOUT,
            )
            raw = (response.choices[0].message.content or "").strip()
            phrase = self._clean_phrase(raw, fragment=is_fragment)
            logging.info(f"Generated practice phrase ({length}): {phrase!r}")
            return phrase
        except Exception:
            logging.exception("Phrase generation error:")
            return ""

    def _current_window(self, source_text: str) -> str:
        """Return the current sliding-window slice of ``source_text``.

        The text is split into sentences; PHRASE_GEN_WINDOW_SENTENCES
        consecutive ones form the window. Each call counts as one use; after
        PHRASE_GEN_WINDOW_REPEATS uses the window advances by half its size
        (overlapping windows shift the topic gradually) and wraps around at
        the end of the text. Editing the source text resets the window, which
        is detected by hashing the text.
        """
        text = source_text.strip()
        sentences = self._split_sentences(text)
        window_size = config.PHRASE_GEN_WINDOW_SENTENCES
        # Short text: the window covers everything, no state to track.
        if len(sentences) <= window_size:
            return text

        text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
        if text_hash != self._window_text_hash:
            self._window_text_hash = text_hash
            self._window_start = 0
            self._window_uses = 0

        start = self._window_start
        end = start + window_size
        if end <= len(sentences):
            window = sentences[start:end]
        else:
            # Window sticks out past the last sentence — wrap to the start.
            window = sentences[start:] + sentences[:end - len(sentences)]

        self._window_uses += 1
        if self._window_uses >= config.PHRASE_GEN_WINDOW_REPEATS:
            shift = max(1, window_size // 2)
            self._window_start = (start + shift) % len(sentences)
            self._window_uses = 0

        return " ".join(window)

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split text into sentences (and standalone lines, e.g. headings)."""
        parts = re.split(r"(?<=[.!?])\s+|\n+", text)
        return [p.strip() for p in parts if p and p.strip()]

    @staticmethod
    def _pick_focus_word(text: str) -> str:
        """Pick a random content word from ``text`` to steer the phrase.

        Returns "" when the text has no suitable word. Randomizing the focus
        word is the main defense against the model converging on one phrase.
        """
        words = {
            w.lower() for w in re.findall(r"[A-Za-z]{4,}", text)
            if w.lower() not in _STOPWORDS
        }
        return random.choice(sorted(words)) if words else ""

    @staticmethod
    def _clean_phrase(text: str, fragment: bool = False) -> str:
        """Strip wrapping quotes, list markers and stray whitespace from a phrase.

        When ``fragment`` is True the result is a sentence fragment, so any
        trailing sentence-ending punctuation the model added is removed.
        """
        # Strip wrapping quotes, including typographic ones the model may emit.
        text = text.strip().strip('"\'«»“”‘’').strip()
        # Drop a leading list marker like "1." or "- " if the model adds one.
        text = re.sub(r"^\s*(?:\d+[.)]|[-*])\s*", "", text)
        text = " ".join(text.split()).strip()
        if fragment:
            text = text.rstrip(".!?").strip()
        else:
            # The model occasionally returns several sentences despite the
            # prompt; keep only the first one. Split on end punctuation followed
            # by an uppercase letter so "1.5" or rare abbreviations survive.
            parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
            if len(parts) > 1:
                text = parts[0].strip()
        return text
