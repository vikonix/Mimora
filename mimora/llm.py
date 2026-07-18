# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev

import re
import json
import random
import hashlib
import logging
from datetime import datetime
from typing import List, Optional
from openai import OpenAI
from mimora import config
from mimora.phrase_source import split_sentences

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
# via _format_hint(): two examples are drawn at random per request - even when
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

# Leading conversational preambles the model sometimes prepends before the
# actual phrase, e.g. "Here's a short sentence to practice pronunciation: ...".
# Matched only when the text before the first colon carries a lead-in keyword,
# so a phrase that legitimately contains a colon is left intact. The {0,100}
# bounds keep the match to a short lead-in rather than a whole sentence.
_PREAMBLE_RE = re.compile(
    r"^[^:]{0,100}?\b(?:here(?:'s| is)|sentence|phrase|fragment|"
    r"practice|pronunciation|sure|okay|ok)\b[^:]{0,100}?:\s*",
    re.IGNORECASE)

# =====================================================================
# Proficiency-level support (config.PHRASE_GEN_LEVEL, 0..5).
# The level's hints are folded into the SYSTEM prompt (which therefore only
# changes when the user changes the level, keeping the llama.cpp prompt-prefix
# cache effective), and the generated phrase is validated afterwards against
# the level's word range and wordfreq Zipf floor. See
# tasks/phrase_level_task.md for the design.
# =====================================================================

# Unicode-aware word tokenizer (Spanish accents, English contractions).
_WORD_RE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)?")

# Lazily bound wordfreq.zipf_frequency: the package loads frequency data on
# import, and the app must not pay that cost at startup - only when the first
# phrase is validated or logged. False marks a failed import (validation
# disabled).
_zipf_fn = None


def _get_zipf_fn():
    """Return wordfreq.zipf_frequency, or None when wordfreq is unavailable."""
    global _zipf_fn
    if _zipf_fn is None:
        try:
            from wordfreq import zipf_frequency
            _zipf_fn = zipf_frequency
        except ImportError:
            _zipf_fn = False
            logging.warning(
                "wordfreq is not installed - phrase-level vocabulary "
                "validation is disabled (pip install wordfreq).")
    return _zipf_fn or None


def _min_zipf(phrase: str, lang: str) -> Optional[float]:
    """Lowest Zipf frequency among the phrase's content words, or None.

    Words shorter than 3 letters are skipped (articles, clitics), as are
    capitalized words that do not open the phrase - likely proper nouns
    carried over from the source text, which must not fail the vocabulary
    check. Returns None when nothing is checkable (no wordfreq, no words).
    """
    zipf = _get_zipf_fn()
    if zipf is None:
        return None
    values = []
    for index, word in enumerate(_WORD_RE.findall(phrase)):
        if len(word) < 3:
            continue
        if index > 0 and word[:1].isupper():
            continue
        # wordfreq tokens carry the straight apostrophe; a typographic one
        # from the model ("don’t") must not read as an unknown word.
        values.append(zipf(word.lower().replace("’", "'"), lang))
    return min(values) if values else None


def _fits_level(phrase: str, level_cfg: dict, lang: str,
                fragment: bool) -> bool:
    """True when *phrase* fits the level's word range and vocabulary floor.

    Fragments only face the vocabulary floor: their 2-4 word length is fixed
    by the fragment prompt, not by the level. When wordfreq is unavailable
    the vocabulary check passes (prompt hints alone still steer the level).
    """
    if not fragment:
        min_words, max_words = level_cfg["words"]
        if not (min_words <= len(_WORD_RE.findall(phrase)) <= max_words):
            return False
    floor = level_cfg["min_zipf"]
    if floor is not None:
        lowest = _min_zipf(phrase, lang)
        if lowest is not None and lowest < floor:
            return False
    return True


# Cap on the level-sample log: on the first append of a run the file is cut
# down to its newest _LEVEL_SAMPLES_KEPT lines (same pattern as the engines'
# sample logs), so it cannot grow without bound.
_LEVEL_SAMPLES_KEPT = 1000
_level_log_trimmed = False  # once-per-process guard


def _log_level_sample(record: dict) -> None:
    """Append one phrase-level sample to logs/phrase_level_samples.jsonl.

    Diagnostics for tuning the per-level Zipf floors offline; a failure to
    write must never break phrase generation, hence every file operation
    sits inside the OSError guard (the record itself is JSON-safe primitives).
    """
    global _level_log_trimmed
    path = config.LOG_DIR / "phrase_level_samples.jsonl"
    try:
        if not _level_log_trimmed:
            _level_log_trimmed = True
            if path.exists():
                lines = path.read_text(encoding="utf-8").splitlines()
                if len(lines) > _LEVEL_SAMPLES_KEPT:
                    path.write_text(
                        "\n".join(lines[-_LEVEL_SAMPLES_KEPT:]) + "\n",
                        encoding="utf-8")
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        logging.exception("Could not write phrase-level sample:")


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
        # A missing client is a programming error, not a connectivity problem -
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
                # Connection failures are expected (e.g. LM Studio offline) -
                # log the message only, not the full traceback.
                logging.error(f"LLM server not available: {error}")
            return False

    def generate_phrase(self, source_text: str, length: str = "full") -> str:
        """Generate one short practice phrase derived from ``source_text``.

        Non-streaming completion(s): normally one, at most
        1 + config.PHRASE_GEN_LEVEL_RETRIES when the phrase fails the
        proficiency-level validator (see _fits_level). To keep output varied,
        the prompt changes from call to call: only a sliding window of the
        source text is sent (see _current_window), plus a randomly picked
        focus word and - for full sentences - a random opening-style hint.
        Listing previously used phrases in the prompt is deliberately avoided:
        small models tend to copy such "do not reuse" lists instead of
        avoiding them.

        ``length`` selects the output style:
          - "full"     → one complete sentence (the default).
          - "fragment" → a short 2-4 word fragment, not a complete sentence.
        """
        if self.client is None:
            raise RuntimeError("LLM client not initialized. Call init_client() first.")

        is_fragment = (length == "fragment")

        # Active proficiency level (0..5); clamped defensively - the value can
        # be set live from the settings window between calls.
        levels = config.PHRASE_GEN_LEVELS
        level = max(0, min(int(config.PHRASE_GEN_LEVEL), len(levels) - 1))
        level_cfg = levels[level]

        # The level hints go into the SYSTEM prompt: it stays identical from
        # call to call (until the user changes the level), so llama.cpp's
        # prompt-prefix cache keeps skipping its evaluation on weak machines.
        if is_fragment:
            # Fragments get the vocabulary hint only: tense/structure hints
            # make no sense for a non-sentence and would confuse the model.
            system_prompt = (config.PHRASE_GEN_FRAGMENT_SYSTEM_PROMPT
                             + " " + level_cfg["vocab_hint"])
            max_tokens = config.PHRASE_GEN_FRAGMENT_MAX_TOKENS
            ask = config.PHRASE_GEN_FRAGMENT_ASK
        else:
            min_words, max_words = level_cfg["words"]
            system_prompt = (
                config.PHRASE_GEN_SYSTEM_PROMPT.format(
                    min_words=min_words, max_words=max_words)
                + " " + level_cfg["vocab_hint"]
                + " " + level_cfg["grammar_hint"])
            max_tokens = config.PHRASE_GEN_MAX_TOKENS
            ask = config.PHRASE_GEN_FULL_ASK

        # The window is computed once per call (a retry is not a new "use").
        window_text = self._current_window(source_text)
        wordfreq_lang = config.PHRASE_GEN_WORDFREQ_LANG

        # Level validator loop: at most 1 + PHRASE_GEN_LEVEL_RETRIES cheap
        # non-streaming completions. The budget is deliberately tiny (prompt
        # eval dominates latency on weak machines) and the degradation is
        # soft: when it is exhausted, the last candidate is returned as is.
        attempts = 1 + max(0, int(config.PHRASE_GEN_LEVEL_RETRIES))
        phrase = ""
        for attempt in range(1, attempts + 1):
            # Focus word and opening hint are re-rolled per attempt, so a
            # retry explores a different phrasing instead of repeating one.
            user_prompt = f"Source text:\n{window_text}\n\n{ask}"
            focus_word = self._pick_focus_word(
                window_text, level_cfg["min_zipf"], wordfreq_lang)
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
            except Exception:
                # Connectivity/timeout: retrying would only stack timeouts on
                # a struggling machine - keep the historical "return empty".
                logging.exception("Phrase generation error:")
                return ""

            raw = (response.choices[0].message.content or "").strip()
            phrase = self._clean_phrase(raw, fragment=is_fragment)
            fits = bool(phrase) and _fits_level(
                phrase, level_cfg, wordfreq_lang, is_fragment)
            _log_level_sample({
                "ts": datetime.now().isoformat(timespec="seconds"),
                "lang": wordfreq_lang,
                "level": level,
                "length": length,
                "attempt": attempt,
                "fits": fits,
                "word_count": len(_WORD_RE.findall(phrase)),
                # Logged on every level, even without a floor: threshold
                # tuning needs the value everywhere. This makes the first
                # generated phrase trigger the lazy wordfreq import
                # regardless of level - still nothing at application startup.
                "min_zipf": _min_zipf(phrase, wordfreq_lang),
                "phrase": phrase,
            })
            if fits:
                break
            logging.info(
                f"Phrase {phrase!r} does not fit level {level} "
                f"(attempt {attempt}/{attempts}).")

        logging.info(f"Generated practice phrase ({length}): {phrase!r}")
        return phrase

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
            # Window sticks out past the last sentence - wrap to the start.
            window = sentences[start:] + sentences[:end - len(sentences)]

        self._window_uses += 1
        if self._window_uses >= config.PHRASE_GEN_WINDOW_REPEATS:
            shift = max(1, window_size // 2)
            self._window_start = (start + shift) % len(sentences)
            self._window_uses = 0

        return " ".join(window)

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split text into sentences (see phrase_source.split_sentences).

        The splitter lives in mimora/phrase_source.py so the "off" backend's
        provider can use it without importing this (openai-dependent) module.
        """
        return split_sentences(text)

    @staticmethod
    def _pick_focus_word(text: str, min_zipf: Optional[float] = None,
                         lang: Optional[str] = None) -> str:
        """Pick a random content word from ``text`` to steer the phrase.

        Returns "" when the text has no suitable word. Randomizing the focus
        word is the main defense against the model converging on one phrase.

        When ``min_zipf`` and ``lang`` are given, words below that Zipf
        frequency are excluded first: steering the vocabulary BEFORE
        generation is free, whereas fixing it afterwards costs a retry.
        Falls back to the unfiltered pool when the filter empties it.
        """
        # Unicode-aware tokenization: an ASCII [A-Za-z] class would drop or
        # mangle accented words ("práctica" -> "ctica", "años" -> nothing),
        # feeding garbage candidates into the prompt for Spanish. Stopwords
        # are per-language profile data (config.PHRASE_GEN_STOPWORDS): the
        # frequency filter below prefers FREQUENT words, so without them the
        # pick would drift toward function words.
        words = {
            w.lower() for w in re.findall(r"[^\W\d_]{4,}", text)
            if w.lower() not in config.PHRASE_GEN_STOPWORDS
        }
        if words and min_zipf is not None and lang:
            zipf = _get_zipf_fn()
            if zipf is not None:
                frequent = {w for w in words if zipf(w, lang) >= min_zipf}
                words = frequent or words
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
        # Drop a leading conversational preamble ("Here's a sentence ...: <phrase>")
        # only when real content remains after the colon, so the guard never
        # empties an otherwise valid phrase.
        preamble = _PREAMBLE_RE.match(text)
        if preamble and text[preamble.end():].strip():
            text = text[preamble.end():].strip()
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
