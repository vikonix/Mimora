import re
import logging
from typing import List, Optional
from openai import OpenAI
from echoloop import config

# Technical configuration parameters
LLM_TIMEOUT = 30.0


class LLMManager:
    def __init__(self, model: Optional[str] = None):
        self.client = None
        # Model name sent in API requests; defaults to LM Studio value from config
        self.model = model or config.LM_STUDIO_MODEL

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

    def generate_phrase(self, source_text: str, recent_phrases: Optional[List[str]] = None,
                        length: str = "full") -> str:
        """Generate one short practice phrase derived from ``source_text``.

        This is a single, non-streaming, stateless completion. ``recent_phrases``
        are passed back to the model so it can avoid immediately repeating itself.

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

        user_prompt = f"Source text:\n{source_text.strip()}\n\n{ask}"
        if recent_phrases:
            avoid = "; ".join(recent_phrases)
            user_prompt += f"\nDo not reuse any of these recent phrases: {avoid}"

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
