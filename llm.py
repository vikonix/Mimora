import re
import logging
import threading
from queue import Queue
from threading import Event
from openai import OpenAI
import config

# Technical configuration parameters
LLM_TIMEOUT = 30.0

# Compiled once at import time — splits on sentence-ending punctuation only when
# followed by an uppercase letter, avoiding false splits on "Mr. Smith" or "1.5 sec".
_SENTENCE_END = re.compile(r'(?<=[.!?])\s+(?=[A-ZА-Я])')


class LLMManager:
    def __init__(self, model: str = None):
        self.client = None
        # Model name sent in API requests; defaults to LM Studio value from config
        self.model = model or config.LM_STUDIO_MODEL
        # Chat history buffer starting with the system instructions
        self.messages = [{"role": "system", "content": config.SYSTEM_PROMPT}]
        # Protects self.messages from concurrent reads/writes across threads
        self._messages_lock = threading.Lock()

    def init_client(self, base_url: str = None, api_key: str = None):
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
        try:
            if self.client is None:
                raise RuntimeError("LLM client not initialized. Call init_client() first.")
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

    def stream_and_queue_tts(self, user_text: str, tts_queue: Queue, stop_event: Event, token_callback=None) -> str:
        """
        Streams text from the LLM, parses sentences using regex on-the-fly,
        and pushes completed strings into the TTS queue.
        """
        if self.client is None:
            raise RuntimeError("LLM client not initialized. Call init_client() first.")

        logging.info(f"LLM request started for user input: {user_text!r}")

        try:
            # Append user message and snapshot history for the API call.
            # Snapshot prevents the lock being held during the entire streaming operation.
            with self._messages_lock:
                self.messages.append({"role": "user", "content": user_text})
                messages_snapshot = list(self.messages)

            stream_response = self.client.chat.completions.create(
                model=self.model,
                messages=messages_snapshot,
                temperature=config.LLM_TEMPERATURE,
                max_tokens=config.LLM_MAX_TOKENS,
                top_p=config.LLM_TOP_P,
                stream=True,
                timeout=LLM_TIMEOUT,
            )

            full_reply = ""
            sentence_buffer = ""

            for chunk in stream_response:
                if stop_event.is_set():
                    logging.info("LLM streaming interrupted by user stop event.")
                    break

                token = chunk.choices[0].delta.content or ""
                if not token:
                    continue

                if token_callback:
                    token_callback(token)
                full_reply += token
                sentence_buffer += token

                parts = _SENTENCE_END.split(sentence_buffer)
                if len(parts) > 1:
                    sentence_buffer = parts.pop()
                    for item in parts:
                        text_to_speak = item.strip()
                        if text_to_speak:
                            logging.info(f"Queued sentence to TTS: {text_to_speak!r}")
                            tts_queue.put(text_to_speak)

            # Flush any residual text remaining inside the buffer
            remaining_text = sentence_buffer.strip()
            if remaining_text and not stop_event.is_set():
                logging.info(f"Queued final sentence segment to TTS: {remaining_text!r}")
                tts_queue.put(remaining_text)

            final_reply = full_reply.strip() if full_reply.strip() else "Sorry, I did not get a response."
            with self._messages_lock:
                self.messages.append({"role": "assistant", "content": final_reply})
                self._trim_history()

            logging.info(f"LLM full response: {final_reply!r}")
            return final_reply

        except Exception:
            # Roll back the user message so history stays consistent (user/assistant pairs)
            with self._messages_lock:
                if self.messages and self.messages[-1].get("role") == "user":
                    self.messages.pop()
            logging.exception("LLM Stream error:")
            return "Sorry, please try again."

    def _trim_history(self):
        """Prunes conversation history to the most recent LLM_HISTORY_MAX_PAIRS turns.

        Must be called with self._messages_lock held.
        """
        system_message = self.messages[0]
        conversation = self.messages[1:]
        max_messages = config.LLM_HISTORY_MAX_PAIRS * 2
        self.messages = [system_message] + conversation[-max_messages:]
