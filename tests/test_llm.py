# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""Unit tests for the request fields of mimora/llm.py.

request_extra_body decides what a phrase request sends outside the OpenAI API.
A wrong answer does not fail loudly: a thinking model on llama-server returns
an empty phrase for every request. phrase_sampling picks the temperature and
top_p for the server that answers. Run from the project root with:

    python -m unittest tests.test_llm
"""

import unittest
from unittest.mock import patch

from mimora import config
from mimora.llm import LLMManager, phrase_sampling, request_extra_body


class RequestExtraBodyTests(unittest.TestCase):
    def test_llama_server_gets_thinking_switched_off(self):
        # Without it a thinking model spends the whole token budget on
        # reasoning_content, and message.content comes back empty.
        body = request_extra_body("llama-server")
        self.assertEqual(body["chat_template_kwargs"],
                         {"enable_thinking": False})

    def test_lm_studio_gets_nothing(self):
        # LM Studio has its own thinking switch; an unknown field is a risk.
        self.assertIsNone(request_extra_body("lm-studio"))

    def test_the_off_backend_gets_nothing(self):
        self.assertIsNone(request_extra_body("off"))

    def test_every_call_returns_a_new_dict(self):
        # A caller that changes the answer must not change the next request.
        first = request_extra_body("llama-server")
        first["chat_template_kwargs"]["enable_thinking"] = True
        second = request_extra_body("llama-server")
        self.assertFalse(second["chat_template_kwargs"]["enable_thinking"])


class PhraseSamplingTests(unittest.TestCase):
    """phrase_sampling: the small model's values unless the server is adopted."""

    VALUES = {
        "PHRASE_GEN_TEMPERATURE": 0.7,
        "PHRASE_GEN_TOP_P": 0.9,
        "ADOPTED_PHRASE_GEN_TEMPERATURE": 1.0,
        "ADOPTED_PHRASE_GEN_TOP_P": 0.95,
    }

    def test_the_own_server_gets_the_small_model_values(self):
        with patch.multiple(config, **self.VALUES):
            self.assertEqual(phrase_sampling(False), (0.7, 0.9))

    def test_an_adopted_server_gets_the_larger_model_values(self):
        # A larger model repeats itself at the small model's values.
        with patch.multiple(config, **self.VALUES):
            self.assertEqual(phrase_sampling(True), (1.0, 0.95))

    def test_a_new_manager_assumes_its_own_server(self):
        # Every backend except an adopted llama-server keeps the tuned values,
        # so the flag starts False and only app.py sets it.
        self.assertFalse(LLMManager().adopted_server)


if __name__ == "__main__":
    unittest.main()
