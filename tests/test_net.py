# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""Tests for mimora/net.py.

No test opens a connection. What is checked is which certificates the context
is given, because that is the whole purpose of the module: the failure it
prevents (CERTIFICATE_VERIFY_FAILED on a new Windows installation) does not
occur on a development computer, so nothing else would show a regression.
"""

import ast
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

from mimora import net


class SslContextTests(unittest.TestCase):
    def test_certifi_bundle_is_added_to_the_system_store(self):
        # The default context is created first and certifi is added to it. A
        # context built from certifi only would stop trusting the certificate
        # of a corporate proxy, which exists only in the system store.
        fake_certifi = types.ModuleType("certifi")
        fake_certifi.where = lambda: "bundle.pem"
        with mock.patch.dict(sys.modules, {"certifi": fake_certifi}), \
                mock.patch.object(net.ssl, "create_default_context") as create:
            context = net.ssl_context()

        create.assert_called_once_with()
        self.assertIs(context, create.return_value)
        context.load_verify_locations.assert_called_once_with(
            cafile="bundle.pem")

    def test_missing_certifi_gives_the_default_context(self):
        # install.py calls llama_server_fetch before the requirements exist.
        # A None entry in sys.modules makes the import raise ImportError.
        with mock.patch.dict(sys.modules, {"certifi": None}), \
                mock.patch.object(net.ssl, "create_default_context") as create:
            context = net.ssl_context()

        self.assertIs(context, create.return_value)
        context.load_verify_locations.assert_not_called()


class ModuleImportTests(unittest.TestCase):
    def test_module_level_imports_are_stdlib_only(self):
        # A module-level "import certifi" would make llama_server_fetch
        # unimportable in the interpreter that runs install.py.
        tree = ast.parse(Path(net.__file__).read_text(encoding="utf-8"))
        imported = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module)
        self.assertEqual(imported, {"__future__", "ssl"})


if __name__ == "__main__":
    unittest.main()
