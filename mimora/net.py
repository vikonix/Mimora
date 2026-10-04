# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""TLS setup for the downloaders that use urllib.

``llama_server_fetch`` and ``spacy_model_fetch`` download with the standard
library, and the standard library trusts only the certificates that are
already in the operating system store. A new Windows installation has an
almost empty store: Windows adds root certificates on demand, but only for its
own TLS stack, never for Python. Those two downloads then fail with
CERTIFICATE_VERIFY_FAILED, while the Hugging Face downloads on the same
computer succeed, because httpx brings its own certificates (certifi).

**Stdlib-only at module level**, for the reason given in ``paths.py``:
``install.py`` uses ``llama_server_fetch`` before the requirements exist.
"""

from __future__ import annotations

import ssl


def ssl_context() -> ssl.SSLContext:
    """Return a context that trusts the system store and the certifi bundle.

    Both, not certifi alone: a corporate proxy re-signs traffic with a
    certificate that exists only in the system store, and a context built from
    certifi only would break the downloads that work there today.

    certifi is imported here and its absence is accepted, because this function
    also runs before the requirements are installed. The result is then the
    default context, which is what urllib used before this module existed.
    """
    context = ssl.create_default_context()
    try:
        import certifi
    except ImportError:
        return context
    context.load_verify_locations(cafile=certifi.where())
    return context
