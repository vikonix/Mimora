# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev

"""Settings persistence glue: save, mirror, and reset user settings.

Extracted from the main controller (the thin split): this module owns the
mechanics around settings.json - persisting one value, mirroring a change
into the open settings window, and the "Default" reset with its no-op
dispatch diffing - plus the two tables describing which settings the
running app can apply live. The *dispatch* stays in the controller
(main.py on_setting_changed and the on_*_changed handlers): applying a
change may regenerate the phrase or touch the view, which is controller
behavior. The controller injects its three touchpoints as callbacks.

All methods run on the Tk main thread (same contract this code had while
it lived in main.py).
"""

import logging
import os
from typing import Callable, Optional

from mimora import config

# settings.json key -> (config attribute, cast) for settings that the
# runtime re-reads from the config module on every use (recorder loop,
# phrase generator, recording dumps). Updating the attribute applies the
# change immediately; everything else in _RESTART-marked fields waits for
# a restart because it was bound at startup.
LIVE_CONFIG_ATTRS = {
    "save_recordings": ("SAVE_RECORDINGS", bool),
    "random_voice": ("RANDOM_VOICE", bool),
    "playback_own_recording": ("PLAYBACK_OWN_RECORDING", bool),
    "max_record_seconds": ("MAX_RECORD_SECONDS", float),
    "silence_timeout": ("SILENCE_TIMEOUT", float),
    "silence_threshold": ("SILENCE_THRESHOLD", float),
    "phrase_gen_window_sentences": ("PHRASE_GEN_WINDOW_SENTENCES", int),
    "phrase_gen_window_repeats": ("PHRASE_GEN_WINDOW_REPEATS", int),
    "phrase_gen_level": ("PHRASE_GEN_LEVEL", int),
}

# settings.json key -> config attribute holding the value the running app
# currently uses, for the handler-driven settings the controller applies
# live (the on_*_changed handlers keep these attributes current). Together
# with LIVE_CONFIG_ATTRS this lets the "Default" reset skip no-op
# dispatches - see SettingsGlue._default_differs_from_live. Restart-only
# keys are absent on purpose: their dispatch would be persist-only, and
# the reset has already removed the persisted overrides.
SETTING_LIVE_ATTRS = {
    "user_name": "USER_NAME",
    "voice": "TTS_VOICE",
    "phrase_length": "PHRASE_LENGTH",
    "translation_language": "TRANSLATION_LANGUAGE",
    "reference_speed": "REFERENCE_SPEED",
    "show_face": "SHOW_FACE",
    "show_prosody": "SHOW_PROSODY",
    "practice_text_collapsed": "PRACTICE_TEXT_COLLAPSED",
    "practice_text_file": "PRACTICE_TEXT_FILE",
}


class SettingsGlue:
    """Persist one setting, mirror it into the settings window, reset all.

    Composed by main.py with three controller touchpoints:
      * ``report_error(message)`` - shows an error in the main window
        (view.append_error_msg).
      * ``get_window()`` - returns the currently open SettingsWindow or
        None; the controller keeps owning the window instance.
      * ``dispatch(key, value)`` - main.py on_setting_changed; used by the
        reset to re-apply default values live.
    """

    def __init__(self, report_error: Callable[[str], None],
                 get_window: Callable[[], Optional[object]],
                 dispatch: Callable[[str, object], None]):
        self._report_error = report_error
        self._get_window = get_window
        self._dispatch = dispatch
        # Set only while reset_to_defaults re-applies the defaults (see
        # persist); written and read on the Tk main thread only.
        self._suppress_persist = False

    def persist(self, key: str, value) -> bool:
        """Save one UI setting to settings.json, reporting failure in the UI.

        No-op while _suppress_persist is set (the "Default" reset just
        removed every override from settings.json; the handlers re-run to
        apply the default values live and must not write them back as
        overrides). Returns True on success (or when suppressed).
        """
        if self._suppress_persist:
            return True
        if config.save_user_setting(key, value):
            return True
        self._report_error(f"Could not save {key} to settings.json.")
        return False

    def persist_and_apply_live(self, key: str, value):
        """Persist a table-driven key and apply it to config when live.

        The catch-all path of on_setting_changed: keys the runtime re-reads
        from the config module get the attribute updated immediately
        (LIVE_CONFIG_ATTRS); restart-only keys are just persisted (the
        settings window shows the pending-restart hint).
        """
        self.persist(key, value)
        live = LIVE_CONFIG_ATTRS.get(key)
        if live is not None:
            attr, cast = live
            setattr(config, attr, cast(value))
            logging.info(f"Applied live setting config.{attr} = {value!r}.")

    def sync_window(self, key: str, value):
        """Mirror a main-window setting change into the settings window.

        No-op when the window is closed. set_value never re-emits, so a
        change that originated in the settings window cannot loop back
        through here.
        """
        window = self._get_window()
        if window is not None and window.exists():
            window.set_value(key, value)

    def reset_to_defaults(self) -> bool:
        """Reset every user setting to its default ("Default" button).

        Two steps, matching the chosen semantics: first every override is
        removed from settings.json (defaults live in the code, so an empty
        file IS the default state), then the known default values are
        pushed through the normal on_setting_changed dispatch to take
        effect live - with persistence suppressed, so the applied defaults
        are not written straight back as overrides. Returns True on success.
        """
        if not config.reset_user_settings():
            self._report_error("Could not reset settings.json.")
            return False
        logging.info("Settings reset to defaults; applying live values.")
        self._suppress_persist = True
        try:
            for key, value in config.default_user_settings().items():
                # Dispatch only actual changes: re-applying an already-default
                # voice or phrase length would needlessly regenerate the
                # current phrase (their handlers regenerate on every call).
                if self._default_differs_from_live(key, value):
                    self._dispatch(key, value)
        finally:
            self._suppress_persist = False
        return True

    def _default_differs_from_live(self, key: str, value) -> bool:
        """True when applying default *value* for *key* would change anything.

        The comparison target is the config attribute the running app reads
        (the handlers and LIVE_CONFIG_ATTRS keep those current). Restart-only
        keys always answer False: their reset dispatch would be persist-only,
        and reset_to_defaults already removed the overrides from the file.
        Numbers compare as floats (an int/float mismatch is not a change) and
        the practice-text path compares normalized, mirroring
        SettingsWindow._differs_from_runtime.
        """
        live = LIVE_CONFIG_ATTRS.get(key)
        attr = live[0] if live is not None else SETTING_LIVE_ATTRS.get(key)
        if attr is None:
            return False
        current = getattr(config, attr)
        if key == "practice_text_file":
            return (os.path.normcase(os.path.normpath(str(current)))
                    != os.path.normcase(os.path.normpath(str(value))))
        if isinstance(value, (int, float)) and not isinstance(value, bool) \
                and isinstance(current, (int, float)):
            return float(current) != float(value)
        return current != value
