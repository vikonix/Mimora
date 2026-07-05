# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev

"""Practice-text panel for the Mimora view layer.

The editable source-text editor with its collapse caption, the Paste/Clear
affordances and the collapsed-state one-line preview. Clipboard paste and
clearing are view-local widget operations with no application logic, so they
live here; the controller still reads the text via ``get_text``.
"""
import tkinter as tk
from tkinter import scrolledtext
from typing import Callable

from mimora import config
from mimora.ui_theme import FONT_FAMILY, THEME


class PracticePanel:
    """Collapsible practice-text editor.

    Owns the caption toggle, Paste/Clear buttons, the ScrolledText editor and
    the collapsed preview label. The collapse flag routes through
    ``on_collapsed_toggled`` so the controller can apply and persist it exactly
    like the other visibility toggles.
    """

    def __init__(self, root, on_collapsed_toggled: Callable[[], None]):
        self.root = root
        self._on_collapsed_toggled = on_collapsed_toggled

        self.frame = tk.Frame(root, bg=THEME["bg_main"])
        self.frame.pack(side=tk.TOP, fill=tk.X, padx=20, pady=(0, 8))

        # Header row above the practice text: the collapse caption on the left
        # with the Paste/Clear affordances. Translation, phrase length, voice,
        # speed and the user name all live in the Settings window (the header
        # here stays deliberately minimal - see the settings_window Field model).
        header = tk.Frame(self.frame, bg=THEME["bg_main"])
        header.pack(fill=tk.X)
        # The caption doubles as the collapse toggle for the text box: clicking
        # it hides/shows the editor (Paste/Clear go with it) to free vertical
        # space; the arrow prefix mirrors the state (see toggle).
        self.collapsed = tk.BooleanVar(value=config.PRACTICE_TEXT_COLLAPSED)
        self._caption = tk.Button(
            header, text="▾ Practice text:",
            command=self._on_caption_clicked,
            font=(FONT_FAMILY, 9, "bold"), fg=THEME["text_dim"], bg=THEME["bg_main"],
            activebackground=THEME["bg_main"], activeforeground=THEME["text"],
            bd=0, padx=0, pady=0, cursor="hand2")
        self._caption.pack(side=tk.LEFT)

        # Quick-edit affordances next to the caption. Their real job is
        # discoverability: a control visible even while the field still shows the
        # pre-filled welcome text is the clearest signal that the box is editable
        # (reviewers kept reading the box as static help text).
        self._paste_btn = tk.Button(
            header, text="Paste", command=self._paste,
            font=(FONT_FAMILY, 9), bg=THEME["bg_accent"], fg=THEME["text_accent"],
            activebackground=THEME["bg_accent_active"], activeforeground=THEME["text_bright"],
            bd=0, width=10, padx=8, pady=1, cursor="hand2")
        self._paste_btn.pack(side=tk.LEFT, padx=(10, 0))
        self._clear_btn = tk.Button(
            header, text="Clear text", command=self._clear,
            font=(FONT_FAMILY, 9), bg=THEME["bg_accent"], fg=THEME["text_accent"],
            activebackground=THEME["bg_accent_active"], activeforeground=THEME["text_bright"],
            bd=0, width=10, padx=8, pady=1, cursor="hand2")
        self._clear_btn.pack(side=tk.LEFT, padx=(6, 0))

        # Collapsed-state preview: when the editor is hidden, this shows the first
        # part of the current text (grey, italic, quoted) so the panel still hints
        # at its content without taking the editor's vertical space - the same
        # affordance a collapsed <summary> line provides. Only visible while
        # collapsed; toggle swaps it against the editor and the Paste/Clear
        # buttons, and refreshes its text on each collapse.
        self._preview = tk.Label(
            header, text="", font=(FONT_FAMILY, 9, "italic"),
            fg=THEME["text_muted"], bg=THEME["bg_main"], anchor=tk.W)

        self.text = scrolledtext.ScrolledText(
            self.frame, bg=THEME["bg_panel"], fg=THEME["text"], insertbackground=THEME["text_bright"],
            font=(FONT_FAMILY, 10), wrap=tk.WORD, bd=0, height=7,
            highlightthickness=1, highlightbackground=THEME["border"], highlightcolor=THEME["accent"],
            padx=10, pady=8)
        self.text.pack(fill=tk.X, pady=4)

        # The editor is packed above by default; hide it now if the persisted
        # state says collapsed (same late-apply idiom as the prosody toggles).
        self.toggle()

    # ------------------------------------------------------------------
    # Text access
    # ------------------------------------------------------------------
    def get_text(self) -> str:
        """Return the editable practice text (without trailing whitespace)."""
        return self.text.get("1.0", tk.END).strip()

    def set_text(self, text: str):
        """Replace the practice-text panel contents."""
        self.text.delete("1.0", tk.END)
        self.text.insert("1.0", text)
        # Keep the collapsed preview in sync when text is loaded while hidden.
        if self.collapsed.get():
            self._update_preview()

    def _paste(self):
        """Insert clipboard text into the practice field (the 'Paste' button).

        Standard paste semantics: any current selection is replaced, then the
        clipboard text is inserted at the caret. The field is focused afterwards
        so the user can keep editing. A missing or non-text clipboard is a no-op.
        """
        try:
            text = self.root.clipboard_get()
        except tk.TclError:
            return  # clipboard empty or holds non-text data
        if not text:
            return
        if self.text.tag_ranges(tk.SEL):
            self.text.delete(tk.SEL_FIRST, tk.SEL_LAST)
        self.text.insert(tk.INSERT, text)
        self.text.focus_set()

    def _clear(self):
        """Empty the practice field and focus it, ready for the user's own text."""
        self.text.delete("1.0", tk.END)
        self.text.focus_set()

    # ------------------------------------------------------------------
    # Collapse toggle
    # ------------------------------------------------------------------
    def _on_caption_clicked(self):
        """Flip the collapse flag and route it through the controller.

        The caption Button has no variable of its own, so the panel flips the
        flag here; the controller then applies and persists it exactly like the
        other visibility toggles.
        """
        self.collapsed.set(not self.collapsed.get())
        self._on_collapsed_toggled()

    def toggle(self):
        """Show/hide the practice-text editor to match the collapse flag."""
        # Return focus to the window so the spacebar record toggle keeps working.
        self.root.focus_set()
        collapsed = self.collapsed.get()
        self._caption.config(
            text="▸ Practice text:" if collapsed else "▾ Practice text:")
        # ScrolledText delegates pack/pack_forget to its outer .frame but NOT
        # winfo_manager: asking the Text itself always answers "pack" (it is
        # permanently packed inside that frame), so probe the frame instead.
        shown = self.text.frame.winfo_manager() == "pack"
        if collapsed and shown:
            self.text.pack_forget()
            self._paste_btn.pack_forget()
            self._clear_btn.pack_forget()
            # Swap the editor row for the one-line text preview next to the caption.
            self._update_preview()
            self._preview.pack(side=tk.LEFT, padx=(10, 0))
        elif not collapsed and not shown:
            # Restore the build order: the editor is the last child of the
            # panel frame, and the buttons pack after the caption (LEFT packing
            # preserves their order).
            self._preview.pack_forget()
            self.text.pack(fill=tk.X, pady=4)
            self._paste_btn.pack(side=tk.LEFT, padx=(10, 0))
            self._clear_btn.pack(side=tk.LEFT, padx=(6, 0))

    def _update_preview(self):
        """Refresh the collapsed-state preview from the current editor content.

        Shows the first ~60 characters of the practice text on a single line
        (newlines and runs of whitespace collapsed to single spaces), quoted, with
        an ellipsis when truncated. Empty text falls back to a neutral placeholder.
        """
        raw = self.text.get("1.0", "end-1c")
        collapsed_ws = " ".join(raw.split())
        if not collapsed_ws:
            self._preview.config(text="(empty)")
            return
        limit = 60
        shown = collapsed_ws[:limit].rstrip()
        if len(collapsed_ws) > limit:
            shown += "..."
        self._preview.config(text=f'"{shown}"')

    def set_collapsed(self, flag: bool):
        self.collapsed.set(bool(flag))

    def get_collapsed(self) -> bool:
        return bool(self.collapsed.get())
