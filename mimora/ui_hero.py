# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev

"""Hero card for the Mimora view layer.

THE screen object ("one screen - one task"): the current phrase with its
translation, and the score row beneath them (big score digit, verdict, WORK ON
phoneme badges, the articulation face). Every word in the phrase is clickable
to hear it slowly; badges speak an example word for the phoneme.

The card is passive like the rest of the view: widget callbacks forward to the
two plain callables passed in at construction (``on_word_clicked``,
``on_sound_example``); the facade drives it through the public methods.
"""
import tkinter as tk
from typing import Callable, Optional

from mimora import config
from mimora.face_widget import FaceWidget
from mimora.phoneme_examples import example_for
from mimora.ui_theme import (
    FONT_FAMILY,
    FONT_SIZE_BODY,
    FONT_SIZE_CAPTION,
    FONT_SIZE_PHRASE,
    FONT_SIZE_SCORE,
    FONT_SIZE_TRANSLATION,
    THEME,
    Tooltip,
)


class HeroCard:
    """The phrase + score card, including the talking face.

    Owns the phrase Text (word tags, miss underlines, hover), the translation
    label, the score column, the WORK ON badges, the interactive-feedback hint
    and the FaceWidget. Built and packed into ``root`` at construction.
    """

    def __init__(self, root,
                 on_word_clicked: Callable[[str], None],
                 on_sound_example: Callable[[str], None]):
        self.root = root
        self._on_word_clicked = on_word_clicked
        self._on_sound_example = on_sound_example

        # Drawn on bg_card, the lightest of the three surfaces (bg_main <
        # bg_panel < bg_card), so the card visually leads the whole window.
        self.frame = tk.Frame(root, bg=THEME["bg_card"],
                              highlightthickness=1, highlightbackground=THEME["border"])
        self.frame.pack(side=tk.TOP, fill=tk.X, padx=20, pady=(8, 10))

        # Phrase: a read-only tk.Text (not a Label) so individual words can carry
        # tags - every word is clickable to hear it slowly (the "word" tag) and
        # the mispronounced ones also get the "miss" underline. Disabled +
        # takefocus=0 keeps it from swallowing the spacebar record toggle
        # (main.py gates hotkeys on a focused Text widget).
        self.phrase_text = tk.Text(
            self.frame, font=(FONT_FAMILY, FONT_SIZE_PHRASE, "bold"),
            fg=THEME["phrase"], bg=THEME["bg_card"], wrap=tk.WORD, bd=0,
            highlightthickness=0, height=1, cursor="arrow", takefocus=0,
            padx=16, pady=0)
        self.phrase_text.pack(fill=tk.X, padx=10, pady=(18, 0))
        # All content is inserted with the "center" tag: tk.Text has no
        # widget-level justify, only per-tag.
        self.phrase_text.tag_configure("center", justify=tk.CENTER)
        # "miss" marks a mispronounced word: a colored underline (kept subtle -
        # the word itself stays in the phrase color). underlinefg needs
        # Tk >= 8.6.6; older Tks fall back to painting the word itself red,
        # which keeps the signal without the separate underline color.
        try:
            self.phrase_text.tag_configure("miss", underline=True,
                                           underlinefg=THEME["bad"])
        except tk.TclError:
            self.phrase_text.tag_configure("miss", underline=True,
                                           foreground=THEME["bad"])
        # Every word carries the "word" tag: click any word (not only the
        # mispronounced ones) to hear it spoken slowly. "hover" highlights the
        # single word under the pointer - applied to just that range by
        # _on_phrase_motion, because tag options are per-tag and a shared
        # background could not target one word without lighting up the rest.
        self.phrase_text.tag_configure("hover", background=THEME["bg_accent_active"])
        self.phrase_text.tag_bind("word", "<Button-1>", self._on_phrase_word_clicked)
        self.phrase_text.bind("<Motion>", self._on_phrase_motion)
        self.phrase_text.bind("<Leave>", self._clear_phrase_hover)
        # Native mouse selection stays enabled on purpose: dragging (or a double-
        # click) selects part of the phrase so it can be copied - the blue
        # partial-word highlight is that selection, not a bug. If it ever needs to
        # be suppressed (e.g. it interferes with the click-to-hear affordance),
        # block the selection-forming events without touching the single click:
        #     for seq in ("<B1-Motion>", "<Double-Button-1>", "<Triple-Button-1>"):
        #         self.phrase_text.bind(seq, lambda e: "break")
        # The Text is disabled between set_phrase calls; its height follows the
        # wrapped content (recomputed on every set_phrase and on resize).
        self.phrase_text.configure(state=tk.DISABLED)
        self.phrase_text.bind("<Configure>", lambda e: self._fit_phrase_height())
        # Right-click the phrase to copy it (whole phrase - a tk.Text has no
        # single "text" option like a Label).
        self._bind_copy_menu(self.phrase_text,
                             getter=lambda: self.phrase_text.get("1.0", "end-1c"))
        # Same "-" placeholder the old Label started with.
        self.set_phrase("-")

        # Translation inside the same card, right under the phrase: just dimmer,
        # smaller text - no separator needed. Packed/unpacked by
        # refresh_translation_ui() when a translation language is (de)selected;
        # "-" stands in until a translation arrives with the next phrase.
        self.translation_label = tk.Label(
            self.frame, text="-", font=(FONT_FAMILY, FONT_SIZE_TRANSLATION),
            fg=THEME["text_dim"], bg=THEME["bg_card"], wraplength=560,
            justify=tk.CENTER)
        # Wrap the translation to the label's real width so a long translation
        # wraps onto more lines instead of being clipped on both sides. A fixed
        # wraplength did not track the card width, so once the card was narrower
        # than that value the centred line overflowed and got cut off.
        self.translation_label.bind("<Configure>", self._wrap_translation)
        # Right-click the translation to copy it (independently of the phrase).
        self._bind_copy_menu(self.translation_label)
        self.refresh_translation_ui()

        # Bottom padding of the phrase block, then a 1px divider above the
        # score row (the card's only internal separator).
        tk.Frame(self.frame, height=14, bg=THEME["bg_card"]).pack(fill=tk.X)
        tk.Frame(self.frame, height=1, bg=THEME["border"]).pack(fill=tk.X)

        # Score row: [SCORE column] [WORK ON badges] [face]. Filled by the
        # facade's show_feedback; reset_score_row shows the empty state ("--"
        # and "record to get a score") until the first take.
        score_row = tk.Frame(self.frame, bg=THEME["bg_card"])
        score_row.pack(fill=tk.X, padx=18, pady=(10, 12))

        score_col = tk.Frame(score_row, bg=THEME["bg_card"])
        score_col.pack(side=tk.LEFT)
        tk.Label(score_col, text="SCORE", font=(FONT_FAMILY, FONT_SIZE_CAPTION, "bold"),
                 fg=THEME["text_dim"], bg=THEME["bg_card"]).pack()
        self.score_num_label = tk.Label(
            score_col, text="--", font=(FONT_FAMILY, FONT_SIZE_SCORE, "bold"),
            fg=THEME["text_dim"], bg=THEME["bg_card"])
        self.score_num_label.pack()
        self.score_verdict_label = tk.Label(
            score_col, text="record to get a score",
            font=(FONT_FAMILY, FONT_SIZE_CAPTION),
            fg=THEME["text_dim"], bg=THEME["bg_card"])
        self.score_verdict_label.pack()

        # WORK ON: caption + a row of flat phoneme badges (top problem sounds).
        # The badges are rebuilt per result (set_badges); clicking one speaks an
        # example word.
        workon_col = tk.Frame(score_row, bg=THEME["bg_card"])
        workon_col.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(20, 8))
        self.workon_caption = tk.Label(
            workon_col, text="", font=(FONT_FAMILY, FONT_SIZE_CAPTION, "bold"),
            fg=THEME["warn"], bg=THEME["bg_card"])
        self.workon_caption.pack(anchor=tk.W)
        self.badges_frame = tk.Frame(workon_col, bg=THEME["bg_card"])
        self.badges_frame.pack(anchor=tk.W, pady=(4, 0))
        # Hint under the badges: how to use the two interactive feedback
        # affordances. Packed only while a scored take is on the card (see
        # set_hint), so the empty/unscored states stay quiet.
        self.hint_label = tk.Label(
            workon_col,
            text="Click any word to hear it slowly; "
                 "click a sound for an example.",
            font=(FONT_FAMILY, FONT_SIZE_CAPTION), fg=THEME["text_muted"],
            bg=THEME["bg_card"], justify=tk.LEFT, wraplength=320)

        # The face lives in the score row (verdict indicator + talking mouth).
        # Same single FaceWidget instance the controller drives via
        # face_play_levels/face_rest; ~78px as mocked up.
        self.face = FaceWidget(score_row, size=78, bg=THEME["bg_card"],
                               face_color=THEME["face"], face_outline=THEME["border"],
                               eye_color=THEME["eyes"], mouth_color=THEME["mouth"])
        self.face.set_expression(":)")  # waiting state
        # The face has no on-main control: its visibility is a settings value
        # ("Show articulation face"), mirrored into this var and applied by
        # toggle_face(). Not packed yet - the facade calls toggle_face() once
        # the whole window is built.
        self.show_face = tk.BooleanVar(value=config.SHOW_FACE)

    # ------------------------------------------------------------------
    # Phrase text
    # ------------------------------------------------------------------
    def set_phrase(self, text: str):
        """Replace the hero phrase and refit its height (every word clickable).

        The phrase is split into words so each carries the "word" tag and can be
        clicked to hear it spoken; no word is a miss yet. The "miss" underlines
        are added later per analysis result (see mark_miss_words).
        """
        words = (text or "-").split() or ["-"]
        self._render_phrase([(word, False) for word in words])

    def _render_phrase(self, tokens: list):
        """Render the hero phrase from ``(word, is_miss)`` pairs.

        Each word gets the "word" tag (clickable to hear it) and is centered; a
        miss word additionally gets the "miss" underline. The single spaces
        between words carry only "center", so gaps are neither clickable nor
        highlighted. The Text is disabled afterwards so it never steals the
        spacebar record toggle, and any stale hover highlight is dropped first.
        """
        self.phrase_text.configure(state=tk.NORMAL)
        self.phrase_text.tag_remove("hover", "1.0", tk.END)
        self.phrase_text.delete("1.0", tk.END)
        last = len(tokens) - 1
        for i, (word, is_miss) in enumerate(tokens):
            tags = ("center", "word", "miss") if is_miss else ("center", "word")
            self.phrase_text.insert(tk.END, word, tags)
            if i < last:
                self.phrase_text.insert(tk.END, " ", ("center",))
        self.phrase_text.configure(state=tk.DISABLED)
        self._fit_phrase_height()

    def _fit_phrase_height(self):
        """Size the phrase Text (in lines) to its wrapped content.

        A tk.Text does not auto-grow like a Label: its height is a fixed line
        count, so it is recomputed after every set_phrase and on width changes
        (the <Configure> binding). No-op before the widget is laid out.
        """
        if self.phrase_text.winfo_width() <= 1:
            return
        # "update" makes Tk compute the line metrics first, so the display-line
        # count is accurate right after an insert; without it the answer can
        # lag one layout pass behind.
        lines = self.phrase_text.count("1.0", "end", "update", "displaylines")
        # Tkinter's Text.count may return the value tuple-wrapped.
        if isinstance(lines, tuple):
            lines = lines[0]
        self.phrase_text.configure(height=max(1, int(lines or 1)))

    def mark_miss_words(self, reference_words: list) -> int:
        """Re-render the hero phrase, underlining the mispronounced words.

        Rebuilds the phrase from the engine's per-word breakdown so the "miss"
        tag lands on exactly the target tokens the engine judged: a word is a
        miss when its three-level ``level`` is "bad" (the phoneme engine) or,
        without a level, when ``correct`` is False (the acoustic engine) - "ok"
        words stay unmarked. With no breakdown the phrase is left as set (no
        underlines). The Text is disabled again afterwards so it never steals the
        spacebar record toggle. Returns the number of underlined words, so the
        caller can decide whether the interactive-feedback hint applies.
        """
        if not reference_words:
            return 0
        tokens = []
        misses = 0
        for word in reference_words:
            level = word.get("level")
            if level in ("good", "ok", "bad"):
                is_miss = level == "bad"
            else:
                is_miss = not word.get("correct", True)
            if is_miss:
                misses += 1
            tokens.append((word.get("word", ""), is_miss))
        self._render_phrase(tokens)
        return misses

    def _phrase_word_at(self, event) -> str:
        """The cleaned phrase word under the pointer, or "" over a gap.

        Reads the exact "word" tag range at the event position (so in-word
        punctuation like apostrophes is not split the way a plain
        wordstart/wordend probe would) and strips surrounding punctuation.
        Returns "" when the pointer is on whitespace or outside any word.
        """
        index = self.phrase_text.index(f"@{event.x},{event.y}")
        if "word" not in self.phrase_text.tag_names(index):
            return ""
        tag_range = self.phrase_text.tag_prevrange("word", f"{index}+1c")
        if not tag_range:
            return ""
        return self.phrase_text.get(*tag_range).strip(".,!?;:\"'()").strip()

    def _on_phrase_word_clicked(self, event):
        """Speak the clicked phrase word slowly (any word, via the controller)."""
        word = self._phrase_word_at(event)
        # Return focus to the window so the spacebar record toggle keeps
        # working (main.py's global click handler skips Text widgets).
        self.root.focus_set()
        if word:
            self._on_word_clicked(word)

    def _on_phrase_motion(self, event):
        """Highlight the word under the pointer and show it is clickable.

        Moves the single-range "hover" highlight onto the word at the pointer
        and switches the cursor to a hand over words / an arrow over the gaps.
        """
        index = self.phrase_text.index(f"@{event.x},{event.y}")
        self.phrase_text.tag_remove("hover", "1.0", tk.END)
        if "word" in self.phrase_text.tag_names(index):
            tag_range = self.phrase_text.tag_prevrange("word", f"{index}+1c")
            if tag_range:
                self.phrase_text.tag_add("hover", *tag_range)
            self.phrase_text.configure(cursor="hand2")
        else:
            self.phrase_text.configure(cursor="arrow")

    def _clear_phrase_hover(self, _event=None):
        """Drop the hover highlight and restore the arrow cursor (pointer left)."""
        self.phrase_text.tag_remove("hover", "1.0", tk.END)
        self.phrase_text.configure(cursor="arrow")

    # ------------------------------------------------------------------
    # Translation
    # ------------------------------------------------------------------
    def set_translation(self, text: str):
        """Set the translation panel text, falling back to '-' when empty.

        '-' marks "language is on, but no translation yet" (e.g. right after the
        language was switched - the translation arrives with the next phrase).
        """
        self.translation_label.config(text=text.strip() if text and text.strip() else "-")

    def _wrap_translation(self, event):
        """Keep the translation wraplength equal to the label's current width.

        Guarded against the feedback loop where changing wraplength alters the
        label's height, which fires <Configure> again: only the width matters
        here and it does not change on a height-only event, so re-setting the
        same value is skipped.
        """
        if int(self.translation_label.cget("wraplength")) != event.width:
            self.translation_label.config(wraplength=max(1, event.width))

    def refresh_translation_ui(self):
        """Show or hide the translation panel to match the current setting.

        Shows the panel whenever a language is selected - in both "Full phrase"
        and "Few words" modes (fragments are translated too). Safe to call
        repeatedly - it reconciles to config.TRANSLATION_LANGUAGE without side
        effects.
        """
        show = bool(config.TRANSLATION_LANGUAGE)
        packed = self.translation_label.winfo_manager() == "pack"
        if show and not packed:
            # Inside the hero card, right under the phrase (no separator - the
            # dimmer, smaller type is the whole distinction).
            self.translation_label.pack(fill=tk.X, padx=26, pady=(8, 0),
                                        after=self.phrase_text)
        elif not show and packed:
            self.translation_label.pack_forget()

    # ------------------------------------------------------------------
    # Score row
    # ------------------------------------------------------------------
    def set_score_row(self, number: str, verdict: str, color: str):
        """Fill the SCORE column: the big number and its verdict, both in the
        already-resolved verdict color (never green for a "needs work")."""
        self.score_num_label.configure(text=number, fg=color)
        self.score_verdict_label.configure(text=verdict, fg=color)

    def set_badges(self, phonemes: list[str]):
        """Rebuild the WORK ON badges from the top problem phonemes.

        An empty list clears the row and hides the caption (the empty state and
        clean takes show no badges). Each badge whose phoneme has a known example
        word gets a hover tooltip ("as in 'put'") and speaks that example at
        normal speed on click (via ``on_sound_example``). An unknown phoneme
        stays a plain, non-playing badge.
        """
        for child in self.badges_frame.winfo_children():
            child.destroy()
        self.workon_caption.configure(text="WORK ON" if phonemes else "")
        for phoneme in phonemes:
            example = example_for(phoneme)
            badge = tk.Button(self.badges_frame, text=f"/{phoneme}/",
                              font=(FONT_FAMILY, FONT_SIZE_BODY),
                              bg=THEME["bg_accent"], fg=THEME["text_accent"],
                              activebackground=THEME["bg_button"],
                              activeforeground=THEME["text_button"],
                              bd=0, padx=10, pady=2, cursor="hand2",
                              highlightthickness=1,
                              highlightbackground=THEME["accent"])
            badge.pack(side=tk.LEFT, padx=(0, 6))
            if example:
                # Bind the current example per button via a default argument so
                # every badge keeps its own word (not the loop's last one).
                badge.configure(
                    command=lambda w=example: self._on_sound_example(w))
                Tooltip(badge, f"as in '{example}'")
            else:
                badge.configure(cursor="arrow")   # nothing to play or name

    def set_hint(self, show: bool):
        """Show or hide the interactive-feedback hint under the badges.

        Shown only while a scored take is on the card and at least one of the
        affordances it describes (underlined words, sound badges) exists.
        """
        if show:
            self.hint_label.pack(anchor=tk.W, pady=(6, 0))
        else:
            self.hint_label.pack_forget()

    def reset_score_row(self):
        """Empty state: no take scored yet (or a new recording just started)."""
        self.score_num_label.configure(text="--", fg=THEME["text_dim"])
        self.score_verdict_label.configure(text="record to get a score",
                                           fg=THEME["text_dim"])
        self.set_badges([])
        self.set_hint(False)
        # Drop any underlines from the previous take (the phrase itself stays;
        # a new phrase is replaced wholesale by set_phrase).
        self.phrase_text.tag_remove("miss", "1.0", tk.END)

    # ------------------------------------------------------------------
    # Copy-to-clipboard (right-click menu on the phrase / translation)
    # ------------------------------------------------------------------
    def _bind_copy_menu(self, widget, getter: Optional[Callable[[], str]] = None):
        """Attach a right-click 'Copy' menu that copies the widget's text.

        ``getter`` returns the text to copy; it defaults to the widget's own
        "text" option (a Label). The phrase tk.Text passes an explicit getter
        (its whole content), since a Text has no such option. Phrase and
        translation get their own binding, so each is copied independently.
        """
        if getter is None:
            getter = lambda: widget.cget("text")
        widget.bind("<Button-3>", lambda event: self._show_copy_menu(event, getter))

    def _show_copy_menu(self, event, getter: Callable[[], str]):
        """Pop up a one-item 'Copy' menu for the clicked widget, if it has text."""
        text = getter().strip()
        # Nothing useful to copy from an empty card or the "-" placeholder.
        if not text or text == "-":
            return
        menu = tk.Menu(self.root, tearoff=0)
        menu.add_command(label="Copy", command=lambda: self._copy_text(text))
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    def _copy_text(self, text: str):
        """Replace the clipboard contents with ``text``."""
        self.root.clipboard_clear()
        self.root.clipboard_append(text)

    # ------------------------------------------------------------------
    # Face (visibility + the talking mouth the controller drives)
    # ------------------------------------------------------------------
    def set_show_face(self, flag: bool):
        self.show_face.set(bool(flag))

    def get_show_face(self) -> bool:
        return bool(self.show_face.get())

    def toggle_face(self):
        """Show/hide the face in the score row (the "Face" setting)."""
        self.root.focus_set()
        shown = self.face.winfo_manager() == "pack"
        if self.show_face.get() and not shown:
            # The only side=RIGHT child of the score row, so packing order
            # relative to the other columns does not matter.
            self.face.pack(side=tk.RIGHT, padx=(8, 0))
        elif not self.show_face.get() and shown:
            self.face.pack_forget()

    def face_fps(self):
        """Frame rate of the talking-mouth animation, or None if no face is shown.

        Returns None while the face panel is hidden (the "Face" setting), not
        just when no face exists: the caller skips building the loudness track
        then, so a hidden face costs no per-playback work (envelope computation
        plus a 30 fps after-loop animating an invisible mouth).
        """
        if not self.get_show_face():
            return None
        return self.face.fps

    def face_play_levels(self, levels, fps):
        """Drive the talking mouth from a pre-computed loudness track."""
        self.face.play_levels(levels, fps=fps)

    def face_rest(self):
        """Close the talking mouth."""
        self.face.rest()
