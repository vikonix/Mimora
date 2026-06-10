"""View layer for the EchoLoop pronunciation trainer.

This module holds the UI as a mixin (:class:`PronunciationTrainerUI`) that the
main controller class inherits. The mixin only builds widgets and renders state;
it relies on attributes (``self.root``, the sub-managers, per-phrase state)
initialised by the controller in ``main.py``. Because it is a mixin, every
``self.<handler>`` reference (e.g. ``self.on_generate_phrase``) still resolves to
the controller's method at runtime.
"""
import logging
import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext

import config

# Phrase-length selector labels. The label maps to generate_phrase's ``length``
# mode: LENGTH_FULL → "full" sentence, LENGTH_FEW_WORDS → "fragment".
LENGTH_FULL = "Full phrase"
LENGTH_FEW_WORDS = "Few words"


class PronunciationTrainerUI:
    """UI construction and rendering, mixed into the controller in main.py."""

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def setup_styles(self):
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure("Vertical.TScrollbar",
                             gripcount=0,
                             background="#1a1a1e",
                             troughcolor="#121214",
                             bordercolor="#121214",
                             arrowcolor="#8a2be2")

        # Dark-theme styling for the reference-speed combobox.
        self.style.configure("TCombobox",
                             fieldbackground="#1f1430",
                             background="#1f1430",
                             foreground="#d6c2ff",
                             arrowcolor="#8a2be2",
                             bordercolor="#25252a",
                             relief="flat")
        self.style.map("TCombobox",
                       fieldbackground=[("readonly", "#1f1430")],
                       foreground=[("readonly", "#d6c2ff")])
        # The popdown list is a classic Tk Listbox, themed via the option DB.
        self.root.option_add("*TCombobox*Listbox.background", "#1a1a1e")
        self.root.option_add("*TCombobox*Listbox.foreground", "#d6c2ff")
        self.root.option_add("*TCombobox*Listbox.selectBackground", "#2a1a45")
        self.root.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")

    def _make_button(self, parent, text, command):
        """Create a consistently styled dark-theme button."""
        return tk.Button(parent, text=text, command=command,
                         font=("Segoe UI", 10, "bold"),
                         bg="#1f1430", fg="#d6c2ff",
                         activebackground="#2a1a45", activeforeground="#ffffff",
                         bd=0, padx=12, pady=6, cursor="hand2",
                         disabledforeground="#555560")

    def build_ui(self):
        # 1. Header
        header_frame = tk.Frame(self.root, bg="#121214", height=60)
        header_frame.pack(side=tk.TOP, fill=tk.X, padx=20, pady=10)

        tk.Label(header_frame, text="ECHOLOOP • Pronunciation",
                 font=("Segoe UI", 16, "bold"), fg="#8a2be2", bg="#121214").pack(side=tk.LEFT)

        tk.Label(header_frame, text=config.TARGET_LANGUAGE,
                 font=("Segoe UI", 9, "bold"), fg="#a0a0a5", bg="#1a1a1e",
                 padx=10, pady=4, bd=0).pack(side=tk.RIGHT)

        # 2. Status bar (absolute bottom)
        self.status_bar = tk.Frame(self.root, bg="#1a1a1e", height=30)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.status_label = tk.Label(self.status_bar, text="Status: Starting...",
                                     font=("Segoe UI", 9), fg="#00e676", bg="#1a1a1e")
        self.status_label.pack(side=tk.LEFT, padx=15, pady=4)

        self.stats_label = tk.Label(self.status_bar,
                                    text=f"Last score: -- | Pass ≥ {config.PRONUNCE_SCORE_THRESHOLD:.0f}",
                                    font=("Segoe UI", 9), fg="#a0a0a5", bg="#1a1a1e")
        self.stats_label.pack(side=tk.RIGHT, padx=15, pady=4)

        # 3. Bottom control panel (mic + instruction + replay buttons)
        control_frame = tk.Frame(self.root, bg="#121214")
        control_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=20, pady=10)

        self.btn_canvas = tk.Canvas(control_frame, width=100, height=100, bg="#121214",
                                    highlightthickness=0, cursor="hand2")
        self.btn_canvas.pack(pady=5)
        self.btn_canvas.bind("<ButtonPress-1>", lambda e: self.on_gui_btn_press())
        self.btn_canvas.bind("<ButtonRelease-1>", lambda e: self.on_gui_btn_release())
        self.draw_mic_button("loading")

        self.instruction_label = tk.Label(control_frame, text="Loading components...",
                                          font=("Segoe UI", 10), fg="#a0a0a5", bg="#121214")
        self.instruction_label.pack(pady=5)

        replay_frame = tk.Frame(control_frame, bg="#121214")
        replay_frame.pack(pady=5)
        # Small diagnostic button (leftmost): run the reference through analysis
        # instead of a recording (it should score near 100 against itself).
        self.test_btn = tk.Button(replay_frame, text="Test", command=self.on_test_reference,
                                  font=("Segoe UI", 8), bg="#1a1a1e", fg="#6272a4",
                                  activebackground="#25252a", activeforeground="#8be9fd",
                                  bd=0, padx=8, pady=3, cursor="hand2",
                                  disabledforeground="#3a3a40")
        self.test_btn.pack(side=tk.LEFT, padx=5)
        self.test_btn.config(state=tk.DISABLED)

        self.user_btn = self._make_button(replay_frame, "▶ My recording", self.play_user_recording)
        self.user_btn.pack(side=tk.LEFT, padx=5)
        self.user_btn.config(state=tk.DISABLED)

        # 4. Source text panel (editable)
        source_frame = tk.Frame(self.root, bg="#121214")
        source_frame.pack(side=tk.TOP, fill=tk.X, padx=20, pady=(0, 5))

        tk.Label(source_frame, text="Practice text (edit freely):",
                 font=("Segoe UI", 9, "bold"), fg="#a0a0a5", bg="#121214").pack(anchor=tk.W)

        self.source_text = scrolledtext.ScrolledText(
            source_frame, bg="#1a1a1e", fg="#f8f8f2", insertbackground="#ffffff",
            font=("Segoe UI", 10), wrap=tk.WORD, bd=0, height=6,
            highlightthickness=1, highlightbackground="#25252a", highlightcolor="#8a2be2",
            padx=10, pady=8)
        self.source_text.pack(fill=tk.X, pady=4)

        # Selector row: voice and reference-playback speed share a single line.
        selectors_frame = tk.Frame(source_frame, bg="#121214")
        selectors_frame.pack(anchor=tk.E, pady=(2, 0))

        # Voice selector for the reference speech. Changing it regenerates the
        # phrase (see on_voice_changed) so the new voice is heard right away.
        tk.Label(selectors_frame, text="Voice:", font=("Segoe UI", 9),
                 fg="#a0a0a5", bg="#121214").pack(side=tk.LEFT, padx=(0, 6))
        self.voice_var = tk.StringVar(value=config.KOKORO_VOICE)
        self.voice_selector = ttk.Combobox(
            selectors_frame, textvariable=self.voice_var, state="readonly",
            width=12, values=tuple(config.KOKORO_VOICES))
        self.voice_selector.pack(side=tk.LEFT, padx=(0, 12))
        self.voice_selector.bind("<<ComboboxSelected>>", self.on_voice_changed)

        # Lower values slow the reference playback (see play_reference). Stored as
        # the displayed label and parsed back to a float by _selected_speed().
        tk.Label(selectors_frame, text="Reference speed:", font=("Segoe UI", 9),
                 fg="#a0a0a5", bg="#121214").pack(side=tk.LEFT, padx=(0, 6))
        self.playback_speed = tk.StringVar(value="1.0×")
        self.speed_selector = ttk.Combobox(
            selectors_frame, textvariable=self.playback_speed, state="readonly",
            width=5, values=("1.0×", "0.85×", "0.7×"))
        self.speed_selector.pack(side=tk.LEFT)

        # Action row: phrase-length selector alongside the Reference replay and
        # New phrase buttons. "Few words" requests a short fragment instead of a
        # full sentence; changing it regenerates the phrase (see on_length_changed).
        action_frame = tk.Frame(source_frame, bg="#121214")
        action_frame.pack(anchor=tk.E, pady=(4, 0))

        tk.Label(action_frame, text="Phrase length:", font=("Segoe UI", 9),
                 fg="#a0a0a5", bg="#121214").pack(side=tk.LEFT, padx=(0, 6))
        self.length_var = tk.StringVar(value=LENGTH_FULL)
        self.length_selector = ttk.Combobox(
            action_frame, textvariable=self.length_var, state="readonly",
            width=12, values=(LENGTH_FULL, LENGTH_FEW_WORDS))
        self.length_selector.pack(side=tk.LEFT, padx=(0, 10))
        self.length_selector.bind("<<ComboboxSelected>>", self.on_length_changed)

        self.ref_btn = self._make_button(action_frame, "▶ Reference", self.play_reference)
        self.ref_btn.pack(side=tk.LEFT, padx=(0, 6))
        self.ref_btn.config(state=tk.DISABLED)

        self.generate_btn = self._make_button(action_frame, "🎲 New phrase", self.on_generate_phrase)
        self.generate_btn.pack(side=tk.LEFT)
        self.generate_btn.config(state=tk.DISABLED)

        # 5. Current phrase card
        phrase_frame = tk.Frame(self.root, bg="#1a1a1e")
        phrase_frame.pack(side=tk.TOP, fill=tk.X, padx=20, pady=5)

        self.phrase_label = tk.Label(phrase_frame, text="—", font=("Segoe UI", 15, "bold"),
                                     fg="#8be9fd", bg="#1a1a1e", wraplength=440, justify=tk.LEFT)
        self.phrase_label.pack(anchor=tk.W, padx=12, pady=(10, 10))

        # 5b. Prosody panel — pitch (F0) and energy sparklines, you vs reference.
        prosody_frame = tk.Frame(self.root, bg="#121214")
        prosody_frame.pack(side=tk.TOP, fill=tk.X, padx=20, pady=(0, 5))

        prosody_header = tk.Frame(prosody_frame, bg="#121214")
        prosody_header.pack(fill=tk.X)
        tk.Label(prosody_header, text="Prosody", font=("Segoe UI", 9, "bold"),
                 fg="#8a2be2", bg="#121214").pack(side=tk.LEFT)
        tk.Label(prosody_header, text="● reference", font=("Segoe UI", 8),
                 fg="#ff79c6", bg="#121214").pack(side=tk.RIGHT, padx=(8, 0))
        tk.Label(prosody_header, text="● you", font=("Segoe UI", 8),
                 fg="#8be9fd", bg="#121214").pack(side=tk.RIGHT)

        tk.Label(prosody_frame, text="Pitch (F0) — intonation, low ↔ high", font=("Segoe UI", 8),
                 fg="#6272a4", bg="#121214").pack(anchor=tk.W)
        self.f0_canvas = tk.Canvas(prosody_frame, height=46, bg="#1a1a1e",
                                   highlightthickness=1, highlightbackground="#25252a")
        self.f0_canvas.pack(fill=tk.X, pady=(0, 4))

        tk.Label(prosody_frame, text="Energy — stress pattern", font=("Segoe UI", 8),
                 fg="#6272a4", bg="#121214").pack(anchor=tk.W)
        self.en_canvas = tk.Canvas(prosody_frame, height=46, bg="#1a1a1e",
                                   highlightthickness=1, highlightbackground="#25252a")
        self.en_canvas.pack(fill=tk.X)

        # Reading hint: horizontal axis is time (stretched to equal width for both),
        # so the goal is matching the *shape* of the reference, not exact overlap.
        tk.Label(prosody_frame,
                 text="Time runs left→right (stretched to equal width). Aim to match the reference shape.",
                 font=("Segoe UI", 8), fg="#6272a4", bg="#121214",
                 wraplength=460, justify=tk.LEFT).pack(anchor=tk.W, pady=(3, 0))

        # fill=X canvases change width on resize, so redraw from the cached prosody.
        self.f0_canvas.bind("<Configure>", lambda e: self._redraw_prosody())
        self.en_canvas.bind("<Configure>", lambda e: self._redraw_prosody())

        # 6. Feedback log (fills remaining space)
        feedback_frame = tk.Frame(self.root, bg="#121214")
        feedback_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=20, pady=5)

        self.feedback_display = scrolledtext.ScrolledText(
            feedback_frame, bg="#1a1a1e", fg="#f8f8f2", insertbackground="#ffffff",
            font=("Segoe UI", 11), wrap=tk.WORD, bd=0,
            highlightthickness=1, highlightbackground="#25252a", highlightcolor="#8a2be2",
            padx=15, pady=15, spacing2=4, spacing3=8)
        self.feedback_display.pack(fill=tk.BOTH, expand=True)
        self.feedback_display.configure(state=tk.DISABLED)

        self.feedback_display.tag_configure("system", foreground="#6272a4", font=("Segoe UI", 10, "italic"))
        self.feedback_display.tag_configure("good", foreground="#50fa7b", font=("Segoe UI", 11, "bold"))
        self.feedback_display.tag_configure("bad", foreground="#ff5555", font=("Segoe UI", 11, "bold"))
        self.feedback_display.tag_configure("label", foreground="#a0a0a5", font=("Segoe UI", 10))
        self.feedback_display.tag_configure("text", foreground="#f1f1f6", font=("Segoe UI", 11))
        # Monospace tag for phoneme strings so they align and read clearly.
        self.feedback_display.tag_configure("mono", foreground="#a0a0a5", font=("Consolas", 10))
        # Amber tag for "no word errors but score still low" guidance.
        self.feedback_display.tag_configure("warn", foreground="#ffb86c", font=("Segoe UI", 11))

    def draw_mic_button(self, state):
        self.btn_canvas.delete("all")
        cx, cy = 50, 50
        r_outer, r_inner = 42, 34
        palette = {
            "loading":   ("#1e1e24", "#44475a", "⌛"),
            "idle":      ("#1f1430", "#8a2be2", "🎤"),
            "recording": ("#3a0c10", "#ff5555", "🔴"),
            "processing":("#36220f", "#ffb86c", "⚡"),
            "speaking":  ("#0f2c1d", "#50fa7b", "🔊"),
        }
        bg_color, outline_color, emoji = palette.get(state, ("#1e1e24", "#44475a", "🎤"))
        self.btn_canvas.create_oval(cx - r_outer, cy - r_outer, cx + r_outer, cy + r_outer,
                                    fill="", outline=outline_color, width=3)
        self.btn_canvas.create_oval(cx - r_inner, cy - r_inner, cx + r_inner, cy + r_inner,
                                    fill=bg_color, outline="")
        self.btn_canvas.create_text(cx, cy, text=emoji, font=("Segoe UI", 20), fill="#ffffff")

    # ------------------------------------------------------------------
    # Feedback / status helpers (always called on the main thread)
    # ------------------------------------------------------------------
    def append_system_msg(self, text: str):
        # System messages are intentionally kept out of the on-screen feedback
        # panel so it stays focused on pronunciation feedback. They are still
        # written to the log file for diagnostics.
        logging.info(f"[System] {text}")

    def update_status(self, text: str, color: str = "#a0a0a5"):
        self.status_label.configure(text=f"Status: {text}", fg=color)

    def update_instruction(self, text: str):
        self.instruction_label.configure(text=text)

    def update_score_stats(self, score: float):
        self.stats_label.configure(
            text=f"Last score: {score:.0f} | Pass ≥ {config.PRONUNCE_SCORE_THRESHOLD:.0f}")

    # ------------------------------------------------------------------
    # Prosody drawing
    # ------------------------------------------------------------------
    @staticmethod
    def _resample_series(values, target: int = 160):
        """Evenly resample a 1-D sequence down to ``target`` points for plotting.

        Prosody contours can be hundreds of frames long; thinning keeps the
        sparkline light without changing its shape.
        """
        n = len(values)
        if n <= target:
            return list(values)
        step = (n - 1) / (target - 1)
        return [values[int(round(i * step))] for i in range(target)]

    def _draw_prosody(self, canvas, series):
        """Draw contours onto ``canvas``. ``series`` is a list of (values, color).

        All series share one vertical scale so they are directly comparable.
        No-op until the canvas has been laid out (winfo_width > 1).
        """
        canvas.delete("all")
        width, height = canvas.winfo_width(), canvas.winfo_height()
        if width <= 1 or height <= 1:
            return
        pad_x, pad_y = 4, 4
        plot_w = max(1, width - 2 * pad_x)
        plot_h = max(1, height - 2 * pad_y)

        all_values = [v for values, _ in series for v in values]
        if not all_values:
            return
        lo, hi = min(all_values), max(all_values)
        span = (hi - lo) or 1.0

        for values, color in series:
            points = self._resample_series(values)
            if len(points) < 2:
                continue
            coords = []
            for i, value in enumerate(points):
                x = pad_x + (i / (len(points) - 1)) * plot_w
                y = pad_y + (1 - (value - lo) / span) * plot_h
                coords.extend((x, y))
            canvas.create_line(*coords, fill=color, width=2, smooth=True)

    def _redraw_prosody(self):
        """Redraw both prosody canvases from the cached result (e.g. after resize)."""
        prosody = self._last_prosody
        if not prosody:
            return
        self._draw_prosody(self.f0_canvas, [
            (prosody.get("ref_f0", []), "#ff79c6"),   # reference
            (prosody.get("f0", []), "#8be9fd"),        # you
        ])
        self._draw_prosody(self.en_canvas, [
            (prosody.get("ref_energy", []), "#ff79c6"),
            (prosody.get("energy", []), "#8be9fd"),
        ])

    # ------------------------------------------------------------------
    # Feedback rendering
    # ------------------------------------------------------------------
    def _show_feedback(self, result: "pronounce.PronunciationResult"):
        self.feedback_display.configure(state=tk.NORMAL)
        tag = "good" if result.passed else "bad"
        self.feedback_display.insert(tk.END, f"Score: {result.score:.0f}/100 ", tag)
        self.feedback_display.insert(tk.END, "(passed)\n" if result.passed else "(try again)\n", tag)
        # First line: the expected phrase, with mispronounced words shown in red.
        error_words = {w.lower() for w in result.words_with_errors}
        self.feedback_display.insert(tk.END, "Phrase: ", "label")
        for token in (self.current_phrase or "—").split():
            is_error = token.lower().strip(".,!?;:\"") in error_words
            self.feedback_display.insert(tk.END, token + " ", "bad" if is_error else "text")
        self.feedback_display.insert(tk.END, "\n")

        # Second line: what the recognizer actually heard.
        self.feedback_display.insert(tk.END, "Heard: ", "label")
        self.feedback_display.insert(tk.END, f"{result.transcription or '—'}\n", "text")

        # When the words are right but the score still failed, the gap is prosodic.
        if not result.word_errors and not result.passed:
            self.feedback_display.insert(
                tk.END, "Words are correct, but your rhythm/intonation differ from the "
                "reference — match the curves above.\n", "warn")

        self.feedback_display.insert(tk.END, "\n")
        self.feedback_display.configure(state=tk.DISABLED)
        self.feedback_display.see(tk.END)

        # Cache prosody and draw the sparklines (you vs reference).
        self._last_prosody = result.prosody or {}
        self.root.update_idletasks()  # ensure the canvases have a real width/height
        self._redraw_prosody()

        self.update_score_stats(result.score)

        # Replay buttons available now that we have both signals.
        self.ref_btn.config(state=tk.NORMAL)
        self.user_btn.config(state=tk.NORMAL)
        self.draw_mic_button("idle")

        if result.passed:
            self.update_status("Passed!", "#50fa7b")
            self.update_instruction("Nice! Click 'New phrase' to continue, or repeat to refine.")
        else:
            self.update_status("Keep practicing", "#ffb86c")
            self.update_instruction("Try again: hold SPACE or click the mic to repeat. ▶ Reference replays the example.")
