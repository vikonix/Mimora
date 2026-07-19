# Configuration

Key options in [`mimora/config.py`](../mimora/config.py) (overridable via [`config/settings.json`](../config/settings.json)). `config/settings.json` is machine-local and not committed: copy [`config/settings.example.json`](../config/settings.example.json) to start, or just launch the app - every key (and the file itself) is optional and falls back to the built-in defaults.

| Setting | Default | Description |
|---|---|---|
| `ENGINE` | `phoneme` | Active scoring engine: `phoneme` (**default**), `acoustic` or `none`. Only engines available for the practice language are offered (settings.json `"engine"`). |
| `PRACTICE_LANGUAGE` | `english` | Practice language (settings.json `"practice_language"`, restart to apply). One entry in `LANGUAGE_PROFILES`, assembled from the per-language modules in `mimora/languages/`; adding a language is adding a profile module plus an engine calibration. |
| `ACCENT` | per language | Regional variant of the practice language (settings.json `"accent"`, restart to apply): `american`/`british` for English, `castilian` for Spanish. The legacy `"english_accent"` key is still read as a fallback and migrated on the next save. |
| `WAV2VEC2_PHONEME_MODEL_NAME` | `facebook/wav2vec2-xlsr-53-espeak-cv-ft` | Phoneme-ASR model for the **default** `phoneme` engine (emits espeak-style IPA). |
| `WAV2VEC2_MODEL_NAME` | `facebook/wav2vec2-large-960h` | Embedding/transcription model used only by the `acoustic` engine. |
| `WAV2VEC2_DEVICE` | `DEVICE` (cuda/cpu) | Device for the active engine's Wav2Vec2 model. Set to `"cpu"` to avoid VRAM contention with llama_cpp / Kokoro. |
| `PRONUNCIATION_SCORE_THRESHOLD` | `70.0` | Target score (0-100): feeds each engine's `result.passed`, which the app does not enforce yet (reserved for a future pass/repeat gate). |
| `PRACTICE_TEXT_FILE` | per language | Source text pre-loaded into the input panel; defaults to the active language's starter text (`texts/practice_text.txt` for English, `texts/practice_text_es.txt` for Spanish). |
| `PHRASE_GEN_TEMPERATURE` / `PHRASE_GEN_MAX_TOKENS` | `0.7` / `40` | Phrase-generation sampling. |
| `PHRASE_GEN_WINDOW_SENTENCES` | `5` | Sentences of the source text sent to the model per request (sliding window). |
| `PHRASE_GEN_WINDOW_REPEATS` | `5` | Phrases generated per window position before it slides forward by half its size. |
| `LLM_BACKEND` | `local_server` | `local_server` (auto-started subprocess), `lm-studio`, or `off` - no LLM is loaded or started; phrases are the practice text's own sentences, verbatim and in order. Aimed primarily at low-end machines. |
| `MAX_RECORD_SECONDS` | `20` | Safety cap on recording length. |
| `RANDOM_VOICE` | `False` | Speak every new phrase with a fresh random voice of the active language/variant (never the one just heard). Needs at least two voices. The `voice` setting is kept and used again when this is off. |
