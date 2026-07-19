# Development

## Running the tests

```bash
# Fast unit tests for every module, both engines (pure logic, no model download, offline)
python -m unittest discover -s tests -v

# Optional end-to-end check on real audio (downloads the model, needs espeak-ng).
# tests/test_speech.py exercises the acoustic engine:
python tests/test_speech.py path/to/user.wav [path/to/reference.wav]
```
