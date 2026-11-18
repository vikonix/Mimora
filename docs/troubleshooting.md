# Troubleshooting

What to attach to a bug report, and where Mimora writes it. Open the issue on
[GitHub](https://github.com/vikonix/Mimora/issues), and say which operating
system and which way of installing you used.

---

## The logs

Mimora writes its logs to the `logs/` folder of the [data
directory](technical-details.md#where-mimora-keeps-your-files) (in a clone, of
the project directory).

| File | What it keeps | When it is written |
|---|---|---|
| `main.log` | everything the app logs | at every start; the next start replaces it |
| `install.log` | what was installed or downloaded | added to, never replaced (see below) |
| `llm_server.log` | the output of llama-server | at every start of the server; replaced like `main.log` |
| `hwdetect.log` | the hardware probe | one section for each probe |

**`main.log` is replaced at every start.** Copy it before you start Mimora
again, or the log of the failed run is gone.

**`install.log` has a section only after one of these:**

- the update check of an install script (`install.ps1`, `install-macos.sh`,
  `install-linux.sh`);
- the downloads of the first-run window;
- `install.py`, in a clone.

A first installation with an install script or with the Setup file writes no
section. Its messages are only on the screen: see the next section.

---

## If the installation fails

For a bug report, keep the messages of the terminal:

- **Setup file** - the installation runs in a console window. When it fails,
  the window stays open with "Mimora installation failed". Select the text
  (Ctrl+A), copy it (Ctrl+C) and paste it into the issue.
- **Install script** - run the command again, so that the messages also go to
  a file in your home folder:

  ```powershell
  # Windows (PowerShell)
  powershell -ExecutionPolicy ByPass -c "irm https://github.com/vikonix/Mimora/releases/latest/download/install.ps1 | iex" 2>&1 | Tee-Object -FilePath "$HOME\mimora-install.txt"
  ```

  ```bash
  # macOS
  curl -LsSf https://github.com/vikonix/Mimora/releases/latest/download/install-macos.sh | sh 2>&1 | tee ~/mimora-install.txt

  # Linux
  wget -qO- https://github.com/vikonix/Mimora/releases/latest/download/install-linux.sh | sh 2>&1 | tee ~/mimora-install.txt
  ```

  Attach `mimora-install.txt` to the issue. For an update, attach
  `logs/install.log` as well.

## If Mimora fails after the installation

Attach `main.log`, and `install.log` if the first start downloaded something.
When the phrases do not come (the chat model), attach `llm_server.log` too.
