#!/bin/sh
# Installs Mimora on macOS with uv, for a user who uses a terminal.
# It also runs from a downloaded file:
#   sh install-macos.sh
#
# Install or upgrade (macOS has curl but not wget):
#   curl -LsSf https://github.com/vikonix/Mimora/releases/latest/download/install-macos.sh | sh
#
# Uninstall:
#   curl -LsSf https://github.com/vikonix/Mimora/releases/latest/download/install-macos.sh | sh -s -- --uninstall
#
# Environment variables:
#   MIMORA_VERSION      install this exact version, for example 1.2.0rc1.
#                       After an exact version, "uv tool upgrade mimora" does
#                       not upgrade. Run this script again: with the new version
#                       in the variable, or without the variable for the newest
#                       normal version (a pre-release comes only with the variable).
#
# The models are not downloaded here: the first start of Mimora asks for them.
# This script does not change shell startup files. It prints the commands
# that do.
#
# POSIX sh, not bash, the same as install-linux.sh.

set -eu

step() {
    printf '==> %s\n' "$1"
}

warn() {
    printf 'WARNING: %s\n' "$1" >&2
}

# The rule that opens a section of logs/install.log, as in install.py.
log_rule='======================================================================'

die() {
    printf 'ERROR: %s\n' "$1" >&2
    exit 1
}

# The first output of an installation: what is installed, and that it can
# take long. Without it, the first line names uv, which the user did not ask for.
banner() {
    printf '\nInstalling %s, a local pronunciation trainer.\n' "$1"
    printf '  This downloads uv, Python, PyTorch and the other dependencies (up to several GB).\n'
    printf '  Depending on the speed of your internet connection, this can take from a few\n'
    printf '  minutes to half an hour or more. Do not close this window.\n\n'
}

# Downloads a URL to standard output with curl or wget.
download() {
    if command -v curl >/dev/null 2>&1; then
        curl -LsSf "$1"
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- "$1"
    else
        die "Neither curl nor wget was found. Install one of them and run this command again."
    fi
}

# Prints the full path of uv, or nothing. The full path is necessary because
# the uv installer changes PATH only for new terminal windows.
find_uv() {
    if command -v uv >/dev/null 2>&1; then
        command -v uv
        return 0
    fi
    # The directories where the official uv installer puts uv, in the order
    # that the installer itself uses.
    for dir in "${UV_INSTALL_DIR:-}" "${XDG_BIN_HOME:-}" "$HOME/.local/bin"; do
        if [ -n "$dir" ] && [ -x "$dir/uv" ]; then
            printf '%s\n' "$dir/uv"
            return 0
        fi
    done
    return 0
}

install_uv() {
    step "Installing uv (the official installer from Astral)"
    download https://astral.sh/uv/install.sh | sh \
        || die "The uv installation failed. Read the messages above."
}

# Sets python_version for this Mac, and python_note when a newer macOS would
# get a newer Python. Two kinds of Mac are narrower than requires-python:
# - an Intel Mac stays on 3.12: the pinned torch==2.2.2 has no wheels newer
#   than cp312, so the resolution stops on any newer Python;
# - Apple Silicon needs macOS 14 or newer for 3.14: the cp314 wheels of
#   onnxruntime (which supertonic pulls) are built for macOS 14.
# The architecture comes from uname -m and not from sysctl hw.optional.arm64.
# In a terminal under Rosetta uname -m says x86_64 on Apple Silicon too, and
# 3.12 then works with a uv of either architecture. sysctl would say arm64,
# and 3.14 with a uv that runs as x86_64 gets the Intel torch pin and stops.
choose_python() {
    python_version=3.12
    python_note=""
    if [ "$(uname -m)" != arm64 ]; then
        return 0
    fi
    macos_major=$(sw_vers -productVersion 2>/dev/null | cut -d. -f1)
    case "$macos_major" in
        "" | *[!0-9]*) return 0 ;;
    esac
    if [ "$macos_major" -ge 14 ]; then
        python_version=3.14
    else
        python_note="Python 3.12: Mimora uses Python 3.14 on macOS 14 and newer"
    fi
}

data_dir() {
    if [ -n "${MIMORA_HOME:-}" ]; then
        printf '%s\n' "$MIMORA_HOME"
    else
        printf '%s\n' "$HOME/Library/Application Support/Mimora"
    fi
}

# The update check (trial_update) writes into logs/install.log of the data
# directory: the same append-only file that install.py and the first start of
# Mimora write, so that one file tells what was installed on this computer
# and how. Each run is one section, in the format of install.py.
log_line() {
    printf '    %s\n' "$1" >>"$install_log"
}

# Writes one line into the log only when this run made an update check. A
# first installation has no check and writes no log.
log_result() {
    if [ -n "$install_log" ]; then
        log_line "$1"
    fi
}

# The macOS version and the architecture, for the log.
system_info() {
    printf 'macOS %s, %s\n' "$(sw_vers -productVersion 2>/dev/null || true)" "$(uname -m)"
}

# Prints the version and Python of the installed Mimora, for example
# "Mimora 1.2.0, Python 3.12.3", or nothing when they cannot be read.
# $1 is the full path of uv.
installed_version() {
    installed_python="$("$1" tool dir)/mimora/bin/python"
    if [ -x "$installed_python" ]; then
        "$installed_python" -c 'import platform, mimora; print("Mimora", mimora.__version__ + ", Python", platform.python_version())' 2>/dev/null || true
    fi
}

# Installs Mimora with the same options into a temporary tool directory
# first, when Mimora is already installed. When uv changes the Python of a
# tool, it deletes the old environment before the new one is complete, so a
# failed update would leave no Mimora at all. When the check fails, the
# installed Mimora is not changed. The check fills the uv cache, so the
# installation after it downloads nothing again.
# A first installation has nothing to keep, so it gets no check.
# $1 is the full path of uv; the other arguments are those of
# "uv tool install". Sets install_log when it writes the log.
trial_update() {
    uv_path=$1
    shift
    tool_dir=$("$uv_path" tool dir) || return 0
    if [ -z "$tool_dir" ] || [ ! -d "$tool_dir/mimora" ]; then
        return 0
    fi
    # Beside the real tool directory, so usually on the same disk as the uv
    # cache: uv then links the files from the cache instead of copying them.
    trial_dir="${tool_dir%/}-mimora-trial"
    log_dir="$(data_dir)/logs"
    mkdir -p "$log_dir" || die "Could not create the log directory $log_dir"
    install_log="$log_dir/install.log"
    # An interrupted check (Ctrl+C) leaves its directory behind. It is
    # removed here, before the next check.
    rm -rf "$trial_dir" \
        || die "Could not delete $trial_dir, which an earlier check left. Delete it yourself and run this command again."

    step "Checking the update in a separate environment first. The installed Mimora stays until the check passes."
    printf '  uv shows no progress bars in this step. Its messages also go to %s\n' "$install_log"
    # A blank line only between two sections, so that a new file opens with
    # the rule, as in install.py.
    if [ -s "$install_log" ]; then
        printf '\n' >>"$install_log"
    fi
    printf '%s\nMimora update check (install-macos.sh)\n%s\n' "$log_rule" "$log_rule" >>"$install_log"
    log_line "started: $(date '+%Y-%m-%d %H:%M:%S %z')"
    log_line "requested: $package"
    installed=$(installed_version "$uv_path")
    log_line "installed: ${installed:-unknown (the installed environment did not answer)}"
    log_line "uv: $("$uv_path" --version 2>/dev/null || true)"
    log_line "system: $(system_info)"
    log_line "command: uv tool install $*"

    # The loop puts each message on the screen and into the log, where it
    # gets the prefix that install.py gives the output of a command: so
    # the messages of uv are not mixed up with the lines of the section.
    # "|| [ -n ... ]" keeps a last line without a newline. A pipeline gives
    # the exit code of its last command (the loop), and POSIX sh has no
    # pipefail, so the exit code of uv goes through a file.
    # PATH with the temporary bin directory: without it uv warns that the
    # directory is not on PATH, which is true and does not matter here.
    status_file=$(mktemp) || die "Could not create a temporary file."
    {
        code=0
        UV_TOOL_DIR="$trial_dir" UV_TOOL_BIN_DIR="$trial_dir/bin" PATH="$trial_dir/bin:$PATH" \
            "$uv_path" tool install "$@" 2>&1 || code=$?
        printf '%s\n' "$code" >"$status_file"
    } | while IFS= read -r line || [ -n "$line" ]; do
        printf '%s\n' "$line"
        log_line "| $line"
    done
    trial_status=$(cat "$status_file")
    rm -f "$status_file"

    if [ "$trial_status" = 0 ]; then
        # --version answers without importing torch (mimora/cli.py), so it
        # costs less than a second. It finds a broken command or a broken
        # installation of the Mimora package itself.
        if trial_version=$("$trial_dir/bin/mimora" --version 2>&1); then
            log_line "check: passed"
            log_line "version: $trial_version"
        else
            log_line "check: installed, but \"mimora --version\" failed: $trial_version"
            trial_status=version
        fi
    else
        log_line "check: failed, exit code $trial_status"
    fi
    rm -rf "$trial_dir"

    if [ "$trial_status" != 0 ]; then
        log_line "result: the installed Mimora was not changed"
        die "The update failed in the check, so the installed Mimora was not changed. Read the messages above; they are also in $install_log"
    fi
    step "The check passed. Replacing the installed Mimora"
}

# Ends the installation after the real "uv tool install" failed.
install_failed() {
    log_result "result: the installation failed after the check passed"
    die "The Mimora installation failed. Read the messages above. If uv reports an unknown option, update uv ('uv self update') and run this command again."
}

install_mimora() {
    package=mimora
    if [ -n "${MIMORA_VERSION:-}" ]; then
        package="mimora==$MIMORA_VERSION"
    fi
    banner "$package"
    install_log=""

    uv=$(find_uv)
    # Set when this run installed uv: the uv installer adds its own directory
    # to PATH in the shell startup files, so a new terminal window finds it.
    uv_installed_now=""
    if [ -n "$uv" ]; then
        step "Using uv: $uv"
    else
        install_uv
        uv=$(find_uv)
        [ -n "$uv" ] || die "uv was installed, but it was not found. Open a new terminal and run this command again."
        uv_installed_now=1
    fi

    choose_python
    if [ -n "$python_note" ]; then
        step "$python_note"
    fi

    step "Installing $package (Python $python_version, PyTorch and the dependencies, this can take several minutes)"
    # --upgrade: a second run of this script is an upgrade.
    # --managed-python: always the Python build of uv, which carries Tk. A
    #   Python that is already on the Mac can be a version that Mimora does
    #   not support, or come without Tk. On macOS the Tk of the uv build
    #   draws the text with Core Text, so the Linux problem with unsmoothed
    #   text does not apply here.
    # --python "$python_version": see choose_python. A uv tool environment
    #   ignores requires-python, so the version is named here.
    # No --torch-backend: macOS has no CUDA. The flag only moves the
    #   resolution to another index, and its result can change between runs.
    #   It is necessary on Windows only (install.ps1).
    # --with "preshed<3.0.14": preshed 3.0.14 breaks the spaCy pipeline load,
    #   and Mimora stops at "Initialization Failed" (explosion/spaCy#14058).
    #   The wheels up to 1.2.0rc1 do not exclude it, and the script installs
    #   them too. uv keeps the requirement, so "uv tool upgrade" keeps it.
    #   Remove it together with the bound in pyproject.toml.
    set -- --upgrade --managed-python --python "$python_version" --with "preshed<3.0.14" "$package"
    trial_update "$uv" "$@"
    # PATH with the bin directory, for this command only: without it uv
    # warns that the directory is not on PATH, and the message at the end
    # already says what to do about PATH. That message reads the PATH of
    # the user, so the PATH of this script must not change.
    bin_dir=$("$uv" tool dir --bin)
    PATH="$bin_dir:$PATH" "$uv" tool install "$@" || install_failed
    if [ -n "$install_log" ]; then
        installed=$(installed_version "$uv")
        log_line "result: installed ${installed:-Mimora, the version could not be read}"
    fi

    printf '\nMimora is installed.\n'
    case ":$PATH:" in
        *":$bin_dir:"*)
            printf '  Start it with: mimora\n'
            ;;
        *)
            printf '  Start it with: %s/mimora\n' "$bin_dir"
            # The uv installer that this run started has already added its
            # directory to the shell startup files. When mimora is in that
            # directory, a new window finds it, and the commands below would
            # only repeat that change. Otherwise (uv was already installed,
            # or UV_TOOL_BIN_DIR moved the commands) a new window does not
            # find mimora, and the user needs those commands.
            if [ -n "$uv_installed_now" ] && [ "$(dirname "$uv")" = "$bin_dir" ]; then
                printf '  Or open a new terminal window and start it with: mimora\n'
            else
                # Shell startup files are not changed without the user's consent.
                printf '  To start it as "mimora", add %s to PATH:\n' "$bin_dir"
                printf '    %s tool update-shell   (changes your shell startup file)\n' "$uv"
                printf '  or add this line to it yourself:\n'
                printf '    export PATH="%s:$PATH"\n' "$bin_dir"
            fi
            ;;
    esac
    printf '  At the first start, Mimora asks to download its models (several GB).\n'
    # uv keeps an exact "==" requirement, so after MIMORA_VERSION
    # "uv tool upgrade mimora" does nothing.
    if [ -n "${MIMORA_VERSION:-}" ]; then
        printf '  Upgrade:   run this command again with MIMORA_VERSION=<new version> before "sh"\n'
    else
        printf '  Upgrade:   uv tool upgrade mimora   (or run this command again)\n'
    fi
    printf '  Uninstall: run this command again with "sh -s -- --uninstall"\n'
}

uninstall_mimora() {
    uv=$(find_uv)
    if [ -n "$uv" ]; then
        step "Removing the Mimora uv tool"
        "$uv" tool uninstall mimora || warn "uv could not remove Mimora. Possibly it was not installed with uv."
    else
        warn "uv was not found, so the Mimora uv tool was not removed."
    fi

    # The data directory is not deleted: it can hold several GB of models
    # that a later installation uses again, and MIMORA_HOME can point to a
    # directory that also holds other files of the user.
    printf '\nMimora is uninstalled.\n'
    data=$(data_dir)
    if [ -d "$data" ]; then
        printf '  The models, settings and logs stay in: %s\n' "$data"
        printf '  Delete this directory yourself if you do not need them.\n'
    fi
    printf '  uv itself stays installed. To remove it, see https://docs.astral.sh/uv/\n'
}

main() {
    action=install
    for arg in "$@"; do
        case "$arg" in
            --uninstall) action=uninstall ;;
            *) die "Unknown option: $arg (the only option is --uninstall)" ;;
        esac
    done

    if [ "$(uname -s)" != Darwin ]; then
        die "This script is for macOS. On Linux, use install-linux.sh. On Windows, use install.ps1."
    fi

    if [ "$action" = uninstall ]; then
        uninstall_mimora
    else
        install_mimora
    fi
}

# All the work is in functions, and this is the only line that starts it.
# When "curl | sh" or "wget | sh" receives only part of the file, this line
# is missing and nothing runs.
main "$@"
