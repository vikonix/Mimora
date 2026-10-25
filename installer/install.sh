#!/bin/sh
# Installs Mimora on macOS and Linux with uv, for a user who uses a terminal.
#
# Install or upgrade:
#   curl -LsSf https://github.com/vikonix/Mimora/releases/latest/download/install.sh | sh
#
# Uninstall:
#   curl -LsSf https://github.com/vikonix/Mimora/releases/latest/download/install.sh | sh -s -- --uninstall
#
# Environment variables:
#   MIMORA_VERSION      install this exact version, for example 1.2.0rc1.
#                       After an exact version, "uv tool upgrade mimora" does
#                       not upgrade. Run this script again: with the new version
#                       in the variable, or without the variable for the newest
#                       normal version (a pre-release comes only with the variable).
#   MIMORA_NO_SHORTCUT  any value: do not create the Linux application menu entry.
#
# The models are not downloaded here: the first start of Mimora asks for them.
# This script does not run sudo and does not change shell startup files. It
# prints the commands that need them.
#
# POSIX sh, not bash: "| sh" is dash on Debian and Ubuntu.

set -eu

step() {
    printf '==> %s\n' "$1"
}

warn() {
    printf 'WARNING: %s\n' "$1" >&2
}

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

# Succeeds when no shared library matching the pattern $1 is known to the
# dynamic linker. Fails when the library is present or the check is not
# possible: without ldconfig (NixOS, for example) a library cannot be found
# this way, and the application itself names a missing library later.
# The libraries checked:
# - libportaudio.so.2: the sounddevice wheel for Linux does not carry it, and
#   Mimora cannot record or play audio without it;
# - libgomp.so.1: the llama-server release links against it, and without it
#   the first start cannot install the phrase generator. A desktop system
#   usually has it; a minimal one (WSL, a container, a server) does not.
library_missing() {
    ldconfig_cmd=""
    for candidate in ldconfig /sbin/ldconfig /usr/sbin/ldconfig; do
        if command -v "$candidate" >/dev/null 2>&1; then
            ldconfig_cmd="$candidate"
            break
        fi
    done
    if [ -z "$ldconfig_cmd" ]; then
        return 1
    fi
    if "$ldconfig_cmd" -p 2>/dev/null | grep -q "$1"; then
        return 1
    fi
    return 0
}

# Succeeds when no Symbola font file is found. Tk cannot draw a colour emoji
# font (fonts-noto-color-emoji, the default one), so without a monochrome one
# the record button is an empty circle. Mimora still works, which is why this
# is only a hint. The font files are searched directly rather than with
# fc-list: a minimal system (a WSL or container image) can lack the fontconfig
# tools, and that is exactly where the font is missing.
symbola_missing() {
    for dir in /usr/share/fonts /usr/local/share/fonts \
            "${XDG_DATA_HOME:-$HOME/.local/share}/fonts" "$HOME/.fonts"; do
        if [ -d "$dir" ] && find "$dir" -iname 'symbola*' 2>/dev/null | grep -q .; then
            return 1
        fi
    done
    return 0
}

# Prints what to install before the first start, with the command for this
# distribution, or nothing when nothing is missing.
print_native_hints() {
    need_portaudio=""
    need_gomp=""
    need_font=""
    if library_missing 'libportaudio\.so\.2'; then
        need_portaudio=1
    fi
    if library_missing 'libgomp\.so\.1'; then
        need_gomp=1
    fi
    if symbola_missing; then
        need_font=1
    fi
    if [ -z "$need_portaudio$need_gomp$need_font" ]; then
        return 0
    fi

    printf '\nBEFORE THE FIRST START, install the missing system packages:\n'
    if [ -n "$need_portaudio" ]; then
        printf '  - PortAudio: Mimora cannot record or play audio without it.\n'
    fi
    if [ -n "$need_gomp" ]; then
        printf '  - the GNU OpenMP library: the phrase generator (llama-server) cannot start without it.\n'
    fi
    if [ -n "$need_font" ]; then
        printf '  - the Symbola font: without it the record button shows no icon.\n'
    fi

    distro=""
    if [ -r /etc/os-release ]; then
        # ID and ID_LIKE only, in a subshell, so that no other variable of
        # os-release changes this script.
        distro=$(. /etc/os-release && printf '%s %s' "${ID:-}" "${ID_LIKE:-}")
    fi
    # One apt command for all packages. fonts-symbola is in Ubuntu's
    # universe repository, which a desktop installation has enabled.
    case " $distro " in
        *" debian "* | *" ubuntu "*)
            packages=""
            if [ -n "$need_portaudio" ]; then
                packages="libportaudio2"
            fi
            if [ -n "$need_gomp" ]; then
                packages="${packages:+$packages }libgomp1"
            fi
            if [ -n "$need_font" ]; then
                packages="${packages:+$packages }fonts-symbola"
            fi
            printf '  sudo apt install %s\n' "$packages"
            return 0
            ;;
    esac
    if [ -n "$need_portaudio" ]; then
        case " $distro " in
            *" fedora "* | *" rhel "*) printf '  sudo dnf install portaudio\n' ;;
            *" arch "*) printf '  sudo pacman -S portaudio\n' ;;
            *" opensuse"*) printf '  sudo zypper install libportaudio2\n' ;;
            *) printf '  Install the PortAudio library (libportaudio.so.2) with the package manager.\n' ;;
        esac
    fi
    # Arch is not named: libgomp is part of gcc-libs there, which every
    # installation has, so the check does not find it missing.
    if [ -n "$need_gomp" ]; then
        case " $distro " in
            *" fedora "* | *" rhel "*) printf '  sudo dnf install libgomp\n' ;;
            *" opensuse"*) printf '  sudo zypper install libgomp1\n' ;;
            *) printf '  Install the GNU OpenMP library (libgomp.so.1) with the package manager.\n' ;;
        esac
    fi
    # The package name of the font differs between distributions and was
    # checked only for Debian and Ubuntu, so no command is guessed here.
    if [ -n "$need_font" ]; then
        printf '  Install the Symbola font with the package manager.\n'
    fi
}

# The icon ships inside the package. Its path contains the Python version, so
# a copy goes to the icon theme directory, where the menu entry finds it by name.
install_icon() {
    tool_dir=$("$1" tool dir)
    icon_source=$(find "$tool_dir/mimora/lib" -path '*/site-packages/mimora/icons/mimora-256.png' 2>/dev/null | head -n 1)
    if [ -z "$icon_source" ]; then
        warn "The Mimora icon was not found. The menu entry has no icon."
        return 0
    fi
    mkdir -p "$icon_dir"
    cp "$icon_source" "$icon_dir/mimora.png"
}

create_desktop_entry() {
    uv=$1
    bin_dir=$2
    # A version without the mimora-gui script (1.1.0 and older) would give
    # a menu entry that starts nothing.
    if [ ! -x "$bin_dir/mimora-gui" ]; then
        warn "$bin_dir/mimora-gui was not found, so no menu entry was created."
        return 0
    fi
    step "Creating the application menu entry $desktop_file"
    install_icon "$uv"
    mkdir -p "$(dirname "$desktop_file")"
    # Exec is quoted, because the home directory can contain spaces.
    # StartupWMClass is WM_CLASS in mimora/window_icon.py: with another value
    # the running window is not matched to this entry and has no Mimora icon.
    cat >"$desktop_file" <<DESKTOP
[Desktop Entry]
Type=Application
Name=Mimora
Comment=Pronunciation trainer
Exec="$bin_dir/mimora-gui"
Icon=mimora
StartupWMClass=Mimora
Terminal=false
Categories=Education;
DESKTOP
    # Some desktops read the menu only from this cache. Without the tool the
    # entry appears after the next login.
    if command -v update-desktop-database >/dev/null 2>&1; then
        update-desktop-database "$(dirname "$desktop_file")" >/dev/null 2>&1 || true
    fi
}

data_dir() {
    if [ -n "${MIMORA_HOME:-}" ]; then
        printf '%s\n' "$MIMORA_HOME"
    elif [ "$os" = macos ]; then
        printf '%s\n' "$HOME/Library/Application Support/Mimora"
    else
        printf '%s\n' "${XDG_DATA_HOME:-$HOME/.local/share}/mimora"
    fi
}

install_mimora() {
    package=mimora
    if [ -n "${MIMORA_VERSION:-}" ]; then
        package="mimora==$MIMORA_VERSION"
    fi
    banner "$package"

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

    step "Installing $package (Python, PyTorch and the dependencies, this can take several minutes)"
    # --upgrade: a second run of this script is an upgrade.
    # --managed-python: always the Python build of uv. It carries tkinter; a
    #   system Python on Linux does not, without the python3-tk package.
    # --python 3.12: a uv tool environment ignores requires-python.
    # No --torch-backend: PyPI's torch is already a CUDA build on Linux, and
    #   macOS has no CUDA. The flag only moves the resolution to another
    #   index, and its result can change between runs. It is necessary on
    #   Windows only (install.ps1).
    # --with "preshed<3.0.14": preshed 3.0.14 breaks the spaCy pipeline load,
    #   and Mimora stops at "Initialization Failed" (explosion/spaCy#14058).
    #   The wheels up to 1.2.0rc1 do not exclude it, and the script installs
    #   them too. uv keeps the requirement, so "uv tool upgrade" keeps it.
    #   Remove it together with the bound in pyproject.toml.
    "$uv" tool install --upgrade --managed-python --python 3.12 --with "preshed<3.0.14" "$package" \
        || die "The Mimora installation failed. Read the messages above. If uv reports an unknown option, update uv ('uv self update') and run this command again."

    bin_dir=$("$uv" tool dir --bin)

    if [ "$os" = linux ]; then
        if [ -n "${MIMORA_NO_SHORTCUT:-}" ]; then
            step "MIMORA_NO_SHORTCUT is set: no menu entry was created"
        else
            create_desktop_entry "$uv" "$bin_dir"
        fi
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
    if [ "$os" = linux ] && [ -f "$desktop_file" ]; then
        printf '  Mimora is also in the application menu.\n'
    fi
    printf '  At the first start, Mimora asks to download its models (several GB).\n'
    # uv keeps an exact "==" requirement, so after MIMORA_VERSION
    # "uv tool upgrade mimora" does nothing.
    if [ -n "${MIMORA_VERSION:-}" ]; then
        printf '  Upgrade:   run this command again with MIMORA_VERSION=<new version> before "sh"\n'
    else
        printf '  Upgrade:   uv tool upgrade mimora   (or run this command again)\n'
    fi
    printf '  Uninstall: run this command again with "sh -s -- --uninstall"\n'
    if [ "$os" = linux ]; then
        print_native_hints
    fi
}

uninstall_mimora() {
    uv=$(find_uv)
    if [ -n "$uv" ]; then
        step "Removing the Mimora uv tool"
        "$uv" tool uninstall mimora || warn "uv could not remove Mimora. Possibly it was not installed with uv."
    else
        warn "uv was not found, so the Mimora uv tool was not removed."
    fi

    if [ "$os" = linux ]; then
        if [ -f "$desktop_file" ]; then
            step "Deleting $desktop_file"
            rm -f "$desktop_file"
        fi
        rm -f "$icon_dir/mimora.png"
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

    case "$(uname -s)" in
        Linux) os=linux ;;
        Darwin) os=macos ;;
        *) die "This script supports macOS and Linux. On Windows, use install.ps1." ;;
    esac

    share_dir="${XDG_DATA_HOME:-$HOME/.local/share}"
    desktop_file="$share_dir/applications/mimora.desktop"
    icon_dir="$share_dir/icons/hicolor/256x256/apps"

    if [ "$action" = uninstall ]; then
        uninstall_mimora
    else
        install_mimora
    fi
}

# All the work is in functions, and this is the only line that starts it.
# When "curl | sh" receives only part of the file, this line is missing and
# nothing runs.
main "$@"
