#!/bin/sh
# Since 1.3.0, Mimora has one install script for each platform:
#   macOS:   install-macos.sh
#   Linux:   install-linux.sh
#   Windows: install.ps1
#
# This file stays only so that the old command, which downloads
# releases/latest/download/install.sh, does not fail without a message.
# It installs nothing and downloads nothing: it prints the command for this
# computer and stops. Do not make it run the platform script. A copy of this
# file from an old release would then run the script of the newest release,
# and that script does not know what older versions need.
#
# POSIX sh, not bash: "| sh" is dash on Debian and Ubuntu.

set -eu

base=https://github.com/vikonix/Mimora/releases/latest/download

# "sh -s -- --uninstall" gives the option to this script. The printed
# command keeps it, so that an uninstall does not become an install.
suffix=''
for arg in "$@"; do
    if [ "$arg" = --uninstall ]; then
        suffix=' -s -- --uninstall'
    fi
done

printf '\nMimora now has a separate install script for each platform.\n'
printf 'Nothing was installed. Copy this command into the terminal and run it:\n\n'

case "$(uname -s)" in
    Darwin)
        # macOS has curl but not wget.
        printf '  curl -LsSf %s/install-macos.sh | sh%s\n' "$base" "$suffix"
        ;;
    Linux)
        # Ubuntu Desktop has wget but not curl. Show the tool that is there,
        # because the user just used one of them to get this file.
        if command -v wget >/dev/null 2>&1; then
            printf '  wget -qO- %s/install-linux.sh | sh%s\n' "$base" "$suffix"
        else
            printf '  curl -LsSf %s/install-linux.sh | sh%s\n' "$base" "$suffix"
        fi
        ;;
    *)
        # Git Bash, MSYS2 or Cygwin on Windows, or a system that Mimora has
        # no script for.
        printf '  macOS:   curl -LsSf %s/install-macos.sh | sh%s\n' "$base" "$suffix"
        printf '  Linux:   wget -qO- %s/install-linux.sh | sh%s\n' "$base" "$suffix"
        printf '  Windows: see https://github.com/vikonix/Mimora#installation\n'
        ;;
esac

# The newest script must not install an old version (see the top of this
# file). The release of each version has its own scripts.
if [ -n "${MIMORA_VERSION:-}" ]; then
    printf '\nFor version %s, use the script from the release of that version:\n' "$MIMORA_VERSION"
    printf '  https://github.com/vikonix/Mimora/releases/tag/v%s\n' "$MIMORA_VERSION"
fi

printf '\n'
# Not 0: nothing was installed, and a calling script must see that.
exit 1
