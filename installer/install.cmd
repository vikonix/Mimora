@echo off
rem Runs at the end of the Windows installer, in the installation directory.
rem It installs Mimora as a uv tool from PyPI: the same wheel that every other
rem way of installation gets, so the installer cannot behave differently.
rem Arguments: what to install, then the model components to download now
rem (see fetch_models.py). What to install is "mimora==<version>" from PyPI,
rem or the path of the wheel that a test build of the installer carries
rem (LocalWheel in mimora.iss). It comes in quotation marks, because cmd
rem splits an argument at "=".
rem Whether PyPI has this version is checked before, by PrepareToInstall in
rem mimora.iss, so that Setup can refuse before it installs anything.
setlocal
set "APP_DIR=%~dp0"

rem Everything uv creates stays below the installation directory. The
rem uninstaller then removes the application with one directory delete, and
rem nothing is added to the user's own uv tools.
set "UV_TOOL_DIR=%APP_DIR%tools"
set "UV_TOOL_BIN_DIR=%APP_DIR%bin"
set "UV_PYTHON_INSTALL_DIR=%APP_DIR%python"
rem The cache too: the default one is in the user profile and would stay
rem there, with several gigabytes of wheels, after Mimora is uninstalled.
set "UV_CACHE_DIR=%APP_DIR%cache"

set "SPEC=%~1"
rem Without this check uv gets an empty argument, and its error message does
rem not name the cause.
if not defined SPEC (
    echo.
    echo No Mimora package was given. The installer is damaged.
    pause
    exit /b 1
)
shift

rem %* does not change after shift, so the components are collected here.
set "COMPONENTS="
:collect_components
if "%~1"=="" goto components_collected
set "COMPONENTS=%COMPONENTS% %~1"
shift
goto collect_components
:components_collected

rem --managed-python: always the Python build of uv, never a Python that is
rem   already on the computer.
rem --python 3.12: a uv tool environment ignores requires-python (see README).
rem --torch-backend auto: without it an NVIDIA computer gets a CPU-only torch.
rem --force: an upgrade replaces the previous installation.
"%APP_DIR%uv.exe" tool install --force --managed-python --python 3.12 --torch-backend auto "%SPEC%"
if errorlevel 1 (
    echo.
    echo Mimora installation failed. Read the messages above.
    pause
    exit /b 1
)

rem uv warns in yellow that the bin directory is not on PATH and suggests
rem "uv tool update-shell". The Start menu shortcuts use the full path, so
rem PATH is not needed. Without this line the warning looks like a failed
rem installation.
echo.
echo The PATH warning above does not apply: the Start menu shortcuts start Mimora.

rem No component selected: the first-run window of Mimora downloads later.
if not defined COMPONENTS exit /b 0

rem The interpreter of the tool environment, so the fetchers write into the
rem same user data directory that the application reads.
"%UV_TOOL_DIR%\mimora\Scripts\python.exe" "%APP_DIR%fetch_models.py" %COMPONENTS%

rem A failed download does not fail the installation: the first-run window of
rem the application offers every missing component again.
if errorlevel 1 (
    echo.
    echo Some downloads failed. Mimora will offer them again at the first start.
    pause
)
exit /b 0
