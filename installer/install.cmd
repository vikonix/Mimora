@echo off
rem Runs at the end of the Windows installer, in the installation directory.
rem It installs Mimora as a uv tool from the wheel that the installer carries.
rem Arguments: the model components to download now (see fetch_models.py).
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

set "WHEEL="
for %%f in ("%APP_DIR%*.whl") do set "WHEEL=%%f"
rem Without this check uv gets an empty argument, and its error message does
rem not name the cause.
if not defined WHEEL (
    echo.
    echo No Mimora wheel found in "%APP_DIR%". The installer is damaged.
    pause
    exit /b 1
)

rem --python 3.12: a uv tool environment ignores requires-python (see README).
rem --torch-backend auto: without it an NVIDIA computer gets a CPU-only torch.
rem --force: an upgrade replaces the previous installation.
"%APP_DIR%uv.exe" tool install --force --python 3.12 --torch-backend auto "%WHEEL%"
if errorlevel 1 (
    echo.
    echo Mimora installation failed. Read the messages above.
    pause
    exit /b 1
)

rem No component selected: the first-run window of Mimora downloads later.
if "%~1"=="" exit /b 0

rem The interpreter of the tool environment, so the fetchers write into the
rem same user data directory that the application reads.
"%UV_TOOL_DIR%\mimora\Scripts\python.exe" "%APP_DIR%fetch_models.py" %*

rem A failed download does not fail the installation: the first-run window of
rem the application offers every missing component again.
if errorlevel 1 (
    echo.
    echo Some downloads failed. Mimora will offer them again at the first start.
    pause
)
exit /b 0
