@echo off
rem Runs at the end of the Windows installer, in the installation directory.
rem It installs Mimora as a uv tool from PyPI: the same wheel that every other
rem way of installation gets, so the installer cannot behave differently.
rem Arguments: the Mimora version, then the model components to download now
rem (see fetch_models.py).
rem Exit codes: 0 success, 2 this version is not on PyPI, 1 any other failure.
rem CurPageChanged in mimora.iss shows a different message for each.
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

set "VERSION=%~1"
rem Without this check uv gets "mimora==", and its error message does not
rem name the cause.
if not defined VERSION (
    echo.
    echo No Mimora version was given. The installer is damaged.
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

rem A clear message when PyPI answers that this version does not exist. Only
rem a 404 stops the installation: without network access (curl prints 000),
rem without curl.exe, or with a package index mirror configured for uv, uv
rem itself reports what is wrong. No quotation marks in the command: for /f
rem removes some of them, and none of these parts contains a space.
set "PYPI_STATUS="
for /f %%c in ('%SystemRoot%\System32\curl.exe -s -o NUL -w %%{http_code} --max-time 30 https://pypi.org/pypi/mimora/%VERSION%/json 2^>NUL') do set "PYPI_STATUS=%%c"
if "%PYPI_STATUS%"=="404" (
    echo.
    echo Mimora %VERSION% is not available for download from PyPI.
    echo This installer cannot install it. Download the current installer from
    echo https://github.com/vikonix/Mimora/releases
    pause
    exit /b 2
)

rem --managed-python: always the Python build of uv, never a Python that is
rem   already on the computer.
rem --python 3.12: a uv tool environment ignores requires-python (see README).
rem --torch-backend auto: without it an NVIDIA computer gets a CPU-only torch.
rem --force: an upgrade replaces the previous installation.
"%APP_DIR%uv.exe" tool install --force --managed-python --python 3.12 --torch-backend auto "mimora==%VERSION%"
if errorlevel 1 (
    echo.
    echo Mimora installation failed. Read the messages above.
    pause
    exit /b 1
)

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
