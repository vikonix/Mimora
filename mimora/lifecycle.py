# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""Process-exit helpers: hard exit and detached self-relaunch.

Extracted from the main controller because both are process-level and
Tk-free. The controller keeps the orchestration (quit_app / restart_app and
_shutdown_runtime, which release the app's own resources first) and calls
these to end or replace the process.
"""

import logging
import os
import subprocess
import sys

# For APPEND_LOG_FLAG only, which spawn_replacement() passes to the process it
# starts. bootstrap is stdlib-only itself, so this keeps the module as free of
# heavy imports as it was.
from mimora import bootstrap


def hard_exit():
    """End the process immediately, bypassing interpreter finalization.

    Hard-exit on the main thread instead of via root.destroy() + the
    interpreter's normal finalization. With CUDA + PyTorch loaded, the
    native CUDA context is torn down while still live and crashes inside
    the C extensions, surfacing as Windows exit code 0xC0000409
    (STATUS_STACK_BUFFER_OVERRUN) with no Python traceback.

    os._exit is NOT enough on Windows: it maps to ExitProcess, which still
    runs DLL_PROCESS_DETACH for every loaded DLL - and the CUDA runtime's
    detach is exactly what crashes. TerminateProcess ends the process at
    the OS level without running any DLL detach handlers, so that crash
    never runs. The external resources that actually need releasing must
    be handled by the caller beforehand (see app.py _shutdown_runtime);
    logs are flushed here first. os._exit is the fallback for non-Windows.
    """
    logging.shutdown()
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        kernel32 = ctypes.windll.kernel32
        # Declare the signatures: GetCurrentProcess returns a HANDLE (a
        # 64-bit pointer). Without this, ctypes defaults the result to a
        # 32-bit c_int and TRUNCATES the pseudo-handle, so TerminateProcess
        # gets a bad handle, silently fails (returns FALSE without killing
        # anything), and we fall through to os._exit - which crashes in the
        # CUDA DLL detach. With the correct types the pseudo-handle (-1) is
        # passed intact and the process ends at once with exit code 0.
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel32.TerminateProcess(kernel32.GetCurrentProcess(), 0)
    os._exit(0)


def relaunch_command() -> list:
    """The command that starts this application again, the way it was started.

    Three launch forms reach the same code, and they need three different
    commands. Prepending sys.executable to sys.argv - which is what this used
    to do unconditionally - is right for exactly one of them:

    * ``python main.py`` (and any direct script path): sys.argv[0] is a .py
      file, so the interpreter goes in front, as before.
    * ``python -m mimora``: sys.argv[0] is the full path to __main__.py, and
      running that file directly is NOT equivalent - executing a file puts its
      own directory on sys.path instead of the current one, so ``import
      mimora`` would fail from a source checkout. The launch is reconstructed
      as ``-m`` against the package the main module came from.
    * the ``mimora`` console script: sys.argv[0] is the script itself
      (``Scripts\\mimora.exe`` on Windows, a shebang file on POSIX), which
      knows how to start the interpreter on its own. Putting sys.executable in
      front would produce ``python.exe mimora.exe``, which does not run at all.

    The console-script case is the one that fails silently until a package
    exists, which is why the test is written the other way round: the
    interpreter is prepended only for something that is recognisably a Python
    source file, and anything else is assumed to be self-executing.
    """
    # __spec__ says the process was started with -m, but only when it NAMES a
    # module: spec.name is "mimora.__main__" for a package run that way, and
    # the package itself is what -m needs back.
    #
    # The name test is load-bearing, not defensive. A Windows console script is
    # a launcher with a zip archive appended, whose __main__.py is imported
    # through the normal machinery - so __spec__ exists and is named literally
    # "__main__". Reading that as a module yields `python.exe -m __main__`, a
    # command that does not run, and the app closes instead of restarting.
    # POSIX hides this: there a console script is started by path and __spec__
    # really is None.
    spec = getattr(sys.modules.get("__main__"), "__spec__", None)
    if spec is not None and spec.name and spec.name != "__main__":
        module = spec.parent if spec.name.endswith(".__main__") else spec.name
        if module:
            return [sys.executable, "-m", module] + sys.argv[1:]

    suffix = os.path.splitext(sys.argv[0])[1].lower()
    if suffix in (".py", ".pyw"):
        return [sys.executable] + sys.argv

    return list(sys.argv)


def spawn_replacement():
    """Spawn a detached replacement process running the same command line.

    subprocess.Popen is used instead of os.execv: on Windows execv detaches
    the console under some launchers and mangles arguments with spaces.

    The replacement must not share the dying parent's console/stdio: when
    launched from an IDE, the IDE closes those pipes as soon as the parent
    exits and the child's first print would crash with [Errno 22] (the same
    failure mode the app.py module-top comment describes for os.execv). So
    stdio is pointed at DEVNULL - the app logs to logs/main.log anyway -
    and on Windows the child is detached from the console and, when the
    launcher allows it, broken out of the IDE's job object so "stop" in the
    IDE cannot kill the restarted app.

    A relaunch failure is logged and swallowed: the caller must still exit
    cleanly, which is exactly what a failed restart degrades to (the user
    relaunches by hand).
    """
    try:
        command = relaunch_command()
        # The replacement continues this session's log instead of truncating
        # it - without the flag the child opens main.log with mode="w" and
        # discards the first-run download or the setting change that led here,
        # which is the reason anyone opens the file afterwards.
        #
        # Added here rather than in relaunch_command(), which answers "how was
        # this process started" and must keep answering only that. Guarded
        # because that function rebuilds from sys.argv, which may already carry
        # the flag from an earlier restart in the same session.
        if bootstrap.APPEND_LOG_FLAG not in command:
            command.append(bootstrap.APPEND_LOG_FLAG)
        logging.info(f"Relaunching: {command}")
        popen_kwargs = {
            "cwd": os.getcwd(),
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
        }
        if sys.platform == "win32":
            flags = (subprocess.DETACHED_PROCESS
                     | subprocess.CREATE_NEW_PROCESS_GROUP)
            try:
                # Escape the launcher's job object (IDEs kill the whole
                # tree on stop). Denied by some jobs - retry without.
                subprocess.Popen(
                    command,
                    creationflags=flags | subprocess.CREATE_BREAKAWAY_FROM_JOB,
                    **popen_kwargs)
            except OSError as exc:
                # The reason is reported rather than assumed. Denied breakaway
                # is the expected failure here, but every other OSError lands
                # in this clause too (a command that is not there, for one),
                # and a line that names a cause it did not check sends the next
                # reader after the wrong thing. The retry is worth making
                # either way: if breakaway was not the problem, the second
                # attempt fails as well and the outer handler says so.
                logging.info("Relaunch with job breakaway failed (%s); "
                             "retrying attached to the current job.", exc)
                subprocess.Popen(command, creationflags=flags,
                                 **popen_kwargs)
        else:
            # POSIX: a new session detaches from the controlling terminal.
            subprocess.Popen(command, start_new_session=True,
                             **popen_kwargs)
    except OSError:
        # The old process must still exit cleanly - the user can relaunch
        # by hand, which is exactly what a failed restart degrades to.
        logging.exception("Relaunch failed; exiting without a new process:")
