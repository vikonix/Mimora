# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev

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
    be handled by the caller beforehand (see main.py _shutdown_runtime);
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


def spawn_replacement():
    """Spawn a detached replacement process running the same command line.

    subprocess.Popen is used instead of os.execv: on Windows execv detaches
    the console under some launchers and mangles arguments with spaces.

    The replacement must not share the dying parent's console/stdio: when
    launched from an IDE, the IDE closes those pipes as soon as the parent
    exits and the child's first print would crash with [Errno 22] (the same
    failure mode the main.py module-top comment describes for os.execv). So
    stdio is pointed at DEVNULL - the app logs to logs/main.log anyway -
    and on Windows the child is detached from the console and, when the
    launcher allows it, broken out of the IDE's job object so "stop" in the
    IDE cannot kill the restarted app.

    A relaunch failure is logged and swallowed: the caller must still exit
    cleanly, which is exactly what a failed restart degrades to (the user
    relaunches by hand).
    """
    try:
        command = [sys.executable] + sys.argv
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
            except OSError:
                logging.info("Job breakaway denied; relaunching attached "
                             "to the current job.")
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
