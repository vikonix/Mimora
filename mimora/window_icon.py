# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""The application icon of the Mimora windows.

Two functions, and both are safe to call on every platform:

* :func:`set_app_user_model_id` gives the process its own identity on the
  Windows taskbar. Without it Windows groups the window with python.exe and
  shows the Python icon.
* :func:`apply_window_icon` puts the icon on a Tk root and on every window
  that the root opens later.
* :func:`sharpen_window_icon` is for each later window (a Toplevel) on
  Windows; see below.

A missing or unreadable icon must never stop the application. Each failure is
written to the log, and the window opens with the default Tk icon.

Why Windows needs a second step
-------------------------------
The process is not DPI-aware, so Windows tells Tk that the icon sizes are 16
and 32 pixels. On a display scaled to 150 % the title bar and the taskbar need
24 and 48 pixels, and Windows stretches the small images: the icon is blurred.
``_set_sharp_windows_icons`` loads the images at the real pixel size and gives
them to the window directly. It works on one window at a time, which is why a
Toplevel needs its own :func:`sharpen_window_icon` call.

The icon files are in ``mimora/icons/`` (see ``paths.shipped_icons_dir``).
"""

import logging
import sys
import tkinter as tk

from mimora import paths

# The taskbar identity. The shortcut in installer/mimora.iss carries the same
# value: if the two values are different, a pinned shortcut and the running
# window become two separate taskbar buttons.
APP_USER_MODEL_ID = "vikonix.Mimora"

ICON_ICO_NAME = "mimora.ico"
ICON_PNG_NAME = "mimora-256.png"

# Win32 constants (winuser.h, wingdi.h).
_WM_SETICON = 0x0080
_ICON_SMALL = 0
_ICON_BIG = 1
_IMAGE_ICON = 1
_LR_LOADFROMFILE = 0x0010
_SM_CXICON = 11
_SM_CXSMICON = 49
_HORZRES = 8
_DESKTOPHORZRES = 118


def set_app_user_model_id() -> None:
    """Give this process its own taskbar identity. Does nothing off Windows.

    Call it before the first window is created: Windows reads the identity
    when it makes the taskbar button.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            APP_USER_MODEL_ID)
    except (AttributeError, OSError):
        logging.warning("Could not set the taskbar identity.", exc_info=True)


def apply_window_icon(root: tk.Misc) -> None:
    """Set the Mimora icon on ``root`` and on its later child windows."""
    icons_dir = paths.shipped_icons_dir()
    try:
        if sys.platform == "win32":
            # An .ico file holds each size as a separate image, so the title
            # bar and the taskbar each get a sharp one. ``default=`` applies
            # the icon to every Toplevel too (the settings window).
            ico_path = icons_dir / ICON_ICO_NAME
            root.iconbitmap(default=str(ico_path))
            _sharpen_when_mapped(root, ico_path)
        else:
            image = tk.PhotoImage(master=root, file=str(icons_dir / ICON_PNG_NAME))
            root.iconphoto(True, image)
            # Tk keeps only the name of the image. Without this reference
            # Python deletes the image and the icon becomes empty.
            root._mimora_icon_image = image
    except (tk.TclError, OSError):
        logging.warning("Could not set the window icon.", exc_info=True)


def sharpen_window_icon(window: tk.Misc) -> None:
    """Give a Toplevel the sharp icons. Does nothing off Windows.

    The root does not need this call: :func:`apply_window_icon` does the same
    work for it.
    """
    if sys.platform != "win32":
        return
    _sharpen_when_mapped(window,
                         paths.shipped_icons_dir() / ICON_ICO_NAME)


def _sharpen_when_mapped(root: tk.Misc, ico_path) -> None:
    """Replace the icons of ``root`` with sharp ones when it first appears.

    ``root`` is a Tk root or a Toplevel. The outer window that owns the icons
    does not exist before the window is shown, so the work waits for the first
    <Map> event. Tk has already set its own (small) icons by then and does not
    set them again.
    """
    done = False

    def on_map(event):
        nonlocal done
        # <Map> arrives for each child widget too, and again after a minimize.
        if done or event.widget is not root:
            return
        done = True
        try:
            _set_sharp_windows_icons(root, ico_path)
        except (AttributeError, OSError, tk.TclError):
            logging.warning("Could not set the sharp window icons.",
                            exc_info=True)

    root.bind("<Map>", on_map, add="+")


def _set_sharp_windows_icons(root: tk.Misc, ico_path) -> None:
    """Give the window icons that have the real pixel size of the display."""
    import ctypes

    # Private library objects: the argument types set below must not change
    # the shared ctypes.windll.user32 that other modules use.
    user32 = ctypes.WinDLL("user32")
    gdi32 = ctypes.WinDLL("gdi32")
    handle = ctypes.c_void_p
    user32.GetParent.restype = handle
    user32.GetParent.argtypes = [handle]
    user32.GetDC.restype = handle
    user32.GetDC.argtypes = [handle]
    user32.ReleaseDC.argtypes = [handle, handle]
    gdi32.GetDeviceCaps.argtypes = [handle, ctypes.c_int]
    user32.LoadImageW.restype = handle
    user32.LoadImageW.argtypes = [handle, ctypes.c_wchar_p, ctypes.c_uint,
                                  ctypes.c_int, ctypes.c_int, ctypes.c_uint]
    user32.SendMessageW.argtypes = [handle, ctypes.c_uint, handle, handle]

    # The display scale. These two values are different only for a process
    # that is not DPI-aware: logical pixels and real pixels.
    screen = user32.GetDC(None)
    try:
        logical_width = gdi32.GetDeviceCaps(screen, _HORZRES)
        real_width = gdi32.GetDeviceCaps(screen, _DESKTOPHORZRES)
    finally:
        user32.ReleaseDC(None, screen)
    scale = real_width / logical_width if logical_width else 1.0

    # winfo_id() is the inner window of Tk. Its parent has the title bar.
    window = user32.GetParent(root.winfo_id())
    for kind, metric in ((_ICON_SMALL, _SM_CXSMICON), (_ICON_BIG, _SM_CXICON)):
        size = round(user32.GetSystemMetrics(metric) * scale)
        # The icon is not released: the window uses it until the process ends.
        icon = user32.LoadImageW(None, str(ico_path), _IMAGE_ICON, size, size,
                                 _LR_LOADFROMFILE)
        if icon:
            user32.SendMessageW(window, _WM_SETICON, kind, icon)
