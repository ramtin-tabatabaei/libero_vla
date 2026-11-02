"""Use antialiased system Tk when a Linux conda build lacks Xft support."""
import ctypes
import os
from pathlib import Path
import sys


def configure_tk_runtime():
    # Must happen before _tkinter is loaded. Already initialized applications
    # keep their existing runtime; this only affects this GUI process.
    if not sys.platform.startswith("linux") or "_tkinter" in sys.modules:
        return
    bundled = Path(sys.prefix) / "lib/libtk8.6.so"
    system = Path("/usr/lib/x86_64-linux-gnu")
    scripts = Path("/usr/share/tcltk")
    if not bundled.exists() or b"libXft" in bundled.read_bytes():
        return
    if not all(path.exists() for path in (
            system / "libtcl8.6.so", system / "libtk8.6.so",
            scripts / "tcl8.6/init.tcl", scripts / "tk8.6/tk.tcl")):
        return
    ctypes.CDLL(str(system / "libtcl8.6.so"), mode=ctypes.RTLD_GLOBAL)
    ctypes.CDLL(str(system / "libtk8.6.so"), mode=ctypes.RTLD_GLOBAL)
    os.environ["TCL_LIBRARY"] = str(scripts / "tcl8.6")
    os.environ["TK_LIBRARY"] = str(scripts / "tk8.6")
