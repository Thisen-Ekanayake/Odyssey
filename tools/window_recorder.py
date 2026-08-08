"""
Background recorder for every GUI window a script opens (cv2 ``imshow``
windows, Open3D viewers -- anything the running process itself puts on
screen).

Importing this module is the whole API:

    from tools import window_recorder  # noqa: E402,F401

That's it -- no context manager, no start()/stop() calls. On import it:

  1. Starts a background thread that watches for X11 windows owned by this
     process (matched by ``_NET_WM_PID``, so it needs no window-title
     conventions and picks up windows opened at any point in the run).
  2. For each window found, saves JPEG frames into
     ``recordings/<run-timestamp>/<N>/`` (``N`` = 1, 2, 3... in discovery
     order -- order carries no meaning, it's just "which window").
  3. Registers an ``atexit`` hook that, once the process is exiting (i.e.
     every window it owned is already closed), renders each ``<N>/``
     folder's frames into a 1920x1080 H.264 ``<N>.mp4`` next to it
     (letterboxed -- windows are captured at their own native size, not
     forced to 1080p on screen), then deletes that ``<N>/`` folder -- a
     run can leave thousands of frames per window, and once the mp4 exists
     they're pure disk cost. A folder is only deleted after its own render
     succeeds, so a failed render's frames stick around to retry/debug.

Safe to import into headless scripts too: if no window is ever opened, or if
``python-xlib``/``ffmpeg``/ImageMagick's ``import`` isn't available, this
degrades to a no-op rather than raising -- a recording failure must never
take down an actual flight script.
"""
from __future__ import annotations

import atexit
import itertools
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from slam.config import REPO_ROOT  # noqa: E402

# Capturing a window means reading its rendered pixels back off the GPU --
# the same GPU AirSim's UE4 renderer uses -- so this rate is a potential
# contention point with the sim itself, not just extra CPU/disk work. Watch
# sim smoothness on multi-window scripts if you raise it further.
FPS = 30
POLL_INTERVAL = 0.5          # seconds between scans for newly opened windows
JPEG_QUALITY = 90
TARGET_RESOLUTION = (1920, 1080)
OUTPUT_ROOT = REPO_ROOT / "recordings"

_stop_event = threading.Event()
_threads: list[threading.Thread] = []
_lock = threading.Lock()
_next_number = itertools.count(1)
_run_dir: Path | None = None
_started = False
_display = None  # set by _discovery_loop once Xlib is confirmed usable


def _log(msg: str) -> None:
    print(f"[window_recorder] {msg}", file=sys.stderr)


def _init() -> None:
    global _started, _run_dir
    if _started:
        return
    _started = True

    try:
        import Xlib.display  # noqa: F401
    except ImportError:
        _log("python-xlib not installed (pip install python-xlib into "
             "airsim_venv) -- recording disabled")
        return

    missing = [tool for tool in ("import", "ffmpeg") if shutil.which(tool) is None]
    if missing:
        _log(f"required tool(s) not on PATH: {', '.join(missing)} -- recording disabled")
        return

    _run_dir = OUTPUT_ROOT / time.strftime("%Y%m%d_%H%M%S")
    t = threading.Thread(target=_discovery_loop, daemon=True, name="window-recorder-discovery")
    t.start()
    atexit.register(_shutdown)


def _walk(win, x_error):
    yield win
    try:
        children = win.query_tree().children
    except x_error:
        return
    for child in children:
        yield from _walk(child, x_error)


def _discovery_loop() -> None:
    from Xlib import X, display as xdisplay
    from Xlib.error import XError

    global _display
    try:
        _display = xdisplay.Display()
    except Exception as exc:  # pragma: no cover - environment dependent
        _log(f"cannot open X display, recording disabled ({exc})")
        return

    root = _display.screen().root
    pid_atom = _display.intern_atom("_NET_WM_PID")
    my_pid = os.getpid()
    seen: set[int] = set()

    while not _stop_event.is_set():
        try:
            for win in _walk(root, XError):
                if win.id in seen:
                    continue
                try:
                    attrs = win.get_attributes()
                    if attrs.map_state != X.IsViewable:
                        continue
                    prop = win.get_full_property(pid_atom, X.AnyPropertyType)
                    if not prop or not prop.value or prop.value[0] != my_pid:
                        continue
                    geom = win.get_geometry()
                    if geom.width <= 1 or geom.height <= 1:
                        continue
                except XError:
                    continue
                seen.add(win.id)
                _register_window(win)
        except Exception as exc:  # pragma: no cover - defensive, must not crash caller
            _log(f"discovery error: {exc}")
        _stop_event.wait(POLL_INTERVAL)


def _register_window(win) -> None:
    with _lock:
        number = next(_next_number)
    folder = _run_dir / str(number)
    folder.mkdir(parents=True, exist_ok=True)

    try:
        title = (win.get_wm_name() or "").strip()
    except Exception:
        title = ""
    _log(f"tracking window {number} ({title or 'untitled'}) -> "
         f"{folder.relative_to(REPO_ROOT)}")

    win_id_hex = f"0x{win.id:x}"
    t = threading.Thread(
        target=_capture_loop, args=(win.id, win_id_hex, folder),
        daemon=True, name=f"window-recorder-capture-{number}",
    )
    _threads.append(t)
    t.start()


def _window_exists(win_id: int) -> bool:
    from Xlib.error import XError
    try:
        _display.create_resource_object("window", win_id).get_geometry()
        return True
    except XError:
        return False
    except Exception:  # pragma: no cover - defensive
        return False


def _capture_frame(win_id_hex: str, frame_path: Path) -> bool:
    try:
        result = subprocess.run(
            ["import", "-silent", "-window", win_id_hex,
             "-quality", str(JPEG_QUALITY), str(frame_path)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5,
        )
        return result.returncode == 0 and frame_path.exists()
    except (subprocess.TimeoutExpired, OSError):
        return False


def _capture_loop(win_id: int, win_id_hex: str, folder: Path) -> None:
    interval = 1.0 / FPS
    frame_idx = 0
    consecutive_failures = 0

    while not _stop_event.is_set():
        tick_start = time.monotonic()
        frame_idx += 1
        ok = _capture_frame(win_id_hex, folder / f"frame_{frame_idx:06d}.jpg")
        if ok:
            consecutive_failures = 0
        else:
            frame_idx -= 1  # don't leave a numbering gap for a failed grab
            consecutive_failures += 1
            if consecutive_failures >= 3 and not _window_exists(win_id):
                break
        elapsed = time.monotonic() - tick_start
        remaining = interval - elapsed
        if remaining > 0:
            _stop_event.wait(remaining)


def _render_all() -> None:
    if _run_dir is None or not _run_dir.exists():
        return
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return

    w, h = TARGET_RESOLUTION
    vf = (f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
          f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black")

    for folder in sorted((p for p in _run_dir.iterdir() if p.is_dir()), key=lambda p: p.name):
        frames = sorted(folder.glob("frame_*.jpg"))
        if not frames:
            continue
        out_path = _run_dir / f"{folder.name}.mp4"
        try:
            result = subprocess.run(
                [ffmpeg, "-y", "-start_number", "1", "-framerate", str(FPS),
                 "-i", str(folder / "frame_%06d.jpg"), "-r", str(FPS),
                 "-vf", vf, "-preset", "veryfast",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_path)],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=600,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            # One folder's encode must never stop the rest of them -- a run
            # can have thousands of frames per window, so give each attempt
            # its own failure boundary instead of one shared try/except.
            _log(f"ffmpeg failed for {folder.relative_to(REPO_ROOT)}: {exc}")
            continue
        if result.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0:
            _log(f"rendered {out_path.relative_to(REPO_ROOT)} ({len(frames)} frames)")
            try:
                shutil.rmtree(folder)
            except OSError as exc:
                _log(f"rendered but could not delete frames for "
                     f"{folder.relative_to(REPO_ROOT)}: {exc}")
        else:
            # Keep the frames on a failed render -- they're the only way to
            # retry or debug it, and deleting a folder we couldn't turn into
            # a video would just lose the recording outright. Do clear away
            # any partial/empty output ffmpeg left, so the folder isn't sat
            # next to a same-named .mp4 that looks finished but isn't.
            if out_path.exists():
                out_path.unlink(missing_ok=True)
            err = result.stderr.decode(errors="replace").strip().splitlines()
            _log(f"ffmpeg failed for {folder.relative_to(REPO_ROOT)}: "
                 f"{err[-1] if err else 'unknown error'}")


def _shutdown() -> None:
    _stop_event.set()
    for t in list(_threads):
        t.join(timeout=2.0)
    try:
        _render_all()
    except Exception as exc:  # pragma: no cover - defensive
        _log(f"render error: {exc}")


_init()
