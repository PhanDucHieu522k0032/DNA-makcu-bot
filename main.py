import argparse
import asyncio
import os
import time
import glob
import shutil
import json
import queue
import re
import threading
import statistics
import types

import cv2
import numpy as np
import NDIlib as ndi
import math

# makcu async controller
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAKCUCOMMANDS_DIR = os.path.join(SCRIPT_DIR, 'MissionTypes', 'Defence', 'Wedge65', 'MakcuCommands')
if MAKCUCOMMANDS_DIR not in sys.path:
    sys.path.insert(0, MAKCUCOMMANDS_DIR)

# Makcu library: not a PyPI dependency in this repo.
# Set env var `MAKCU_PY_LIB` to the folder containing the `makcu` module.
try:
    from makcu import create_async_controller, MouseButton
except Exception:
    _makcu_lib = os.environ.get('MAKCU_PY_LIB', r'C:\okDNA\makcu-py-lib-main')
    if _makcu_lib and os.path.isdir(_makcu_lib) and _makcu_lib not in sys.path:
        sys.path.insert(0, _makcu_lib)
    try:
        from makcu import create_async_controller, MouseButton
    except Exception as e:
        raise SystemExit(
            "[FATAL] Could not import `makcu`. Set env var MAKCU_PY_LIB to your makcu-py-lib path. "
            f"Import error: {e}"
        )
import ctypes
import makcu_helper as mh
from telemetry import Telemetry
from dataset_logger import DatasetLogger
import defence_wedge_65_makcu


class LatestNDIFrame:
    """Producer thread that continuously captures NDI frames and keeps only the latest.

    This prevents backlog/lag in the consumer loop (preview stays responsive; logic uses freshest frame).
    """

    def __init__(self, ndi_recv, *, timeout_ms: int = 100):
        self._ndi_recv = ndi_recv
        self._timeout_ms = int(timeout_ms)
        self._lock = threading.Lock()
        self._latest = None  # (frame_bgr, xres, yres, ts)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="ndi-producer", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            self._thread.join(timeout=1.0)
        except Exception:
            pass

    def get_latest(self):
        with self._lock:
            return self._latest

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                t, v, _, _ = ndi.recv_capture_v2(self._ndi_recv, self._timeout_ms)
                if t != ndi.FRAME_TYPE_VIDEO:
                    time.sleep(0.001)
                    continue

                frame_data = np.copy(v.data)
                xres = int(v.xres)
                yres = int(v.yres)
                ndi.recv_free_video_v2(self._ndi_recv, v)

                try:
                    frame_data = frame_data.reshape((yres, xres, 4))
                    frame_bgr = np.ascontiguousarray(frame_data[:, :, :3])
                except Exception:
                    continue

                ts = time.time()
                with self._lock:
                    self._latest = (frame_bgr, xres, yres, ts)
            except Exception:
                time.sleep(0.005)
                continue


def _report_cv_cuda():
    try:
        if hasattr(cv2, 'cuda'):
            n = int(cv2.cuda.getCudaEnabledDeviceCount())
            print(f"[INIT] OpenCV CUDA devices: {n}")
            if n <= 0:
                print("[INIT] OpenCV was built without CUDA support (GPU not used for cv2 operations).")
        else:
            print("[INIT] This OpenCV build has no cv2.cuda module (GPU not used for cv2 operations).")
    except Exception as e:
        print(f"[INIT] CUDA capability check failed: {e}")


def _configure_cv_opencl(enable: bool) -> None:
    try:
        if not hasattr(cv2, 'ocl'):
            print('[INIT] OpenCV has no cv2.ocl module (OpenCL disabled).')
            return
        cv2.ocl.setUseOpenCL(bool(enable))
        if bool(enable):
            have = bool(cv2.ocl.haveOpenCL())
            use = bool(cv2.ocl.useOpenCL())
            print(f"[INIT] OpenCL available={have} enabled={use} (UMat acceleration)")
        else:
            print('[INIT] OpenCL disabled')
    except Exception as e:
        print(f"[INIT] OpenCL config failed: {e}")


class DummyMakcu:
    """Fallback when hardware is unavailable. Provides same async API."""
    async def move_abs(self, coords):
        print(f"[DUMMY MAKCU] move_abs {coords}")

    async def click(self, button):
        print(f"[DUMMY MAKCU] click {button}")


def _makcu_transport_send(makcu, cmd: str) -> bool:
    """Send a raw Makcu serial command string (e.g. 'km.move(10,0)')."""
    try:
        transport = getattr(makcu, 'transport', None)
        if transport and transport.is_connected():
            transport.send_command(str(cmd), expect_response=False)
            return True
    except Exception:
        return False
    return False


_RE_KM_MOVE = re.compile(
    r"^\s*(?:km\.)?move\s*(?:\(\s*([-+]?\d+)\s*,\s*([-+]?\d+)\s*\)|\s+([-+]?\d+)\s+([-+]?\d+))\s*$",
    re.IGNORECASE,
)


def _parse_km_move(line: str):
    """Parse 'km.move(10,0)' or 'km.move 10 0' (also accepts 'move ...')."""
    if not line:
        return None
    s = line.strip()
    s = s.replace('km.move', 'move').replace('KM.MOVE', 'move')
    m = _RE_KM_MOVE.match(s)
    if not m:
        return None
    if m.group(1) is not None and m.group(2) is not None:
        return int(m.group(1)), int(m.group(2))
    if m.group(3) is not None and m.group(4) is not None:
        return int(m.group(3)), int(m.group(4))
    return None


def _parse_ndi_point(s: str):
    """Parse an NDI point string like '123,456' or '123 456'. Returns (x,y) or None."""
    if not s:
        return None
    t = str(s).strip().replace('(', '').replace(')', '')
    t = t.replace(',', ' ')
    parts = [p for p in t.split() if p]
    if len(parts) != 2:
        return None
    try:
        return int(float(parts[0])), int(float(parts[1]))
    except Exception:
        return None


def _km_counts_to_ndi_delta(dx_cnt: int, dy_cnt: int, *, km_calibration=None):
    """Approximate NDI pixel delta produced by a km.move(dx_cnt,dy_cnt).

    Inverse of `makcu_helper.map_ndidelta_to_km` (overlay/logging only).
    """
    try:
        dx_cnt = int(dx_cnt)
        dy_cnt = int(dy_cnt)
    except Exception:
        return 0, 0

    if km_calibration:
        try:
            px_x = float(km_calibration.get("px_per_count_x", 0.0))
            px_y = float(km_calibration.get("px_per_count_y", 0.0))
            sx = 1 if int(km_calibration.get("sign_x", 1)) >= 0 else -1
            sy = 1 if int(km_calibration.get("sign_y", 1)) >= 0 else -1
            if px_x > 0 and px_y > 0:
                # Invert the sign application done in map_ndidelta_to_km.
                dx_ndi = int(round(float(dx_cnt) * px_x * (1 if sx >= 0 else -1)))
                dy_ndi = int(round(float(dy_cnt) * px_y * (1 if sy >= 0 else -1)))
                return dx_ndi, dy_ndi
        except Exception:
            pass

    # Fallback: treat counts roughly as NDI pixels (often close when px_per_count ~= 1).
    return int(dx_cnt), int(dy_cnt)


def get_cursor_pos():
    class POINT(ctypes.Structure):
        _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]
    pt = POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
    return pt.x, pt.y

# --- CONFIG ---
REAL_MONITOR_WIDTH = 1920
REAL_MONITOR_HEIGHT = 1080
TEMPLATE_PATH = os.path.join(SCRIPT_DIR, 'MissionTypes', 'Defence', 'Wedge65', 'samples')
PREVIEW_WIDTH = 960
MATCH_THRESHOLD = 0.90
# manual adjustment if mapping needs to be nudged (pixels)
CLICK_OFFSET_X = 0
CLICK_OFFSET_Y = 0


def _match_template(gray_small, templ_small):
    res = cv2.matchTemplate(gray_small, templ_small, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    return float(max_val), (int(max_loc[0]), int(max_loc[1]))


def _match_scaled(gray_for_match, scaled_templates, scaled_templates_u, name: str, *, use_opencl: bool):
    """Return (score, loc) for a scaled template name on the given gray image.

    gray_for_match can be a numpy array or cv2.UMat.
    """
    if scaled_templates is None or name not in scaled_templates:
        return 0.0, (0, 0)

    try:
        if use_opencl and isinstance(gray_for_match, cv2.UMat) and scaled_templates_u.get(name) is not None:
            r_u = cv2.matchTemplate(gray_for_match, scaled_templates_u[name], cv2.TM_CCOEFF_NORMED)
            r = r_u.get()
            _, s, _, loc = cv2.minMaxLoc(r)
            return float(s), (int(loc[0]), int(loc[1]))
        g = gray_for_match.get() if isinstance(gray_for_match, cv2.UMat) else gray_for_match
        return _match_template(g, scaled_templates[name])
    except Exception:
        return 0.0, (0, 0)


def _match_scaled_tracked(
    gray_for_match,
    scaled_templates,
    scaled_templates_u,
    name: str,
    *,
    use_opencl: bool,
    last_loc,
    search_radius: int,
):
    """Template matching with local search around last_loc (preview coords).

    Speeds up "next target" acquisition by avoiding full-frame matching every loop.
    Falls back to full-frame if last_loc is None or ROI becomes invalid.
    """
    if scaled_templates is None or name not in scaled_templates:
        return 0.0, (0, 0)

    templ = scaled_templates[name]
    th, tw = templ.shape[:2]

    # Full-frame fallback
    if last_loc is None:
        return _match_scaled(
            gray_for_match,
            scaled_templates,
            scaled_templates_u,
            name,
            use_opencl=bool(use_opencl),
        )

    try:
        if isinstance(gray_for_match, cv2.UMat):
            g_h, g_w = gray_for_match.get().shape[:2]
        else:
            g_h, g_w = gray_for_match.shape[:2]

        x0 = int(max(0, min(g_w - 1, int(last_loc[0]) - int(search_radius))))
        y0 = int(max(0, min(g_h - 1, int(last_loc[1]) - int(search_radius))))
        x1 = int(max(0, min(g_w, int(last_loc[0]) + int(search_radius) + int(tw))))
        y1 = int(max(0, min(g_h, int(last_loc[1]) + int(search_radius) + int(th))))

        # ROI too small -> fallback full
        if (x1 - x0) < max(3, tw) or (y1 - y0) < max(3, th):
            return _match_scaled(
                gray_for_match,
                scaled_templates,
                scaled_templates_u,
                name,
                use_opencl=bool(use_opencl),
            )

        roi = gray_for_match[y0:y1, x0:x1]
        if use_opencl and isinstance(gray_for_match, cv2.UMat) and scaled_templates_u.get(name) is not None:
            r_u = cv2.matchTemplate(roi, scaled_templates_u[name], cv2.TM_CCOEFF_NORMED)
            r = r_u.get()
            _, s, _, loc = cv2.minMaxLoc(r)
            return float(s), (int(loc[0] + x0), int(loc[1] + y0))
        g = roi.get() if isinstance(roi, cv2.UMat) else roi
        res = cv2.matchTemplate(g, templ, cv2.TM_CCOEFF_NORMED)
        _, s, _, loc = cv2.minMaxLoc(res)
        return float(s), (int(loc[0] + x0), int(loc[1] + y0))
    except Exception:
        return _match_scaled(
            gray_for_match,
            scaled_templates,
            scaled_templates_u,
            name,
            use_opencl=bool(use_opencl),
        )


def _clip_int(v: int, lo: int, hi: int) -> int:
    return int(max(int(lo), min(int(hi), int(v))))


def _template_center_ndi(loc, *, tw: int, th: int, scale: float, offset_x: int = 0, offset_y: int = 0):
    """Convert a preview-space template top-left `loc` into an NDI-space center point."""
    if loc is None:
        return None
    try:
        x = int((int(loc[0]) + (float(tw) / 2.0)) / max(1e-6, float(scale))) + int(offset_x)
        y = int((int(loc[1]) + (float(th) / 2.0)) / max(1e-6, float(scale))) + int(offset_y)
        return (int(x), int(y))
    except Exception:
        return None


def _challenge_is_safe(
    *,
    chal_val: float,
    chal_center_ndi,
    v_xres: int,
    max_x_frac: float,
    exit_val: float = 0.0,
    exit_margin: float = 0.0,
) -> bool:
    """Return True if it's safe to treat the current END-screen match as Challenge Again.

    Safety rules:
    - Position gating: Challenge Again should be on the left; ignore matches on the right.
    - Optional negative-check: if an exit button template exists and matches nearly as well, don't click.
    """
    if chal_center_ndi is None:
        return False
    try:
        x = float(chal_center_ndi[0])
        w = float(max(1, int(v_xres)))
        if (x / w) > float(max_x_frac):
            return False
        if float(exit_margin) > 0.0 and float(exit_val) > 0.0:
            # If Exit matches almost as well as Challenge, do not click.
            if float(chal_val) < (float(exit_val) + float(exit_margin)):
                return False
    except Exception:
        return False
    return True


async def _scripted_move_then_click(
    makcu,
    *,
    cursor_ndi,
    target_ndi,
    v_xres: int,
    v_yres: int,
    km_calibration,
    max_step_counts: int,
    settle_s: float,
) -> bool:
    """Scripted: compute B-A once, send a single km.move, then click immediately.

    This intentionally does NOT verify cursor correctness before clicking.
    Success is verified by frame templates (e.g., START appearing).
    """
    if cursor_ndi is None or target_ndi is None:
        return False

    ax, ay = int(cursor_ndi[0]), int(cursor_ndi[1])
    bx, by = int(target_ndi[0]), int(target_ndi[1])
    dx_ndi = int(bx - ax)
    dy_ndi = int(by - ay)

    km_dx, km_dy = mh.map_ndidelta_to_km(dx_ndi, dy_ndi, int(v_xres), int(v_yres), km_calibration=km_calibration)
    km_dx = _clip_int(int(km_dx), -int(max_step_counts), int(max_step_counts))
    km_dy = _clip_int(int(km_dy), -int(max_step_counts), int(max_step_counts))

    ok_move = _makcu_transport_send(makcu, f"km.move({km_dx},{km_dy})")
    await asyncio.sleep(float(settle_s))
    ok_click = await mh.makcu_left_click(makcu)
    return bool(ok_move and ok_click)


def load_cursor_template_gray(path):
    if not path:
        return None
    if not os.path.exists(path):
        return None
    img = cv2.imread(path)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def load_cursor_templates_gray(base_path):
    """Load one or more cursor templates.

    If base_path exists, load it.
    Also load any siblings matching cursor_template*.png in the same folder.
    """
    out = []
    if not base_path:
        return out

    # If a folder is provided, load all templates from it.
    if os.path.isdir(base_path):
        for p in sorted(glob.glob(os.path.join(base_path, 'cursor_template*.png'))):
            t = load_cursor_template_gray(p)
            if t is not None:
                out.append(t)
        return out

    # Otherwise treat as a file path.
    t = load_cursor_template_gray(base_path)
    if t is not None:
        out.append(t)

    folder = os.path.dirname(base_path)
    if folder and os.path.isdir(folder):
        for p in sorted(glob.glob(os.path.join(folder, 'cursor_template*.png'))):
            if os.path.abspath(p) == os.path.abspath(base_path):
                continue
            t2 = load_cursor_template_gray(p)
            if t2 is not None:
                out.append(t2)
    return out


def ensure_cursor_templates_folder(script_dir):
    """Ensure cursor templates live in a dedicated folder.

    Returns the folder path. Also migrates old cursor_template*.png files from the script directory.
    """
    folder = os.path.join(script_dir, 'cursor_templates')
    os.makedirs(folder, exist_ok=True)

    # Migrate legacy templates from script dir -> cursor_templates/
    legacy = sorted(glob.glob(os.path.join(script_dir, 'cursor_template*.png')))
    for src in legacy:
        dst = os.path.join(folder, os.path.basename(src))
        if os.path.abspath(src) == os.path.abspath(dst):
            continue
        if os.path.exists(dst):
            # don't overwrite; keep existing
            continue
        try:
            shutil.move(src, dst)
        except Exception:
            # best-effort migration; ignore failures
            pass

    return folder


def load_templates(folder_path, *, exclude_names=None):
    templates = {}
    templates_gray = {}
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")
    exclude = set(exclude_names or [])

    if not os.path.exists(folder_path):
        print(f"[ERROR] Template folder not found: {folder_path}")
        return {}, {}

    print(f"[INIT] Loading images from: {folder_path}")
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(valid_extensions):
            continue
        name = os.path.splitext(filename)[0]
        if name in exclude:
            continue
        path = os.path.join(folder_path, filename)
        img = cv2.imread(path)
        if img is not None:
            templates[name] = img
            templates_gray[name] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = img.shape[:2]
            print(f"   -> Loaded: {name} ({w}x{h})")
        else:
            print(f"   -> [WARN] Failed to load: {filename}")
    return templates, templates_gray


def draw_virtual_cursor(frame, loc, template_shape, scale=1.0):
    # loc is the top-left in preview coords
    h, w = template_shape[:2]
    cx = int((loc[0] + w // 2) * scale)
    cy = int((loc[1] + h // 2) * scale)
    cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (0, 0, 255), 2)
    cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (0, 0, 255), 2)
    cv2.circle(frame, (cx, cy), 15, (0, 0, 255), 2)
    return frame


def _cv_poll_key(delay_ms: int = 1) -> int:
    """Poll GUI key input without blocking the asyncio event loop.

    On some Windows/OpenCV setups, cv2.waitKey(1) can occasionally block much longer
    than expected, which starves asyncio tasks (e.g. the Makcu script).
    """
    try:
        if hasattr(cv2, "pollKey"):
            return int(cv2.pollKey()) & 0xFF
    except Exception:
        pass
    try:
        return int(cv2.waitKey(int(delay_ms))) & 0xFF
    except Exception:
        return -1


async def main():
    parser = argparse.ArgumentParser(description="NDI template bot (recovered)")
    parser.add_argument("--no-hw", action="store_true", help="Don't connect to Makcu hardware (use dummy)")
    parser.add_argument("--telemetry", action="store_true", help="Print periodic perf telemetry")
    parser.add_argument("--telemetry-interval", type=float, default=1.0, help="Telemetry print interval (seconds)")
    parser.add_argument("--log-dir", type=str, default="", help="Write JSONL dataset/action logs to this folder")
    parser.add_argument("--log-frames", action="store_true", help="Save a frame JPEG when an action occurs")
    parser.add_argument("--log-matches", action="store_true", help="Log match events (above --log-match-min)")
    parser.add_argument("--log-match-min", type=float, default=0.60, help="Minimum match score to log when --log-matches is set")
    parser.add_argument("--cursor-template", type=str, default="", help="Path to cursor template file OR folder (defaults to cursor_templates/) ")
    parser.add_argument("--cursor-gain", type=float, default=1.0, help="Proportional gain for cursor feedback movement")
    parser.add_argument("--cursor-min", type=float, default=0.55, help="Minimum cursor template match score")
    parser.add_argument("--cursor-offset-x", type=int, default=0, help="Cursor anchor X offset (pixels) added to detected cursor center")
    parser.add_argument("--cursor-offset-y", type=int, default=0, help="Cursor anchor Y offset (pixels) added to detected cursor center")
    parser.add_argument(
        "--cursor-max-step",
        type=int,
        default=100,
        help="Maximum per-iteration cursor move (km.move counts; also clamps internal pixel estimate)",
    )
    parser.add_argument(
        "--cursor-one-shot-max-step",
        type=int,
        default=2500,
        help="Maximum single-move step for auto one-shot (km.move counts). Set high to allow 1 move like manual B-A.",
    )
    parser.add_argument(
        "--display-fps",
        type=float,
        default=15.0,
        help="Throttle preview rendering to this FPS to avoid UI freezing (processing continues at full speed).",
    )
    parser.add_argument("--cursor-margin", type=int, default=30, help="Safety margin from screen edges (pixels)")
    parser.add_argument("--cursor-max-lost", type=int, default=8, help="Stop moving if cursor is lost this many times after lock")
    parser.add_argument(
        "--km-calibration",
        type=str,
        default="",
        help="Path to makcu_calibration.json (optional; default: ./makcu_calibration.json if present)",
    )
    parser.add_argument(
        "--terminal-control",
        action="store_true",
        help="Enable reading commands from stdin (km.move/...) while running",
    )
    parser.add_argument(
        "--click-min",
        type=float,
        default=0.75,
        help="Minimum template match score to trigger auto point-and-click",
    )
    parser.add_argument(
        "--challenge-click-min",
        type=float,
        default=0.70,
        help="Minimum match score to click challenge_btn (END state). Often lower than --click-min.",
    )
    parser.add_argument(
        "--challenge-click-stable-frames",
        type=int,
        default=2,
        help="Consecutive frames above --challenge-click-min before clicking challenge_btn.",
    )
    parser.add_argument(
        "--challenge-max-x-frac",
        type=float,
        default=0.82,
        help="Safety: only click challenge_btn if its center X is <= this fraction of NDI width (filters far-right Exit while allowing Challenge Again near center-right).",
    )
    parser.add_argument(
        "--challenge-exit-margin",
        type=float,
        default=0.08,
        help="Safety: if an exit_btn template exists and its score is within this margin of challenge_btn, do not click.",
    )
    parser.add_argument(
        "--challenge-unsafe-lockout",
        type=float,
        default=1.2,
        help="Seconds to lock out actions when challenge_btn is detected but deemed unsafe (right-side/Exit-like).",
    )
    parser.add_argument(
        "--click-hitbox-pad",
        type=int,
        default=24,
        help="Extra padding (NDI pixels) added to the button hitbox for deciding when to click.",
    )
    parser.add_argument(
        "--flow",
        type=str,
        default="run_cycle",
        choices=["confirm_to_start", "confirm_only", "run_cycle"],
        help="Automation flow: confirm->start, confirm only, or full run cycle.",
    )
    parser.add_argument(
        "--post-click-verify-timeout",
        type=float,
        default=0.6,
        help="Seconds to wait for the success template after clicking.",
    )
    parser.add_argument(
        "--post-click-verify-min",
        type=float,
        default=0.70,
        help="Minimum match score to consider the post-click success template present.",
    )
    parser.add_argument(
        "--state-detect-min",
        type=float,
        default=0.60,
        help="Minimum match score to consider a UI state template present (combat/end/etc).",
    )
    parser.add_argument(
        "--recover-after-fail",
        action="store_true",
        default=True,
        help="After failed verification, recover cursor to top-left and retry.",
    )
    parser.add_argument(
        "--no-recover-after-fail",
        dest="recover_after_fail",
        action="store_false",
        help="Disable recovery after failed verification.",
    )
    parser.add_argument(
        "--click-stable-frames",
        type=int,
        default=1,
        help="Require this many consecutive frames above --click-min before clicking",
    )
    parser.add_argument(
        "--click-cooldown",
        type=float,
        default=0.2,
        help="Seconds to wait after an auto-click before clicking again",
    )
    parser.add_argument(
        "--no-auto-click",
        action="store_true",
        help="Disable auto point-and-click (use for measurement/debug).",
    )
    parser.add_argument(
        "--action-cooldown",
        type=float,
        default=0.2,
        help="Seconds to wait after an auto action attempt (move/click) before trying again (helps with NDI lag).",
    )
    parser.add_argument(
        "--require-cursor-for-auto",
        dest="require_cursor_for_auto",
        action="store_true",
        default=True,
        help="Only auto act when cursor is currently detected and stable (default: on).",
    )
    parser.add_argument(
        "--no-require-cursor-for-auto",
        dest="require_cursor_for_auto",
        action="store_false",
        help="Allow auto action even without a stable cursor (not recommended).",
    )
    parser.add_argument(
        "--cursor-action-stable-frames",
        type=int,
        default=2,
        help="Require this many consecutive stable cursor frames before auto acting.",
    )
    parser.add_argument(
        "--cursor-action-max-jitter",
        type=float,
        default=6.0,
        help="Max cursor movement (in NDI pixels) between frames to be considered stable.",
    )
    parser.add_argument(
        "--use-opencl",
        dest="use_opencl",
        action="store_true",
        default=None,
        help="Enable OpenCL/UMat acceleration for some OpenCV ops (if available).",
    )
    parser.add_argument(
        "--no-opencl",
        dest="use_opencl",
        action="store_false",
        default=None,
        help="Disable OpenCL/UMat acceleration.",
    )
    parser.add_argument(
        "--pause-at-start",
        action="store_true",
        help="Start paused: never move/click until you unpause with SPACE.",
    )
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        help="Disable rendering and cursor overlays; use only latest-frame template matches for maximum speed.",
    )
    parser.add_argument(
        "--enable-manual-click-target",
        action="store_true",
        help="Enable manual targeting by clicking inside the preview window (prints '[USER] Click at preview ...').",
    )
    parser.add_argument(
        "--fixed-start-ndi",
        type=str,
        default="",
        help="Skip detecting start_btn: after clicking confirm, move+click this fixed START coordinate in NDI pixels (disabled by default).",
    )
    parser.add_argument(
        "--fixed-start-km",
        type=str,
        default="-436,-257",
        help="After clicking confirm, send this fixed km.move(dx,dy) then click (default: -436,-257).",
    )
    parser.add_argument(
        "--fixed-start-delay",
        type=float,
        default=0.35,
        help="Seconds to wait after confirm click before executing fixed START move+click (allows UI to transition).",
    )
    parser.add_argument(
        "--post-start-script-delay",
        type=float,
        default=6.4,
        help="Seconds to wait after the fixed START click before running the Makcu script (default: 10s).",
    )
    parser.add_argument(
        "--doing-yield-s",
        type=float,
        default=0.01,
        help="Extra asyncio sleep added each loop while the Makcu script is pending/running to avoid event-loop starvation (default: 0.01s).",
    )
    parser.add_argument(
        "--use-combat-ui",
        action="store_true",
        help="Use combat_ui template verification to start the Makcu script (default: off; uses --post-start-script-delay instead).",
    )
    parser.add_argument(
        "--post-challenge-km",
        type=str,
        default="-256,-235",
        help="After clicking challenge_btn (END state), send this fixed km.move(dx,dy) then click (default: -256,-235).",
    )
    parser.add_argument(
        "--post-challenge-delay",
        type=float,
        default=0.35,
        help="Seconds to wait after clicking challenge_btn before executing --post-challenge-km move+click.",
    )
    args = parser.parse_args()

    # Defaults: prefer GPU-ish acceleration and stable measurements unless user opts out.
    if args.use_opencl is None:
        args.use_opencl = True

    _report_cv_cuda()
    _configure_cv_opencl(bool(args.use_opencl))

    print("[INIT] Connecting to Makcu Stick...")
    makcu = None
    if args.no_hw:
        print("[INFO] --no-hw specified: using DummyMakcu")
        makcu = DummyMakcu()
    else:
        try:
            makcu = await create_async_controller(fallback_com_port='COM7', auto_reconnect=True)
        except Exception as e:
            print(f"[WARN] Hardware Mouse Failed, falling back to DummyMakcu: {e}")
            makcu = DummyMakcu()

    if not ndi.initialize():
        print("[FATAL] NDI Init Failed")
        return
    ndi_inited = True

    # recv settings
    recv_desc = ndi.RecvCreateV3()
    recv_desc.color_format = ndi.RECV_COLOR_FORMAT_BGRX_BGRA
    recv_desc.bandwidth = ndi.RECV_BANDWIDTH_HIGHEST
    ndi_recv = ndi.recv_create_v3(recv_desc)
    if ndi_recv is None:
        print("[FATAL] recv_create_v3 returned None")
        return

    find = ndi.find_create_v2()
    sources = []
    print("[INIT] Searching for Stream...")
    for _ in range(10):
        ndi.find_wait_for_sources(find, 500)
        sources = ndi.find_get_current_sources(find)
        if sources:
            break

    if not sources:
        print("[FATAL] No NDI Sources found. Is 'NDI Screen Capture' running?")
        return

    print(f"[INIT] Connected to: {sources[0].ndi_name}")
    ndi.recv_connect(ndi_recv, sources[0])
    ndi.find_destroy(find)

    # load templates (color + gray)
    exclude_names = []
    if not bool(args.use_combat_ui):
        exclude_names.append("combat_ui")
    templates, templates_gray = load_templates(TEMPLATE_PATH, exclude_names=exclude_names)
    if "confirm_btn" not in templates:
        print("[ERROR] 'confirm_btn' template missing! Check folder path.")
        return

    # Start producer thread that always keeps the freshest frame.
    frame_src = LatestNDIFrame(ndi_recv, timeout_ms=120)
    frame_src.start()

    print('--- BOT STARTED ---')
    cv2.namedWindow("Bot View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Bot View", 960, 540)

    last_display_ts = 0.0

    # interactive click events (preview coords) - disabled by default to avoid accidental clicks.
    click_events = []
    if bool(args.enable_manual_click_target):
        def _mouse_cb(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONUP:
                click_events.append((x, y))

        cv2.setMouseCallback("Bot View", _mouse_cb)

    printed_res = False
    # cache scaled templates per stream resolution to avoid heavy per-frame work
    scaled_templates = None
    last_xres = 0
    last_yres = 0

    telemetry = Telemetry(report_interval_s=args.telemetry_interval, enabled=bool(args.telemetry))
    ds_logger = DatasetLogger(args.log_dir, enabled=bool(args.log_dir), save_frames=bool(args.log_frames))

    # Cursor templates (optional). You can capture at runtime with key 't'.
    # NOTE: After moving `main.py` to repo root, the runtime assets live under MakcuCommands.
    root_dir = os.path.dirname(__file__)
    assets_dir = MAKCUCOMMANDS_DIR
    cursor_folder = ensure_cursor_templates_folder(assets_dir)
    cursor_template_path = args.cursor_template
    if not cursor_template_path:
        cursor_template_path = cursor_folder

    cursor_template_gray = load_cursor_templates_gray(cursor_template_path)
    if cursor_template_gray:
        print(f"[INIT] Cursor templates loaded: {cursor_template_path} (count={len(cursor_template_gray)})")
    else:
        print("[INIT] No cursor templates loaded. Press 't' to capture one (draw a tight box around the cursor).")

    # Optional interactive terminal control + snapshot-based calibration.
    # Disabled by default because some launchers (e.g. VS Code Debug Console) can mis-wire stdin.
    cmd_queue: "queue.Queue[str]" = queue.Queue()
    cmd_stop = threading.Event()
    terminal_control_enabled = bool(args.terminal_control)

    if terminal_control_enabled:
        def _stdin_reader():
            while not cmd_stop.is_set():
                try:
                    line = sys.stdin.readline()
                    if not line:
                        time.sleep(0.05)
                        continue
                    cmd_queue.put(line.strip())
                except Exception:
                    time.sleep(0.10)

        try:
            t_stdin = threading.Thread(target=_stdin_reader, name="stdin-reader", daemon=True)
            t_stdin.start()
            print("[INIT] Terminal control enabled: type commands like `km.move(10,0)` while running.")
            print("[INIT] Snapshot calibration: press 's' before/after each typed km.move; press 'w' to write makcu_calibration.json")
        except Exception:
            terminal_control_enabled = False
            print("[WARN] Failed to start stdin reader; terminal command input disabled")

    # Snapshot calibration state
    calib_dir = os.path.join(root_dir, "calibration_frames")
    os.makedirs(calib_dir, exist_ok=True)
    calib_idx = 0
    calib_pre_cursor = None  # (x,y)
    calib_pending_move = None  # (dx,dy) in km.move counts
    calib_samples_x = []  # list[float] px_per_count
    calib_samples_y = []  # list[float] px_per_count
    calib_sign_samples_x = []  # list[int] (+1/-1)
    calib_sign_samples_y = []  # list[int] (+1/-1)

    # Optional: load km.move calibration (pixels-per-count + sign) if available.
    km_calib_path = (args.km_calibration or "").strip()
    if not km_calib_path:
        # Prefer the MakcuCommands calibration file (tracked with the mode assets).
        cand = os.path.join(assets_dir, "makcu_calibration.json")
        if os.path.exists(cand):
            km_calib_path = cand
        else:
            # Backwards-compatible fallback for older layouts.
            km_calib_path = os.path.join(root_dir, "makcu_calibration.json")
    km_calib_path = os.path.abspath(km_calib_path)

    km_calibration = mh.load_km_calibration(km_calib_path)
    if km_calibration:
        print(f"[INIT] Loaded km calibration: {km_calib_path} -> {km_calibration}")
    else:
        if (args.km_calibration or "").strip():
            print(f"[WARN] --km-calibration was set but could not be loaded: {km_calib_path}")
        else:
            print("[INIT] No km calibration found (makcu_calibration.json). Using approximate mapping.")

    # Latest detected cursor in NDI space (continuous tracking for UI + seeding).
    last_cursor_ndi = None  # (x,y)
    last_cursor_score = 0.0
    last_cursor_ts = 0.0
    cursor_candidate = None  # (x,y)
    cursor_candidate_hits = 0
    cursor_freeze_until_ts = 0.0

    # Latest target (for overlay line). By default this is the current best template match
    # (so you can see if the bot "knows" the target even before clicking).
    last_target_preview = None  # (x,y) in preview coords
    last_target_ndi = None      # (x,y) in NDI coords
    last_target_score = 0.0
    last_target_ts = 0.0

    # Current target (what the bot intends to click next).
    current_target_label = None  # str
    current_target_preview = None  # (x,y) preview coords
    current_target_ndi = None  # (x,y) NDI coords
    current_target_score = 0.0
    current_target_ts = 0.0
    last_target_print_ts = 0.0
    last_target_print_key = None

    # Auto-click debounce state
    stable_hits = 0
    last_auto_click_ts = 0.0

    # Prevent repeated actions while waiting for NDI frames to catch up.
    action_lockout_until = 0.0

    # Cursor stability gating for auto actions.
    cursor_action_stable_hits = 0
    last_cursor_for_action = None

    def _assume_cursor_at(ndi_xy) -> None:
        """After a scripted move, assume the cursor is now at the target.

        This avoids waiting on vision cursor detection between consecutive actions.
        """
        nonlocal last_cursor_ndi, last_cursor_ts, last_cursor_score, cursor_action_stable_hits, last_cursor_for_action, cursor_freeze_until_ts
        try:
            last_cursor_ndi = (int(ndi_xy[0]), int(ndi_xy[1]))
            last_cursor_ts = time.time()
            # treat as confident enough for gating
            last_cursor_score = 1.0
            last_cursor_for_action = last_cursor_ndi
            cursor_action_stable_hits = int(args.cursor_action_stable_frames)
            # After a scripted move/click, ignore vision cursor updates briefly (NDI lag + false positives).
            cursor_freeze_until_ts = float(last_cursor_ts) + 0.75
        except Exception:
            pass

    fixed_start_ndi = _parse_ndi_point(getattr(args, 'fixed_start_ndi', '') or '')
    fixed_start_km = _parse_ndi_point(getattr(args, 'fixed_start_km', '') or '')
    fixed_start_mode = None  # None | 'km' | 'ndi'
    if fixed_start_km is not None:
        fixed_start_mode = 'km'
        print(f"[INIT] Fixed-start (km) enabled: km.move{tuple(fixed_start_km)} delay={float(args.fixed_start_delay):.2f}s")
    elif fixed_start_ndi is not None:
        fixed_start_mode = 'ndi'
        print(f"[INIT] Fixed-start (ndi) enabled: START_NDI={fixed_start_ndi} delay={float(args.fixed_start_delay):.2f}s")

    pending_fixed_start_ts = 0.0

    post_challenge_km = _parse_ndi_point(getattr(args, 'post_challenge_km', '') or '')
    post_challenge_enabled = post_challenge_km is not None
    if post_challenge_enabled:
        print(f"[INIT] Post-challenge fixed (km) enabled: km.move{tuple(post_challenge_km)} delay={float(args.post_challenge_delay):.2f}s")

    pending_post_challenge_ts = 0.0

    # Flow state
    flow_state = "CONFIRM"  # CONFIRM -> START -> COMBAT -> END -> CHALLENGE -> START

    # Fast tracking of button locations in preview space.
    last_confirm_loc = None
    last_start_loc = None

    # Non-blocking post-click verification state (avoid freezing the loop)
    pending_verify_template = None
    pending_verify_deadline_ts = 0.0
    pending_verify_min = 0.0
    pending_verify_success_state = None
    pending_verify_fail_state = None

    # Debug / measurement control
    paused = bool(args.pause_at_start)
    if paused:
        print('[INIT] PAUSED at start (press SPACE in the Bot View window to toggle)')

    # Combat script state
    combat_task = None
    combat_cancel = asyncio.Event()
    last_wait_combat_log_ts = 0.0
    post_start_script_at_ts = 0.0
    awaiting_challenge = False

    while True:
        # use a short timeout so the loop frequently returns to pump GUI events
        loop_t0 = time.perf_counter()
        cap_t0 = loop_t0
        latest = frame_src.get_latest()
        capture_s = time.perf_counter() - cap_t0
        match_s = 0.0
        ok_frame = False
        if latest is not None:
            frame_bgr, xres, yres, frame_ts = latest
            v = types.SimpleNamespace(xres=int(xres), yres=int(yres))
            ok_frame = True

            # Harvest combat script completion.
            if combat_task is not None and combat_task.done():
                try:
                    combat_task.result()
                except Exception as e:
                    print(f"[DOING] combat script error: {e}")
                combat_task = None
                combat_cancel.clear()
                flow_state = "END"
                awaiting_challenge = True

            # Timed start of the Makcu script (skip combat_ui verification path).
            if (not bool(args.use_combat_ui)) and post_start_script_at_ts > 0.0:
                if (not paused) and (combat_task is None or combat_task.done()) and time.time() >= float(post_start_script_at_ts):
                    combat_cancel.clear()
                    combat_task = asyncio.create_task(defence_wedge_65_makcu.run_combat(makcu, cancel_event=combat_cancel))
                    print("[DOING] Started defence_wedge_65_makcu.run_combat() (timed)")
                    flow_state = "DOING"
                    post_start_script_at_ts = 0.0

            if not printed_res and v.xres and v.yres:
                print(f"[NDI] Stream Res: {v.xres}x{v.yres}")
                printed_res = True

            # If the Makcu script is pending/running, keep the loop extremely light.
            # This avoids starving the asyncio task (which shows up as 1–2s delays between actions).
            doing_active_now = (combat_task is not None) and (not combat_task.done())
            script_wait_now = float(post_start_script_at_ts) > 0.0
            if doing_active_now or script_wait_now:
                if not args.fast_mode:
                    key = _cv_poll_key(1)
                    if key == ord(' ') :
                        paused = not paused
                        print(f"[MODE] paused={paused}")
                        if paused and combat_task is not None and not combat_task.done():
                            combat_cancel.set()
                    if key == ord('q') or key == 27:
                        break
                try:
                    if cv2.getWindowProperty("Bot View", cv2.WND_PROP_VISIBLE) < 1:
                        print("[INFO] Bot View window closed by user")
                        break
                except Exception:
                    pass

                telemetry.on_frame(ok=ok_frame, loop_s=(time.perf_counter() - loop_t0), capture_s=capture_s, match_s=0.0)
                snap = telemetry.maybe_report()
                if snap is not None:
                    fps = (snap.frames_ok / max(1e-6, snap.window_s))
                    print(
                        f"[TEL] fps={fps:.1f} ok={snap.frames_ok} drop={snap.frames_drop} actions={snap.actions} "
                        f"loop={snap.avg_loop_ms:.1f}ms cap={snap.avg_capture_ms:.1f}ms match={snap.avg_match_ms:.1f}ms"
                    )

                # Yield a bit so the script task runs with stable timing.
                await asyncio.sleep(max(0.005, float(getattr(args, 'doing_yield_s', 0.01) or 0.01)))
                continue

            # Prepare small grayscale for fast matching
            scale = PREVIEW_WIDTH / float(v.xres)
            disp_h_small = int(PREVIEW_WIDTH * v.yres / max(1, v.xres))
            if args.use_opencl:
                small_u = cv2.resize(cv2.UMat(frame_bgr), (PREVIEW_WIDTH, disp_h_small))
                gray_small_u = cv2.cvtColor(small_u, cv2.COLOR_BGR2GRAY)
                small = small_u.get()
                gray_small = gray_small_u
            else:
                small = cv2.resize(frame_bgr, (PREVIEW_WIDTH, disp_h_small))
                gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

            # Scale templates once per stream resolution (cached)
            if scaled_templates is None or last_xres != v.xres or last_yres != v.yres:
                scaled_templates = {}
                scaled_templates_u = {}
                for name, templ_gray in templates_gray.items():
                    th, tw = templ_gray.shape[:2]
                    tw_s = max(1, int(tw * scale))
                    th_s = max(1, int(th * scale))
                    t_small = cv2.resize(templ_gray, (tw_s, th_s), interpolation=cv2.INTER_AREA)
                    scaled_templates[name] = t_small
                    if args.use_opencl:
                        try:
                            scaled_templates_u[name] = cv2.UMat(t_small)
                        except Exception:
                            scaled_templates_u[name] = None
                last_xres = v.xres
                last_yres = v.yres

            # Match
            match_t0 = time.perf_counter()
            gray_for_match = gray_small

            # While the Makcu script is pending/running, avoid heavy template matching.
            # Otherwise this loop can hog CPU and starve the asyncio task, causing extra delays
            # between scripted commands compared to running the script standalone.
            doing_active_now = (combat_task is not None) and (not combat_task.done())
            script_wait_now = float(post_start_script_at_ts) > 0.0
            reduce_load_now = bool(doing_active_now or script_wait_now)

            # If we are waiting for a post-click UI transition, suppress new actions.
            # Do NOT extend action_lockout_until to the deadline; that causes unnecessary waiting
            # even when the success template appears quickly.

            if reduce_load_now:
                confirm_val, confirm_loc = 0.0, (0, 0)
                start_val, start_loc = 0.0, (0, 0)
            else:
                # Performance: use local/ROI tracking for the two most frequently checked buttons.
                confirm_val, confirm_loc = _match_scaled_tracked(
                    gray_for_match,
                    scaled_templates,
                    scaled_templates_u,
                    "confirm_btn",
                    use_opencl=bool(args.use_opencl),
                    last_loc=last_confirm_loc,
                    search_radius=180,
                )
                start_val, start_loc = _match_scaled_tracked(
                    gray_for_match,
                    scaled_templates,
                    scaled_templates_u,
                    "start_btn",
                    use_opencl=bool(args.use_opencl),
                    last_loc=last_start_loc,
                    search_radius=220,
                )
                # Update trackers (keep even if score is low; it helps the next ROI)
                last_confirm_loc = (int(confirm_loc[0]), int(confirm_loc[1]))
                last_start_loc = (int(start_loc[0]), int(start_loc[1]))

            combat_val, combat_loc = 0.0, (0, 0)
            end_val, end_loc = 0.0, (0, 0)
            chal_val, chal_loc = 0.0, (0, 0)
            exit_val, exit_loc = 0.0, (0, 0)

            # Only check combat/end/challenge when run_cycle is enabled or when pending verification needs them.
            if (not reduce_load_now) and args.flow == "run_cycle":
                # When waiting for the run to finish, keep scanning for the Challenge Again button.
                if awaiting_challenge:
                    chal_val, chal_loc = _match_scaled(
                        gray_for_match,
                        scaled_templates,
                        scaled_templates_u,
                        "challenge_btn",
                        use_opencl=bool(args.use_opencl),
                    )
                    # Optional safety: also match Exit if template exists.
                    if "exit_btn" in (scaled_templates or {}):
                        exit_val, exit_loc = _match_scaled(
                            gray_for_match,
                            scaled_templates,
                            scaled_templates_u,
                            "exit_btn",
                            use_opencl=bool(args.use_opencl),
                        )
                elif bool(args.use_combat_ui):
                    if flow_state == "COMBAT":
                        end_val, end_loc = _match_scaled(
                            gray_for_match,
                            scaled_templates,
                            scaled_templates_u,
                            "end_screen",
                            use_opencl=bool(args.use_opencl),
                        )
                    else:
                        combat_val, combat_loc = _match_scaled(
                            gray_for_match,
                            scaled_templates,
                            scaled_templates_u,
                            "combat_ui",
                            use_opencl=bool(args.use_opencl),
                        )
                        if float(combat_val) < float(args.state_detect_min):
                            end_val, end_loc = _match_scaled(
                                gray_for_match,
                                scaled_templates,
                                scaled_templates_u,
                                "end_screen",
                                use_opencl=bool(args.use_opencl),
                            )

                    # Only match challenge when we believe we're on end screen.
                    if float(end_val) >= float(args.state_detect_min):
                        chal_val, chal_loc = _match_scaled(
                            gray_for_match,
                            scaled_templates,
                            scaled_templates_u,
                            "challenge_btn",
                            use_opencl=bool(args.use_opencl),
                        )
                        if "exit_btn" in (scaled_templates or {}):
                            exit_val, exit_loc = _match_scaled(
                                gray_for_match,
                                scaled_templates,
                                scaled_templates_u,
                                "exit_btn",
                                use_opencl=bool(args.use_opencl),
                            )

            # Ensure we compute any pending verification template.
            if (not reduce_load_now) and pending_verify_template is not None:
                if pending_verify_template == "combat_ui" and float(combat_val) <= 0.0 and bool(args.use_combat_ui):
                    combat_val, combat_loc = _match_scaled(
                        gray_for_match,
                        scaled_templates,
                        scaled_templates_u,
                        "combat_ui",
                        use_opencl=bool(args.use_opencl),
                    )
                if pending_verify_template == "end_screen" and float(end_val) <= 0.0:
                    end_val, end_loc = _match_scaled(
                        gray_for_match,
                        scaled_templates,
                        scaled_templates_u,
                        "end_screen",
                        use_opencl=bool(args.use_opencl),
                    )
                if pending_verify_template == "challenge_btn" and float(chal_val) <= 0.0:
                    chal_val, chal_loc = _match_scaled(
                        gray_for_match,
                        scaled_templates,
                        scaled_templates_u,
                        "challenge_btn",
                        use_opencl=bool(args.use_opencl),
                    )
                if pending_verify_template == "challenge_btn" and float(exit_val) <= 0.0 and "exit_btn" in (scaled_templates or {}):
                    exit_val, exit_loc = _match_scaled(
                        gray_for_match,
                        scaled_templates,
                        scaled_templates_u,
                        "exit_btn",
                        use_opencl=bool(args.use_opencl),
                    )

            # Resolve pending verification using current-frame matches.
            if pending_verify_template is not None:
                now_v = time.time()
                score_map = {
                    "start_btn": float(start_val),
                    "combat_ui": float(combat_val),
                    "end_screen": float(end_val),
                    "challenge_btn": float(chal_val),
                    "confirm_btn": float(confirm_val),
                }
                cur_score = float(score_map.get(str(pending_verify_template), 0.0))

                # Periodic visibility while waiting for combat_ui.
                if bool(args.use_combat_ui) and str(pending_verify_template) == "combat_ui" and (now_v - float(last_wait_combat_log_ts)) >= 0.50:
                    last_wait_combat_log_ts = float(now_v)
                    print(
                        f"[WAIT] combat_ui score={cur_score:.2f} min={float(pending_verify_min):.2f} "
                        f"until={max(0.0, float(pending_verify_deadline_ts) - now_v):.2f}s"
                    )

                if cur_score >= float(pending_verify_min):
                    verified_t = str(pending_verify_template)
                    print(f"[FLOW] Verified: {verified_t} score={cur_score:.2f}")

                    # If combat_ui is verified, run the combat Makcu script once.
                    if verified_t == "combat_ui" and bool(args.use_combat_ui):
                        if combat_task is None or combat_task.done():
                            combat_cancel.clear()
                            combat_task = asyncio.create_task(
                                defence_wedge_65_makcu.run_combat(makcu, cancel_event=combat_cancel)
                            )
                            print("[DOING] Started defence_wedge_65_makcu.run_combat()")
                        flow_state = "DOING"
                    else:
                        if pending_verify_success_state is not None:
                            flow_state = str(pending_verify_success_state)
                    pending_verify_template = None
                    # Release lockout quickly so the next action can happen immediately.
                    action_lockout_until = min(float(action_lockout_until), time.time() + 0.05)
                elif now_v >= float(pending_verify_deadline_ts):
                    print(f"[FLOW] Verify timeout: {pending_verify_template} (score={cur_score:.2f})")
                    timed_out_template = str(pending_verify_template)
                    if pending_verify_fail_state is not None:
                        flow_state = str(pending_verify_fail_state)
                    pending_verify_template = None
                    action_lockout_until = min(float(action_lockout_until), time.time() + 0.05)

                    # In fixed-start mode, don't fall back to template start_btn clicking.
                    # Instead, schedule a fixed START retry after a short delay.
                    if fixed_start_mode is not None and timed_out_template == "combat_ui" and bool(args.use_combat_ui):
                        pending_fixed_start_ts = time.time() + max(0.25, float(args.fixed_start_delay))
                        if fixed_start_mode == 'km' and fixed_start_km is not None:
                            print(f"[FLOW] Scheduling fixed START retry -> km.move{tuple(fixed_start_km)}")
                        else:
                            print(f"[FLOW] Scheduling fixed START retry -> {fixed_start_ndi}")
                    if bool(args.recover_after_fail):
                        print('[FLOW] Recovering cursor to top-left and retrying')
                        _makcu_transport_send(makcu, 'km.move(-3000,-3000)')
                        action_lockout_until = max(float(action_lockout_until), time.time() + 0.35)

            # If we have a fixed START click scheduled, execute it as soon as allowed.
            if fixed_start_mode is not None and pending_fixed_start_ts > 0 and (time.time() >= float(pending_fixed_start_ts)):
                # Only fire when not already waiting on another verification.
                if pending_verify_template is None and (time.time() >= float(action_lockout_until)) and (not paused) and (not args.no_auto_click):
                    ok2 = False
                    if fixed_start_mode == 'km' and fixed_start_km is not None:
                        dx_cnt, dy_cnt = int(fixed_start_km[0]), int(fixed_start_km[1])
                        cmd = f"km.move({dx_cnt},{dy_cnt})"
                        print(f"[FIXED] START {cmd} then click")
                        ok_move = _makcu_transport_send(makcu, cmd)
                        await asyncio.sleep(0.08)
                        ok_click = await mh.makcu_left_click(makcu, hold_s=0.03)
                        ok2 = bool(ok_move) and bool(ok_click)

                        # Best-effort assumed cursor update for overlays.
                        if last_cursor_ndi is not None and (time.time() - float(last_cursor_ts)) < 2.0:
                            dx_ndi, dy_ndi = _km_counts_to_ndi_delta(dx_cnt, dy_cnt, km_calibration=km_calibration)
                            _assume_cursor_at((int(last_cursor_ndi[0]) + int(dx_ndi), int(last_cursor_ndi[1]) + int(dy_ndi)))
                    elif fixed_start_mode == 'ndi' and fixed_start_ndi is not None:
                        # We assume the cursor is currently at the last confirm click location (set by _assume_cursor_at).
                        if last_cursor_ndi is None or (time.time() - float(last_cursor_ts)) > 2.0:
                            print('[FIXED] Cursor unknown for fixed START; recovering then retry')
                            _makcu_transport_send(makcu, 'km.move(-3000,-3000)')
                            action_lockout_until = max(float(action_lockout_until), time.time() + 0.35)
                        else:
                            print(f"[FIXED] START move+click -> {fixed_start_ndi}")
                            ok2 = await _scripted_move_then_click(
                                makcu,
                                cursor_ndi=last_cursor_ndi,
                                target_ndi=fixed_start_ndi,
                                v_xres=int(v.xres),
                                v_yres=int(v.yres),
                                km_calibration=km_calibration,
                                max_step_counts=int(args.cursor_one_shot_max_step),
                                settle_s=0.16,
                            )
                            if ok2:
                                _assume_cursor_at(fixed_start_ndi)

                    if not bool(ok2):
                        # If the move/click failed, retry START soon.
                        pending_fixed_start_ts = time.time() + max(0.25, float(args.fixed_start_delay))
                        print("[FLOW] Fixed START action failed; retrying")
                    else:
                        if bool(args.use_combat_ui):
                            # Verify combat UI after clicking start (both km/ndi modes).
                            pending_verify_template = "combat_ui"
                            pending_verify_min = float(args.post_click_verify_min)
                            pending_verify_deadline_ts = time.time() + float(args.post_click_verify_timeout)
                            pending_verify_success_state = "COMBAT"
                            pending_verify_fail_state = "START"
                            action_lockout_until = max(float(action_lockout_until), time.time() + 0.10)
                            flow_state = "START"
                        else:
                            # Skip combat_ui: wait a bit then run the Makcu script.
                            post_start_script_at_ts = time.time() + float(args.post_start_script_delay)
                            awaiting_challenge = False
                            action_lockout_until = max(float(action_lockout_until), time.time() + 0.10)
                            flow_state = "DOING"

                        # Success: don't keep firing repeatedly.
                        pending_fixed_start_ts = 0.0

            # After clicking Challenge Again, optionally do a fixed confirm-like move+click.
            if post_challenge_enabled and pending_post_challenge_ts > 0 and (time.time() >= float(pending_post_challenge_ts)):
                if pending_verify_template is None and (time.time() >= float(action_lockout_until)) and (not paused) and (not args.no_auto_click):
                    dx_cnt, dy_cnt = int(post_challenge_km[0]), int(post_challenge_km[1])
                    cmd = f"km.move({dx_cnt},{dy_cnt})"
                    print(f"[FIXED] POST-CHALLENGE {cmd} then click")
                    ok_move = _makcu_transport_send(makcu, cmd)
                    await asyncio.sleep(0.08)
                    ok_click = await mh.makcu_left_click(makcu, hold_s=0.03)
                    ok3 = bool(ok_move) and bool(ok_click)

                    # Best-effort assumed cursor update for overlays.
                    if ok3 and last_cursor_ndi is not None and (time.time() - float(last_cursor_ts)) < 2.0:
                        dx_ndi, dy_ndi = _km_counts_to_ndi_delta(dx_cnt, dy_cnt, km_calibration=km_calibration)
                        _assume_cursor_at((int(last_cursor_ndi[0]) + int(dx_ndi), int(last_cursor_ndi[1]) + int(dy_ndi)))

                    if ok3 and fixed_start_mode is not None:
                        pending_fixed_start_ts = time.time() + float(args.fixed_start_delay)
                        flow_state = "START"
                    action_lockout_until = max(float(action_lockout_until), time.time() + 0.10)
                    pending_post_challenge_ts = 0.0

            # Keep existing variables for downstream drawing/logic (confirm).
            max_val = float(confirm_val)
            max_loc = (int(confirm_loc[0]), int(confirm_loc[1]))

            match_s = time.perf_counter() - match_t0

            if ds_logger.enabled and args.log_matches and max_val >= float(args.log_match_min):
                try:
                    ds_logger.log_match(
                        template="confirm_btn",
                        match_val=float(max_val),
                        match_loc=(int(max_loc[0]), int(max_loc[1])),
                        ndi_res=(int(v.xres), int(v.yres)),
                    )
                except Exception:
                    pass

            now_ts = time.time()
            render_now = True
            try:
                fps_limit = float(args.display_fps)
                if fps_limit > 0:
                    render_now = (now_ts - float(last_display_ts)) >= (1.0 / fps_limit)
            except Exception:
                render_now = True

            if args.fast_mode:
                render_now = False

            display = small.copy() if render_now else None

            # Publish the current (intended) target *before* drawing overlays.
            # This makes pause mode useful: you can see exactly what coordinate/template is being targeted.
            try:
                now_t = time.time()
                tgt_label = None
                tgt_score = 0.0
                tgt_ndi = None

                # If a fixed START click is scheduled, that is the next intended target.
                if fixed_start_mode is not None and float(pending_fixed_start_ts) > 0:
                    tgt_score = 1.0
                    if fixed_start_mode == 'km' and fixed_start_km is not None:
                        tgt_label = f"fixed_km{tuple(fixed_start_km)}"
                        if last_cursor_ndi is not None and (now_t - float(last_cursor_ts)) < 2.0:
                            dx_ndi, dy_ndi = _km_counts_to_ndi_delta(int(fixed_start_km[0]), int(fixed_start_km[1]), km_calibration=km_calibration)
                            tgt_ndi = (int(last_cursor_ndi[0]) + int(dx_ndi), int(last_cursor_ndi[1]) + int(dy_ndi))
                        else:
                            tgt_ndi = None
                    else:
                        tgt_label = "fixed_start"
                        tgt_ndi = (int(fixed_start_ndi[0]), int(fixed_start_ndi[1]))
                else:
                    # Otherwise, the intended target is the active flow button.
                    if flow_state == "CONFIRM":
                        tgt_label = "confirm_btn"
                        tgt_score = float(confirm_val)
                        loc = confirm_loc
                        th, tw = scaled_templates["confirm_btn"].shape[:2]
                    elif flow_state == "START":
                        # In fixed-start mode, we intentionally avoid start_btn detection clicks.
                        if fixed_start_mode is not None:
                            tgt_score = 1.0
                            if fixed_start_mode == 'km' and fixed_start_km is not None:
                                tgt_label = f"fixed_km{tuple(fixed_start_km)}"
                                if last_cursor_ndi is not None and (now_t - float(last_cursor_ts)) < 2.0:
                                    dx_ndi, dy_ndi = _km_counts_to_ndi_delta(int(fixed_start_km[0]), int(fixed_start_km[1]), km_calibration=km_calibration)
                                    tgt_ndi = (int(last_cursor_ndi[0]) + int(dx_ndi), int(last_cursor_ndi[1]) + int(dy_ndi))
                                else:
                                    tgt_ndi = None
                            else:
                                tgt_label = "fixed_start"
                                tgt_ndi = (int(fixed_start_ndi[0]), int(fixed_start_ndi[1]))
                            loc = None
                        else:
                            tgt_label = "start_btn"
                            tgt_score = float(start_val)
                            loc = start_loc
                            th, tw = scaled_templates["start_btn"].shape[:2]
                    elif flow_state == "END":
                        tgt_label = "challenge_btn"
                        tgt_score = float(chal_val)
                        loc = chal_loc
                        th, tw = scaled_templates["challenge_btn"].shape[:2]
                    else:
                        tgt_label = None
                        loc = None

                    if tgt_ndi is None and loc is not None:
                        orig_x = int((int(loc[0]) + (tw / 2.0)) / max(1e-6, scale)) + CLICK_OFFSET_X
                        orig_y = int((int(loc[1]) + (th / 2.0)) / max(1e-6, scale)) + CLICK_OFFSET_Y
                        tgt_ndi = (int(orig_x), int(orig_y))

                # Update current target state.
                current_target_label = tgt_label
                current_target_score = float(tgt_score)
                current_target_ndi = tgt_ndi
                current_target_ts = now_t

                # Also update legacy target variables so hotkeys (D/K) keep working.
                if tgt_ndi is not None:
                    last_target_ndi = tgt_ndi
                    last_target_score = float(tgt_score)
                    last_target_ts = now_t
                    if display is not None:
                        disp_h, disp_w = display.shape[:2]
                        last_target_preview = (
                            int(float(tgt_ndi[0] - CLICK_OFFSET_X) / float(v.xres) * float(disp_w)),
                            int(float(tgt_ndi[1] - CLICK_OFFSET_Y) / float(v.yres) * float(disp_h)),
                        )

                # Print target coordinates when it changes (throttled).
                key = (current_target_label, current_target_ndi)
                if key != last_target_print_key and (now_t - float(last_target_print_ts)) >= 0.15:
                    last_target_print_key = key
                    last_target_print_ts = now_t
                    if current_target_ndi is not None:
                        msg = f"[TARGET] flow={flow_state} target={current_target_label} score={float(current_target_score):.2f} NDI={current_target_ndi}"
                        if last_cursor_ndi is not None and (now_t - float(last_cursor_ts)) < 1.0:
                            dx_ndi = int(current_target_ndi[0] - int(last_cursor_ndi[0]))
                            dy_ndi = int(current_target_ndi[1] - int(last_cursor_ndi[1]))
                            msg += f" A(cursor)={tuple(map(int, last_cursor_ndi))} B-A=({dx_ndi},{dy_ndi})"
                        print(msg)
            except Exception:
                pass

            # Always-on cursor detection for overlay + seeding (skip in fast-mode to save time).
            if (not args.fast_mode) and cursor_template_gray and (time.time() >= float(cursor_freeze_until_ts)):
                try:
                    score_c, cx_c, cy_c = mh.detect_cursor_in_frame(frame_bgr, cursor_template_gray, match_min=args.cursor_min)
                    if cx_c is not None:
                        cand = (int(cx_c) + int(args.cursor_offset_x), int(cy_c) + int(args.cursor_offset_y))

                        # Anti-jump filter: large jumps require much higher confidence.
                        if last_cursor_ndi is not None:
                            j = math.hypot(cand[0] - last_cursor_ndi[0], cand[1] - last_cursor_ndi[1])
                            if j > 220 and float(score_c) < float(args.cursor_min) + 0.20:
                                cand = None

                        # Two-hit stabilization: only promote a candidate after 2 consecutive nearby detections.
                        if cand is not None:
                            if cursor_candidate is None:
                                cursor_candidate = cand
                                cursor_candidate_hits = 1
                            else:
                                dj = math.hypot(cand[0] - cursor_candidate[0], cand[1] - cursor_candidate[1])
                                if dj <= 40:
                                    cursor_candidate_hits += 1
                                else:
                                    cursor_candidate = cand
                                    cursor_candidate_hits = 1

                            if cursor_candidate_hits >= 2:
                                last_cursor_ndi = cursor_candidate
                                last_cursor_score = float(score_c)
                                last_cursor_ts = time.time()

                                # Cursor stability gating for auto actions.
                                # Require N consecutive frames where cursor doesn't jitter much.
                                if last_cursor_for_action is None:
                                    last_cursor_for_action = last_cursor_ndi
                                    cursor_action_stable_hits = 1
                                else:
                                    dj_action = math.hypot(
                                        float(last_cursor_ndi[0] - last_cursor_for_action[0]),
                                        float(last_cursor_ndi[1] - last_cursor_for_action[1]),
                                    )
                                    if dj_action <= float(args.cursor_action_max_jitter):
                                        cursor_action_stable_hits += 1
                                    else:
                                        last_cursor_for_action = last_cursor_ndi
                                        cursor_action_stable_hits = 1
                except Exception:
                    pass

            # Draw cursor marker on preview.
            if render_now and last_cursor_ndi is not None and (time.time() - last_cursor_ts) < 1.0:
                disp_h, disp_w = display.shape[:2]
                px = int(last_cursor_ndi[0] / float(v.xres) * disp_w)
                py = int(last_cursor_ndi[1] / float(v.yres) * disp_h)
                cv2.circle(display, (px, py), 9, (0, 0, 255), 2)
                cv2.putText(
                    display,
                    f"cursor {last_cursor_score:.2f} ({int(last_cursor_ndi[0])},{int(last_cursor_ndi[1])})",
                    (max(5, px + 12), max(20, py - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 255),
                    2,
                )

                # Draw line from cursor -> current target (if set).
                # Prefer current_target_ndi (robust even if preview bookkeeping is stale).
                tx = ty = None
                if current_target_ndi is not None and (time.time() - float(current_target_ts)) < 2.0:
                    try:
                        tx = int(float(current_target_ndi[0] - CLICK_OFFSET_X) / float(v.xres) * float(disp_w))
                        ty = int(float(current_target_ndi[1] - CLICK_OFFSET_Y) / float(v.yres) * float(disp_h))
                    except Exception:
                        tx = ty = None
                if (tx is None or ty is None) and last_target_preview is not None and (time.time() - last_target_ts) < 2.0:
                    tx, ty = int(last_target_preview[0]), int(last_target_preview[1])

                if tx is not None and ty is not None:
                    cv2.line(display, (px, py), (tx, ty), (0, 255, 255), 2)

                    # Show delta (B-A) on the line so you can manually test km.move.
                    # A = cursor (optionally adjusted by cursor_offset), B = target.
                    # Use NDI coords for delta computation.
                    if last_target_ndi is not None and last_cursor_ndi is not None:
                        # NOTE: last_cursor_ndi already includes cursor_offset_x/y (applied at detection time).
                        ax_ndi = int(last_cursor_ndi[0])
                        ay_ndi = int(last_cursor_ndi[1])
                        bx_ndi = int(last_target_ndi[0])
                        by_ndi = int(last_target_ndi[1])
                        dx_ndi = int(bx_ndi - ax_ndi)
                        dy_ndi = int(by_ndi - ay_ndi)

                        km_dx, km_dy = mh.map_ndidelta_to_km(
                            dx_ndi,
                            dy_ndi,
                            int(v.xres),
                            int(v.yres),
                            km_calibration=km_calibration,
                        )

                        midx = int((px + tx) / 2)
                        midy = int((py + ty) / 2)
                        dist = math.hypot(float(dx_ndi), float(dy_ndi))
                        lbl = current_target_label or "target"
                        txt = f"{lbl}  B-A ndi=({dx_ndi},{dy_ndi}) |d|={dist:.1f}  km=({int(km_dx)},{int(km_dy)})"
                        # Outline for readability
                        cv2.putText(display, txt, (max(5, midx + 8), max(18, midy - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 3)
                        cv2.putText(display, txt, (max(5, midx + 8), max(18, midy - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)

                    cv2.circle(display, (tx, ty), 8, (0, 255, 255), 2)
                    cv2.putText(
                        display,
                        f"target {last_target_score:.2f} ({int(last_target_ndi[0])},{int(last_target_ndi[1])})" if last_target_ndi is not None else f"target {last_target_score:.2f}",
                        (max(5, tx + 12), max(20, ty - 12)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 255),
                        2,
                    )
            # Draw virtual cursor for low-confidence positives
            if render_now and max_val > 0.6:
                draw_virtual_cursor(display, max_loc, scaled_templates["confirm_btn"].shape, scale=1.0)

            # Debug flow overlay
            if render_now:
                cv2.putText(
                    display,
                    f"flow={flow_state} paused={paused} confirm={confirm_val:.2f} start={start_val:.2f} combat={combat_val:.2f} end={end_val:.2f} chal={chal_val:.2f}",
                    (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (255, 255, 255),
                    2,
                )
                if current_target_label is not None and current_target_ndi is not None:
                    cv2.putText(
                        display,
                        f"target={current_target_label} score={float(current_target_score):.2f} NDI=({int(current_target_ndi[0])},{int(current_target_ndi[1])})",
                        (8, 46),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.58,
                        (0, 255, 255),
                        2,
                    )

            # (target publish moved earlier so the line renders immediately)

            # Auto point-and-click when the active flow button is stably detected.
            # Disabled in measurement mode or when paused.
            if (not args.no_auto_click) and (not paused):
                # If we're in a lockout window, don't accumulate stability; wait for frames to catch up.
                if time.time() < float(action_lockout_until):
                    stable_hits = 0
                    can_click_now = False
                else:
                    # Pick the active button based on flow_state.
                    active_val = float(confirm_val)

                    doing_active = (combat_task is not None) and (not combat_task.done())
                    script_wait_active = (post_start_script_at_ts is not None) and float(post_start_script_at_ts) > 0.0
                    if doing_active or script_wait_active:
                        # While the script is pending/running, do not let run_cycle override the state.
                        flow_state = "DOING"

                    if bool(awaiting_challenge):
                        # After the script finishes, only wait for Challenge Again.
                        flow_state = "END"

                    if (not doing_active) and (not script_wait_active) and (not bool(awaiting_challenge)) and args.flow == "run_cycle":
                        # Prefer explicit matches over previous state.
                        if float(end_val) >= float(args.state_detect_min):
                            flow_state = "END"
                        elif float(combat_val) >= float(args.state_detect_min):
                            flow_state = "COMBAT"
                        elif float(start_val) >= float(args.state_detect_min):
                            flow_state = "START"
                        elif float(confirm_val) >= float(args.state_detect_min):
                            flow_state = "CONFIRM"

                    if flow_state == "CONFIRM":
                        active_val = float(confirm_val)
                    elif flow_state == "START":
                        active_val = float(start_val)
                    elif flow_state == "END":
                        # Safety: never treat a right-side match as "Challenge Again".
                        chal_center_ndi = None
                        try:
                            th_ch, tw_ch = scaled_templates["challenge_btn"].shape[:2]
                            chal_center_ndi = _template_center_ndi(
                                chal_loc,
                                tw=int(tw_ch),
                                th=int(th_ch),
                                scale=float(scale),
                                offset_x=int(CLICK_OFFSET_X),
                                offset_y=int(CLICK_OFFSET_Y),
                            )
                        except Exception:
                            chal_center_ndi = None

                        safe_chal = _challenge_is_safe(
                            chal_val=float(chal_val),
                            chal_center_ndi=chal_center_ndi,
                            v_xres=int(v.xres),
                            max_x_frac=float(args.challenge_max_x_frac),
                            exit_val=float(exit_val),
                            exit_margin=float(args.challenge_exit_margin),
                        )
                        if not safe_chal and float(chal_val) >= float(args.challenge_click_min):
                            # Don't let stability build; cool off briefly.
                            stable_hits = 0
                            action_lockout_until = max(float(action_lockout_until), time.time() + float(args.challenge_unsafe_lockout))
                            print(
                                f"[SAFE] Suppressing challenge click: chal={float(chal_val):.2f} exit={float(exit_val):.2f} "
                                f"center={chal_center_ndi} max_x_frac={float(args.challenge_max_x_frac):.2f}"
                            )
                            active_val = 0.0
                        else:
                            active_val = float(chal_val)

                    click_min = float(args.click_min)
                    if flow_state == "END":
                        click_min = float(args.challenge_click_min)
                    if float(active_val) >= click_min:
                        stable_hits += 1
                    else:
                        stable_hits = 0

                    now = time.time()
                    can_click_now = (
                        (now >= float(action_lockout_until))
                        and ((now - float(last_auto_click_ts)) >= float(args.click_cooldown))
                    )
            else:
                stable_hits = 0
                can_click_now = False

            cursor_ready = True
            if args.require_cursor_for_auto:
                cursor_fresh = (last_cursor_ndi is not None) and ((time.time() - float(last_cursor_ts)) < 0.35)
                cursor_ready = cursor_fresh and (cursor_action_stable_hits >= int(args.cursor_action_stable_frames))

            required_stable_frames = int(args.click_stable_frames)
            if flow_state == "END":
                required_stable_frames = int(args.challenge_click_stable_frames)

            should_act = (
                (not args.no_auto_click)
                and (not paused)
                and stable_hits >= int(required_stable_frames)
                and can_click_now
                and cursor_ready
                and (pending_verify_template is None)
            )

            # In fixed-start mode, we intentionally do NOT click start_btn by template.
            # The START click should happen only via the fixed coordinate pipeline.
            if fixed_start_mode is not None and flow_state == "START":
                should_act = False

            # Flow gating.
            if args.flow == "confirm_to_start" and flow_state != "CONFIRM":
                should_act = False
            if args.flow == "run_cycle" and flow_state == "COMBAT":
                # In-combat: no clicking; wait for end screen.
                should_act = False
            if flow_state == "DOING":
                should_act = False

            if should_act:
                if flow_state == "CONFIRM":
                    action_template = "confirm_btn"
                    action_score = float(confirm_val)
                    verify_template = "start_btn"
                elif flow_state == "START":
                    action_template = "start_btn"
                    action_score = float(start_val)
                    verify_template = "combat_ui"
                elif flow_state == "END":
                    action_template = "challenge_btn"
                    action_score = float(chal_val)
                    if post_challenge_enabled and fixed_start_mode is not None:
                        # We'll do a fixed post-challenge move+click instead of template-verifying confirm.
                        verify_template = None
                    else:
                        verify_template = "confirm_btn" if fixed_start_mode is not None else "start_btn"
                else:
                    action_template = "confirm_btn"
                    action_score = float(confirm_val)
                    verify_template = "start_btn"

                if action_template == "challenge_btn":
                    awaiting_challenge = False

                print(f"[ACTION] {action_template} ({action_score:.2f}) -> MOVE then CLICK (scripted)")
                telemetry.on_action()
                last_auto_click_ts = time.time()
                action_lockout_until = float(last_auto_click_ts) + float(args.action_cooldown)
                stable_hits = 0

                # draw green rectangle on display
                if render_now:
                    h, w = scaled_templates["confirm_btn"].shape[:2]
                    cv2.rectangle(display, max_loc, (max_loc[0] + w, max_loc[1] + h), (0, 255, 0), 2)
                    cv2.imshow("Bot View", display)
                    last_display_ts = time.time()
                    # Pump GUI so overlays actually appear before we run a long click routine.
                    _cv_poll_key(1)

                # Compute target center in NDI pixels for the active action template.
                if action_template == "confirm_btn":
                    loc = confirm_loc
                    th, tw = scaled_templates["confirm_btn"].shape[:2]
                elif action_template == "start_btn":
                    loc = start_loc
                    th, tw = scaled_templates["start_btn"].shape[:2]
                else:
                    loc = chal_loc
                    th, tw = scaled_templates["challenge_btn"].shape[:2]

                center_ndi = _template_center_ndi(
                    loc,
                    tw=int(tw),
                    th=int(th),
                    scale=float(scale),
                    offset_x=int(CLICK_OFFSET_X),
                    offset_y=int(CLICK_OFFSET_Y),
                )
                if center_ndi is None:
                    # Shouldn't happen, but don't click blindly.
                    print(f"[SAFE] No center computed for template={action_template}; skipping action")
                    stable_hits = 0
                    action_lockout_until = max(float(action_lockout_until), time.time() + 0.35)
                    continue

                orig_x, orig_y = int(center_ndi[0]), int(center_ndi[1])

                # Re-check safety right before clicking Challenge Again.
                if action_template == "challenge_btn":
                    safe_now = _challenge_is_safe(
                        chal_val=float(chal_val),
                        chal_center_ndi=(orig_x, orig_y),
                        v_xres=int(v.xres),
                        max_x_frac=float(args.challenge_max_x_frac),
                        exit_val=float(exit_val),
                        exit_margin=float(args.challenge_exit_margin),
                    )
                    if not safe_now:
                        print(
                            f"[SAFE] Skipping challenge click at {orig_x},{orig_y}: chal={float(chal_val):.2f} exit={float(exit_val):.2f}"
                        )
                        stable_hits = 0
                        action_lockout_until = max(float(action_lockout_until), time.time() + float(args.challenge_unsafe_lockout))
                        continue

                if last_cursor_ndi is not None and (time.time() - float(last_cursor_ts)) < 1.0:
                    ax = int(last_cursor_ndi[0])
                    ay = int(last_cursor_ndi[1])
                    bx = int(orig_x)
                    by = int(orig_y)
                    print(f"[ACTION] A(cursor)=({ax},{ay}) B(target)=({bx},{by}) B-A=({bx-ax},{by-ay})")

                # Target overlay already updated each frame from template matching.

                try:
                    # Scripted behavior: do exactly what you do manually.
                    # A = last_cursor_ndi (already includes cursor_offset), B = target center.
                    # Send one km.move(B-A), then click immediately. Verification happens by frame templates.
                    cursor_fresh = (last_cursor_ndi is not None) and ((time.time() - float(last_cursor_ts)) < 0.60)
                    if not cursor_fresh:
                        print('[FLOW] Cursor not fresh; recovering to top-left and waiting')
                        _makcu_transport_send(makcu, 'km.move(-3000,-3000)')
                        action_lockout_until = max(float(action_lockout_until), time.time() + 0.35)
                        ok = False
                    else:
                        ok = await _scripted_move_then_click(
                            makcu,
                            cursor_ndi=last_cursor_ndi,
                            target_ndi=(orig_x, orig_y),
                            v_xres=int(v.xres),
                            v_yres=int(v.yres),
                            km_calibration=km_calibration,
                            max_step_counts=int(args.cursor_one_shot_max_step),
                            settle_s=0.18,
                        )
                        # Immediately assume cursor is now at the clicked target to avoid waiting
                        # for remote cursor re-detection before the next action.
                        if ok:
                            _assume_cursor_at((orig_x, orig_y))
                            # Fixed-start pipeline: after confirm click, immediately move+click known START.
                            if fixed_start_mode is not None and flow_state == "CONFIRM":
                                pending_fixed_start_ts = time.time() + float(args.fixed_start_delay)
                                # Don’t keep targeting confirm while we wait for the fixed START click.
                                flow_state = "START"
                            # After Challenge Again click, optionally do a fixed follow-up move+click.
                            if action_template == "challenge_btn" and post_challenge_enabled and fixed_start_mode is not None:
                                pending_post_challenge_ts = time.time() + float(args.post_challenge_delay)
                                flow_state = "DOING"
                    if ds_logger.enabled:
                        ds_logger.log_action(
                            action="click",
                            template="confirm_btn",
                            ndi_target=(int(orig_x), int(orig_y)),
                            ndi_res=(int(v.xres), int(v.yres)),
                            ok=bool(ok),
                            frame_bgr=frame_bgr,
                        )
                except Exception as e:
                    if ds_logger.enabled:
                        ds_logger.log_action(
                            action="click",
                            template="confirm_btn",
                            ndi_target=(int(orig_x), int(orig_y)),
                            ndi_res=(int(v.xres), int(v.yres)),
                            ok=False,
                            error=str(e),
                            frame_bgr=frame_bgr,
                        )

                # Post-click verification (non-blocking): watch the latest frames until the next UI appears.
                if args.flow in ("confirm_to_start", "run_cycle"):
                    # If fixed-start mode is active, skip verifying start_btn (we'll click START by coordinate).
                    if fixed_start_mode is not None and verify_template == "start_btn":
                        verify_template = None

                    if verify_template is None:
                        pass
                    else:
                        pending_verify_template = str(verify_template)
                        pending_verify_min = float(args.post_click_verify_min)
                        pending_verify_deadline_ts = time.time() + float(args.post_click_verify_timeout)
                        if verify_template == "start_btn":
                            pending_verify_success_state = "START"
                        elif verify_template == "combat_ui":
                            pending_verify_success_state = "COMBAT"
                        elif verify_template == "confirm_btn":
                            pending_verify_success_state = "CONFIRM"
                        else:
                            pending_verify_success_state = None

                        # On failure, keep/start over from CONFIRM (same behavior as before).
                        pending_verify_fail_state = "START" if flow_state == "START" else "CONFIRM"
                        # Keep only a short lockout; verification gating blocks actions.
                        action_lockout_until = max(float(action_lockout_until), time.time() + 0.10)
            else:
                if render_now and display is not None:
                    cv2.imshow("Bot View", display)
                    last_display_ts = time.time()
        else:
            # no frame this iteration; fall through to pump GUI events
            pass

        telemetry.on_frame(ok=ok_frame, loop_s=(time.perf_counter() - loop_t0), capture_s=capture_s, match_s=match_s)
        snap = telemetry.maybe_report()
        if snap is not None:
            fps = (snap.frames_ok / max(1e-6, snap.window_s))
            print(
                f"[TEL] fps={fps:.1f} ok={snap.frames_ok} drop={snap.frames_drop} actions={snap.actions} "
                f"loop={snap.avg_loop_ms:.1f}ms cap={snap.avg_capture_ms:.1f}ms match={snap.avg_match_ms:.1f}ms"
            )

        # Always process GUI events and allow closing the window (unless fast-mode skips rendering)
        if args.fast_mode:
            key = -1
        else:
            key = _cv_poll_key(1)
        if key == ord(' ') :
            paused = not paused
            print(f"[MODE] paused={paused}")
            if paused and combat_task is not None and not combat_task.done():
                combat_cancel.set()
        if key == ord('q') or key == 27:
            break
        if key == ord('c') or key == ord('C'):
            if last_cursor_ndi is not None and (time.time() - last_cursor_ts) < 1.0:
                print(f"[CURSOR] cursorNDI=({int(last_cursor_ndi[0])},{int(last_cursor_ndi[1])}) score={last_cursor_score:.3f}")
            else:
                print("[CURSOR] cursor unknown / stale")
        if key == ord('d') or key == ord('D'):
            if (
                last_cursor_ndi is not None
                and (time.time() - last_cursor_ts) < 1.0
                and last_target_ndi is not None
                and (time.time() - last_target_ts) < 1.0
            ):
                # NOTE: last_cursor_ndi already includes cursor_offset_x/y (applied at detection time).
                ax_ndi = int(last_cursor_ndi[0])
                ay_ndi = int(last_cursor_ndi[1])
                bx_ndi = int(last_target_ndi[0])
                by_ndi = int(last_target_ndi[1])
                dx_ndi = int(bx_ndi - ax_ndi)
                dy_ndi = int(by_ndi - ay_ndi)
                km_dx, km_dy = mh.map_ndidelta_to_km(dx_ndi, dy_ndi, int(v.xres), int(v.yres), km_calibration=km_calibration)
                print(
                    f"[DELTA] A=({ax_ndi},{ay_ndi}) B=({bx_ndi},{by_ndi}) B-A ndi=({dx_ndi},{dy_ndi}) km.move({int(km_dx)},{int(km_dy)})"
                )
            else:
                print('[DELTA] Need fresh cursor + target to compute B-A')
        if key == ord('k') or key == ord('K'):
            # One-shot test: km.move(B-A) then immediate click.
            if (
                last_cursor_ndi is not None
                and (time.time() - last_cursor_ts) < 1.0
                and last_target_ndi is not None
                and (time.time() - last_target_ts) < 1.0
            ):
                # NOTE: last_cursor_ndi already includes cursor_offset_x/y (applied at detection time).
                ax_ndi = int(last_cursor_ndi[0])
                ay_ndi = int(last_cursor_ndi[1])
                bx_ndi = int(last_target_ndi[0])
                by_ndi = int(last_target_ndi[1])
                dx_ndi = int(bx_ndi - ax_ndi)
                dy_ndi = int(by_ndi - ay_ndi)
                km_dx, km_dy = mh.map_ndidelta_to_km(dx_ndi, dy_ndi, int(v.xres), int(v.yres), km_calibration=km_calibration)
                cmd = f"km.move({int(km_dx)},{int(km_dy)})"
                ok = _makcu_transport_send(makcu, cmd)
                print(
                    f"[SEND] A=({ax_ndi},{ay_ndi}) B=({bx_ndi},{by_ndi}) B-A ndi=({dx_ndi},{dy_ndi}) {cmd} sent={ok}"
                )

                # Give NDI a moment to catch up, then click.
                await asyncio.sleep(0.08)
                ok_click = await mh.makcu_left_click(makcu, hold_s=0.03)
                print(f"[SEND] post-move left click sent={ok_click}")
            else:
                print('[SEND] Need fresh cursor + target to send B-A')
        if key == ord('l') or key == ord('L'):
            ok = await mh.makcu_left_click(makcu, hold_s=0.03)
            print(f"[SEND] left click sent={ok}")
        if key == ord('s') or key == ord('S'):
            # Snapshot-based calibration: press 's' before and after typed km.move.
            try:
                if 'frame_bgr' not in locals() or frame_bgr is None:
                    print('[CAL] No frame available to snapshot yet')
                elif last_cursor_ndi is None or (time.time() - last_cursor_ts) >= 1.0:
                    print('[CAL] Cursor not detected (or stale); snapshot saved but no measurement')
                    calib_idx += 1
                    ts_ms = int(time.time() * 1000)
                    save_path = os.path.join(calib_dir, f"snap_{calib_idx:03d}_{ts_ms}.png")
                    cv2.imwrite(save_path, frame_bgr)
                    print(f"[CAL] Saved snapshot -> {save_path}")
                else:
                    calib_idx += 1
                    ts_ms = int(time.time() * 1000)
                    save_path = os.path.join(calib_dir, f"snap_{calib_idx:03d}_{ts_ms}.png")
                    cv2.imwrite(save_path, frame_bgr)
                    cur = (int(last_cursor_ndi[0]), int(last_cursor_ndi[1]))
                    print(f"[CAL] Saved snapshot -> {save_path} cursorNDI={cur} score={last_cursor_score:.3f}")

                    if calib_pending_move is not None and calib_pre_cursor is not None:
                        dx_cnt, dy_cnt = int(calib_pending_move[0]), int(calib_pending_move[1])
                        dx_px = int(cur[0] - calib_pre_cursor[0])
                        dy_px = int(cur[1] - calib_pre_cursor[1])
                        print(f"[CAL] Observed delta_px=({dx_px},{dy_px}) for km.move({dx_cnt},{dy_cnt})")

                        if dx_cnt != 0 and dx_px != 0:
                            calib_samples_x.append(abs(float(dx_px) / float(dx_cnt)))
                            calib_sign_samples_x.append(1 if (dx_px * dx_cnt) > 0 else -1)
                        if dy_cnt != 0 and dy_px != 0:
                            calib_samples_y.append(abs(float(dy_px) / float(dy_cnt)))
                            calib_sign_samples_y.append(1 if (dy_px * dy_cnt) > 0 else -1)

                        if calib_samples_x:
                            mx = float(statistics.median(calib_samples_x))
                        else:
                            mx = None
                        if calib_samples_y:
                            my = float(statistics.median(calib_samples_y))
                        else:
                            my = None

                        print(f"[CAL] median px_per_count: x={mx} y={my} samples: x={len(calib_samples_x)} y={len(calib_samples_y)}")

                        # Update in-memory calibration so cursor controller uses it immediately.
                        if km_calibration is None:
                            km_calibration = {}
                        if mx is not None:
                            km_calibration['px_per_count_x'] = mx
                        if my is not None:
                            km_calibration['px_per_count_y'] = my
                        if calib_sign_samples_x:
                            km_calibration['sign_x'] = 1 if (sum(calib_sign_samples_x) >= 0) else -1
                        if calib_sign_samples_y:
                            km_calibration['sign_y'] = 1 if (sum(calib_sign_samples_y) >= 0) else -1

                        calib_pending_move = None

                    # Arm next measurement.
                    calib_pre_cursor = cur
            except Exception as e:
                print(f"[CAL] Snapshot/calibration failed: {e}")
        if key == ord('w') or key == ord('W'):
            # Write current median calibration to makcu_calibration.json
            try:
                if not calib_samples_x and not calib_samples_y and not km_calibration:
                    print('[CAL] No calibration samples to write yet')
                else:
                    out_path = os.path.join(MAKCUCOMMANDS_DIR, 'makcu_calibration.json')
                    payload = {}
                    if km_calibration:
                        payload.update(km_calibration)
                    if calib_samples_x:
                        payload['px_per_count_x'] = float(statistics.median(calib_samples_x))
                    if calib_samples_y:
                        payload['px_per_count_y'] = float(statistics.median(calib_samples_y))
                    if calib_sign_samples_x:
                        payload['sign_x'] = 1 if (sum(calib_sign_samples_x) >= 0) else -1
                    if calib_sign_samples_y:
                        payload['sign_y'] = 1 if (sum(calib_sign_samples_y) >= 0) else -1
                    payload['source'] = 'testing.recovered.py snapshot calibration'
                    payload['ts'] = time.strftime('%Y-%m-%d %H:%M:%S')
                    payload['samples_x'] = int(len(calib_samples_x))
                    payload['samples_y'] = int(len(calib_samples_y))
                    with open(out_path, 'w', encoding='utf-8') as f:
                        json.dump(payload, f, ensure_ascii=False, indent=2)
                    print(f"[CAL] Wrote calibration -> {out_path}: {payload}")
            except Exception as e:
                print(f"[CAL] Write calibration failed: {e}")
        if key == ord('p') or key == ord('P'):
            # Debug: detect cursor in the CURRENT raw frame and print what we see.
            try:
                if 'frame_bgr' in locals() and frame_bgr is not None and cursor_template_gray:
                    score, cx, cy = mh.detect_cursor_in_frame(frame_bgr, cursor_template_gray, match_min=args.cursor_min)
                    if cx is None:
                        print(f"[CURSOR DBG] not found (score={score:.3f} < {args.cursor_min})")
                    else:
                        print(f"[CURSOR DBG] score={score:.3f} cursorNDI=({cx},{cy})")

                        # draw marker on preview display
                        disp_h, disp_w = display.shape[:2]
                        px = int(cx / float(v.xres) * disp_w)
                        py = int(cy / float(v.yres) * disp_h)
                        dbg = display.copy()
                        cv2.circle(dbg, (px, py), 10, (0, 0, 255), 2)
                        cv2.putText(dbg, f"cursor {score:.2f}", (max(5, px+12), max(20, py-12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                        cv2.imshow("Bot View", dbg)
                else:
                    print('[CURSOR DBG] no frame or no cursor templates loaded')
            except Exception as e:
                print(f"[CURSOR DBG] failed: {e}")
        if key == ord('t') or key == ord('T'):
            # Capture a cursor template by letting the user draw a box around the cursor in the preview.
            try:
                if 'display' not in locals() or display is None:
                    print('[WARN] No frame available yet for template capture')
                    continue
                roi = cv2.selectROI("Bot View", display, fromCenter=False, showCrosshair=True)
                rx, ry, rw, rh = [int(v) for v in roi]
                if rw > 2 and rh > 2:
                    # Map ROI in preview -> ROI in original frame
                    disp_h, disp_w = display.shape[:2]
                    ox = int(rx / float(disp_w) * v.xres)
                    oy = int(ry / float(disp_h) * v.yres)
                    ow = int(rw / float(disp_w) * v.xres)
                    oh = int(rh / float(disp_h) * v.yres)
                    ox = max(0, min(ox, v.xres - 1))
                    oy = max(0, min(oy, v.yres - 1))
                    ow = max(1, min(ow, v.xres - ox))
                    oh = max(1, min(oh, v.yres - oy))
                    crop = frame_bgr[oy:oy + oh, ox:ox + ow]
                    # Save as cursor_templateN.png so we can keep multiple shapes.
                    folder = cursor_folder
                    existing = sorted(glob.glob(os.path.join(folder, 'cursor_template*.png')))
                    next_idx = len(existing) + 1
                    save_path = os.path.join(folder, f'cursor_template{next_idx}.png')
                    cv2.imwrite(save_path, crop)
                    cursor_template_gray = load_cursor_templates_gray(cursor_template_path)
                    if cursor_template_gray:
                        print(f"[INIT] Saved cursor template -> {save_path} ({ow}x{oh})")
                    else:
                        print('[WARN] Saved cursor template but failed to reload it')
            except Exception as e:
                print(f"[WARN] Cursor template capture failed: {e}")
        # detect window close (user pressed the X button)
        try:
            if cv2.getWindowProperty("Bot View", cv2.WND_PROP_VISIBLE) < 1:
                print("[INFO] Bot View window closed by user")
                break
        except Exception:
            pass

        # IMPORTANT: yield to asyncio so background tasks (combat_task) can run.
        # When frames are flowing, this loop can hog CPU and delay task wakeups.
        try:
            doing_active = (combat_task is not None) and (not combat_task.done())
            script_wait_active = float(post_start_script_at_ts) > 0.0
            extra = float(args.doing_yield_s) if (doing_active or script_wait_active) else 0.0
        except Exception:
            extra = 0.0
        await asyncio.sleep(max(0.0, float(extra)))

        if terminal_control_enabled:
            # Process any terminal commands typed by the user.
            # Example: km.move(10,0) then press 's' after the cursor moves.
            while True:
                try:
                    line = cmd_queue.get_nowait()
                except Exception:
                    break

                if not line:
                    continue
                low = line.strip().lower()
                if low in ('help', '?'):
                    print("[TERM] Examples: km.move(10,0) ; km.left(1) ; km.left(0) ; km.wheel(-120)")
                    continue

                parsed = _parse_km_move(line)
                if parsed is not None:
                    dx_cnt, dy_cnt = int(parsed[0]), int(parsed[1])
                    cmd = f"km.move({dx_cnt},{dy_cnt})"
                    if _makcu_transport_send(makcu, cmd):
                        print(f"[TERM] Sent {cmd}. After cursor moves, press 's' to snapshot.")
                        if dx_cnt != 0 or dy_cnt != 0:
                            calib_pending_move = (dx_cnt, dy_cnt)
                    else:
                        print(f"[TERM] Failed to send {cmd} (no connected transport)")
                    continue

                # Allow raw km.* commands
                if low.startswith('km.'):
                    if _makcu_transport_send(makcu, line.strip()):
                        print(f"[TERM] Sent {line.strip()}")
                    else:
                        print(f"[TERM] Failed to send: {line.strip()} (no connected transport)")
                    continue

                print(f"[TERM] Unrecognized command: {line}")
        # process any user clicks (only when enabled and when we have valid frame/display geometry)
        if bool(args.enable_manual_click_target) and ok_frame and display is not None:
            while click_events:
                cx, cy = click_events.pop(0)
                last_target_preview = (int(cx), int(cy))
                # compute mapping from preview coords to original NDI pixels
                disp_h, disp_w = display.shape[:2]
                orig_x = int(cx / float(disp_w) * v.xres)
                orig_y = int(cy / float(disp_h) * v.yres)
                orig_x += CLICK_OFFSET_X
                orig_y += CLICK_OFFSET_Y
                print(f"[USER] Click at preview ({cx},{cy}) -> NDI ({orig_x},{orig_y})")
                last_target_ndi = (int(orig_x), int(orig_y))

                # In paused / measurement mode: NEVER move/click; only update the target.
                if paused or args.no_auto_click:
                    print('[MODE] Measurement mode: target updated (no movement).')
                    continue

                # Keep manual clicks consistent with the main automation (no extra NDI capture loops).
                cursor_fresh = (last_cursor_ndi is not None) and ((time.time() - float(last_cursor_ts)) < 0.60)
                if not cursor_fresh:
                    print('[FLOW] Cursor not fresh; recovering to top-left and waiting')
                    _makcu_transport_send(makcu, 'km.move(-3000,-3000)')
                    action_lockout_until = max(float(action_lockout_until), time.time() + 0.25)
                    continue

                await _scripted_move_then_click(
                    makcu,
                    cursor_ndi=last_cursor_ndi,
                    target_ndi=(orig_x, orig_y),
                    v_xres=int(v.xres),
                    v_yres=int(v.yres),
                    km_calibration=km_calibration,
                    max_step_counts=int(args.cursor_one_shot_max_step),
                    settle_s=0.18,
                )
                _assume_cursor_at((orig_x, orig_y))
                await asyncio.sleep(0.10)

        # small sleep to avoid tight-spin when we don't have fresh frames
        if not ok_frame:
            await asyncio.sleep(0.01)

    try:
        cmd_stop.set()
        try:
            if 'frame_src' in locals() and frame_src is not None:
                frame_src.stop()
        except Exception:
            pass
        if 'ndi_recv' in locals() and ndi_recv is not None:
            ndi.recv_destroy(ndi_recv)
    except Exception:
        pass
    try:
        # only call destroy if initialize succeeded
        if 'ndi_inited' in locals() and ndi_inited:
            ndi.destroy()
    except Exception:
        pass
    try:
        ds_logger.close()
    except Exception:
        pass
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass


if __name__ == "__main__":
    asyncio.run(main())
