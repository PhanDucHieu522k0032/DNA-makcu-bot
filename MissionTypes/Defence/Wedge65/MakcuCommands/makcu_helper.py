import asyncio
import math
import time
import json
import os
import cv2
import numpy as np
import NDIlib as ndi
import ctypes

# Configuration defaults (can be overridden by caller)
REAL_MONITOR_WIDTH = 1920
REAL_MONITOR_HEIGHT = 1080


def map_ndidelta_to_screen(delta_x, delta_y, frame_w, frame_h, monitor_w=REAL_MONITOR_WIDTH, monitor_h=REAL_MONITOR_HEIGHT):
    """Map delta in NDI pixels to delta in screen pixels/counts."""
    sx = int(delta_x * (monitor_w / max(1, frame_w)))
    sy = int(delta_y * (monitor_h / max(1, frame_h)))
    return sx, sy


def load_km_calibration(path):
    """Load km.move calibration from JSON.

    Expected keys:
      - px_per_count_x (float)
      - px_per_count_y (float)
      - sign_x (int, +1 or -1)
      - sign_y (int, +1 or -1)
    """
    if not path:
        return None
    try:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        px_x = float(data.get("px_per_count_x"))
        px_y = float(data.get("px_per_count_y"))
        sx = int(data.get("sign_x", 1))
        sy = int(data.get("sign_y", 1))
        if px_x <= 0 or px_y <= 0:
            return None
        sx = 1 if sx >= 0 else -1
        sy = 1 if sy >= 0 else -1
        return {
            "px_per_count_x": px_x,
            "px_per_count_y": px_y,
            "sign_x": sx,
            "sign_y": sy,
        }
    except Exception:
        return None


def map_ndidelta_to_km(delta_x, delta_y, frame_w, frame_h, *, km_calibration=None, monitor_w=REAL_MONITOR_WIDTH, monitor_h=REAL_MONITOR_HEIGHT):
    """Map desired delta in NDI pixels to km.move counts.

    If km_calibration is present, use measured pixels-per-count and sign.
    Otherwise fall back to monitor scaling (approximate).
    """
    if km_calibration:
        px_x = float(km_calibration.get("px_per_count_x", 0.0))
        px_y = float(km_calibration.get("px_per_count_y", 0.0))
        sx = int(km_calibration.get("sign_x", 1))
        sy = int(km_calibration.get("sign_y", 1))
        if px_x > 0 and px_y > 0:
            km_dx = int(round(float(delta_x) / px_x)) * (1 if sx >= 0 else -1)
            km_dy = int(round(float(delta_y) / px_y)) * (1 if sy >= 0 else -1)
            return km_dx, km_dy

    # Fallback: treat km counts approximately as screen pixels.
    return map_ndidelta_to_screen(delta_x, delta_y, frame_w, frame_h, monitor_w=monitor_w, monitor_h=monitor_h)


def map_ndipx_to_screen(ndi_x, ndi_y, frame_w, frame_h, monitor_w=REAL_MONITOR_WIDTH, monitor_h=REAL_MONITOR_HEIGHT):
    """Map NDI pixel coordinates to monitor screen pixels."""
    sx = int(ndi_x * (monitor_w / max(1, frame_w)))
    sy = int(ndi_y * (monitor_h / max(1, frame_h)))
    return sx, sy


def get_cursor_pos():
    class POINT(ctypes.Structure):
        _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]
    pt = POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
    return pt.x, pt.y


async def safe_move_abs(makcu, screen_x, screen_y, speed=6, wait_ms=4, settle=0.05):
    """Call makcu.move_abs and print before/after cursor position for diagnostics.
    This wrapper is asynchronous and will sleep briefly to allow smoothing.
    """
    try:
        before = get_cursor_pos()
    except Exception:
        before = (None, None)
    print(f"[HELPER] cursor before move_abs: {before}")
    # Do NOT call makcu.move_abs (it blocks waiting for local cursor to reach target).
    # Use move_smooth (non-blocking) if available, otherwise send one bounded raw km.move via transport.
    try:
        local_dx = int(screen_x - (before[0] if before[0] is not None else 0))
        local_dy = int(screen_y - (before[1] if before[1] is not None else 0))

        if hasattr(makcu, 'move_smooth'):
            segs = max(6, int(max(1, math.hypot(local_dx, local_dy) / 12)))
            try:
                await makcu.move_smooth(local_dx, local_dy, segments=segs)
                print(f"[HELPER] sent move_smooth delta ({local_dx},{local_dy}) segs={segs}")
            except Exception as e:
                print(f"[HELPER] move_smooth failed: {e}")
        else:
            transport = getattr(makcu, 'transport', None)
            if transport and transport.is_connected():
                # send a single km.move with the full delta (device will attempt to execute)
                try:
                    transport.send_command(f"km.move({local_dx},{local_dy})", expect_response=False)
                    print(f"[HELPER] transport sent km.move({local_dx},{local_dy})")
                except Exception as e:
                    print(f"[HELPER] transport send failed: {e}")
            else:
                print('[HELPER] No move method available on makcu')

    except Exception as e:
        print(f"[HELPER] Exception preparing move: {e}")

    await asyncio.sleep(settle)

    # Poll local cursor a few times to detect movement (helps detect remote HID)
    after = None
    polls = int(max(1, min(20, int(1.0 / max(0.01, settle)))))
    for i in range(polls):
        try:
            after = get_cursor_pos()
        except Exception:
            after = (None, None)
        if after != before:
            break
        await asyncio.sleep(0.05)

    print(f"[HELPER] cursor after move_abs: {after}")
    if after == before:
        print('[HELPER] WARNING: local cursor unchanged — Makcu may be attached to a different host')
        # If move didn't affect local cursor, try a bounded transport small-step correction
        try:
            print('[HELPER] attempting bounded transport correction')
            ok = send_small_steps_via_transport(makcu, screen_x, screen_y, step_px=8, delay=0.04, max_steps=80)
            print(f'[HELPER] transport correction result: {ok}')
        except Exception as e:
            print(f'[HELPER] transport correction exception: {e}')


async def vision_feedback_click(makcu, desired_ndix, desired_ndiy, template_name, templates_gray, ndi_recv, v_xres, v_yres, max_attempts=20, pixel_tolerance=12, speed=6):
    """Closed-loop: move cursor to desired NDI pixel, verify visually, click when within tolerance.
    Uses the provided `ndi_recv` handle to pull fresh frames for verification.
    """
    screen_x, screen_y = map_ndipx_to_screen(desired_ndix, desired_ndiy, v_xres, v_yres)
    print(f"[HELPER VISION] Desired NDI ({desired_ndix},{desired_ndiy}) -> screen ({screen_x},{screen_y})")

    attempts = 0
    last_obs = None
    stuck_count = 0
    # iterative correction loop using vision feedback
    while attempts < max_attempts:
        attempts += 1
        await safe_move_abs(makcu, screen_x, screen_y, speed=speed)

        # grab a fresh frame
        t, v, _, _ = ndi.recv_capture_v2(ndi_recv, 250)
        if t != ndi.FRAME_TYPE_VIDEO:
            await asyncio.sleep(0.03)
            continue

        frame_data = np.copy(v.data)
        ndi.recv_free_video_v2(ndi_recv, v)
        try:
            frame_data = frame_data.reshape((v.yres, v.xres, 4))
            frame_bgr = np.ascontiguousarray(frame_data[:, :, :3])
        except ValueError:
            print("[HELPER VISION] Frame reshape failed")
            await asyncio.sleep(0.02)
            continue

        gray_full = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        templ = templates_gray.get(template_name)
        if templ is None:
            print(f"[HELPER VISION] Template {template_name} missing")
            return False

        res = cv2.matchTemplate(gray_full, templ, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        th, tw = templ.shape[:2]
        obs_x = max_loc[0] + (tw // 2)
        obs_y = max_loc[1] + (th // 2)

        error = math.hypot(obs_x - desired_ndix, obs_y - desired_ndiy)
        print(f"[HELPER VISION] Attempt {attempts}: match={max_val:.3f} obsNDI=({obs_x},{obs_y}) error={error:.1f}px")

        if max_val > 0.45 and error <= pixel_tolerance:
            print("[HELPER VISION] Target acquired — clicking")
            await makcu.click(makcu_mouse_button())
            return True

        # compute correction in NDI pixels and map to screen delta
        delta_ndix = desired_ndix - obs_x
        delta_ndiy = desired_ndiy - obs_y
        if abs(delta_ndix) < 1 and abs(delta_ndiy) < 1:
            print('[HELPER VISION] negligible delta, clicking')
            await makcu.click(makcu_mouse_button())
            return True

        # map NDI delta to screen delta
        corr_sx, corr_sy = map_ndipx_to_screen(delta_ndix, delta_ndiy, v.xres, v.yres)

        # Conservative per-iteration cap to avoid overshoot (pixels)
        MAX_PER_ITER = 120
        corr_sx = int(_clip(corr_sx, -MAX_PER_ITER, MAX_PER_ITER))
        corr_sy = int(_clip(corr_sy, -MAX_PER_ITER, MAX_PER_ITER))

        # detect if observed position is not changing between iterations
        if last_obs is not None and (obs_x, obs_y) == last_obs:
            stuck_count += 1
        else:
            stuck_count = 0
        last_obs = (obs_x, obs_y)

        # If the observed NDI location is stuck across iterations, try aggressive transport fallback
        if stuck_count >= 2:
            print(f"[HELPER VISION] Observed NDI position unchanged for {stuck_count} attempts — using aggressive transport steps")
            try:
                _aggressive_transport_steps(makcu, corr_sx, corr_sy, step_px=8, batches=40, delay=0.03)
            except Exception as e:
                print(f"[HELPER VISION] aggressive transport fallback failed: {e}")
            # after aggressive attempts, continue loop to re-capture frame
            await asyncio.sleep(0.06)
            continue

        # Use relative smooth move if available
        try:
            if hasattr(makcu, 'move_smooth'):
                # segments scale with distance to make movement smooth, but bounded
                segs = int(max(6, min(48, math.hypot(corr_sx, corr_sy) / 6)))
                print(f"[HELPER VISION] Applying smooth correction ({corr_sx},{corr_sy}) segments={segs}")
                await makcu.move_smooth(int(corr_sx), int(corr_sy), segments=segs)
            else:
                # fallback: send raw transport moves
                transport = getattr(makcu, 'transport', None)
                if transport and transport.is_connected():
                    print(f"[HELPER VISION] transport fallback move ({corr_sx},{corr_sy})")
                    transport.send_command(f"km.move({int(corr_sx)},{int(corr_sy)})", expect_response=False)
                else:
                    print('[HELPER VISION] No transport fallback available')
        except Exception as e:
            print(f"[HELPER VISION] correction move failed: {e}")

        await asyncio.sleep(0.05)

    print("[HELPER VISION] Failed to acquire target within attempts")
    return False


def makcu_mouse_button():
    # default left button enum is provided by caller context; import locally to avoid circular import
    from makcu import MouseButton
    return MouseButton.LEFT


async def makcu_left_click(makcu, *, hold_s: float = 0.02) -> bool:
    """Left click using raw Makcu commands when possible.

    Uses:
    - km.left(1) then km.left(0)
    Fallback:
    - makcu.click(MouseButton.LEFT)
    """
    try:
        transport = getattr(makcu, 'transport', None)
        if transport and transport.is_connected():
            transport.send_command('km.left(1)', expect_response=False)
            await asyncio.sleep(float(hold_s))
            transport.send_command('km.left(0)', expect_response=False)
            return True
    except Exception:
        pass

    try:
        if hasattr(makcu, 'click'):
            await makcu.click(makcu_mouse_button())
            return True
    except Exception:
        return False

    return False


def _clip(v, lo, hi):
    return max(lo, min(hi, v))


def _capture_frame_bgr(ndi_recv, timeout_ms=250):
    """Capture one NDI video frame and return (frame_bgr, xres, yres) or (None,0,0)."""
    t, v, _, _ = ndi.recv_capture_v2(ndi_recv, timeout_ms)
    if t != ndi.FRAME_TYPE_VIDEO:
        return None, 0, 0
    frame_data = np.copy(v.data)
    xres = int(v.xres)
    yres = int(v.yres)
    ndi.recv_free_video_v2(ndi_recv, v)
    try:
        frame_data = frame_data.reshape((yres, xres, 4))
        frame_bgr = np.ascontiguousarray(frame_data[:, :, :3])
    except Exception:
        return None, 0, 0
    return frame_bgr, xres, yres


def detect_cursor_in_frame(frame_bgr, cursor_templates_gray, *, match_min=0.55, gray_min=None, agree_px=12):
    """Detect cursor position in a raw BGR frame using one or more templates.

    Returns (score, center_x, center_y) in raw frame pixel coordinates.
    """
    if frame_bgr is None:
        return 0.0, None, None

    if cursor_templates_gray is None:
        return 0.0, None, None

    if isinstance(cursor_templates_gray, (list, tuple)):
        templates = [t for t in cursor_templates_gray if t is not None]
    else:
        templates = [cursor_templates_gray]

    if not templates:
        return 0.0, None, None

    if gray_min is None:
        gray_min = float(match_min) - 0.10

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)

    best_score = 0.0
    best_cx = None
    best_cy = None

    for templ in templates:
        templ_gray = templ
        templ_edges = cv2.Canny(templ_gray, 60, 160)
        if edges.shape[0] < templ_edges.shape[0] or edges.shape[1] < templ_edges.shape[1]:
            continue

        # Match on edges and grayscale; require that their best locations agree.
        se, cxe, cye = _match_center(edges, templ_edges)
        if cxe is None:
            continue
        if float(se) < float(match_min):
            continue

        sg, cxg, cyg = _match_center(gray, templ_gray)
        if cxg is None:
            continue
        if float(sg) < float(gray_min):
            continue

        if abs(int(cxg) - int(cxe)) > int(agree_px) or abs(int(cyg) - int(cye)) > int(agree_px):
            continue

        combined = float(se) * 0.65 + float(sg) * 0.35
        if combined > best_score:
            best_score = combined
            best_cx = int(cxe)
            best_cy = int(cye)

    if best_cx is None:
        return float(best_score), None, None

    return float(best_score), int(best_cx), int(best_cy)


def _match_center(img, templ):
    """Return (score, center_x, center_y) in img coordinates.

    Works for both single-channel and multi-channel arrays as long as channels match.
    """
    if img is None or templ is None:
        return 0.0, None, None
    if img.shape[0] < templ.shape[0] or img.shape[1] < templ.shape[1]:
        return 0.0, None, None
    res = cv2.matchTemplate(img, templ, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    th, tw = templ.shape[:2]
    cx = int(max_loc[0] + (tw // 2))
    cy = int(max_loc[1] + (th // 2))
    return float(max_val), cx, cy


async def cursor_feedback_click(
    makcu,
    target_ndix,
    target_ndiy,
    cursor_template_gray,
    ndi_recv,
    v_xres,
    v_yres,
    *,
    initial_cursor_ndix=None,
    initial_cursor_ndiy=None,
    km_calibration=None,
    preview_width=960,
    max_attempts=35,
    pixel_tolerance=10,
    target_hitbox_half_w=None,
    target_hitbox_half_h=None,
    gain=1.0,
    max_step=140,
    cursor_match_min=0.55,
    cursor_offset_x=0,
    cursor_offset_y=0,
    settle_s=0.03,
    safe_margin=30,
    max_lost_after_lock=3,
    recover_to_topleft_on_lost=True,
    recover_move_counts=(-3000, -3000),
    recover_cooldown_s=2.0,
    max_recover_attempts=2,
    require_recent_measurement_for_click=True,
    max_click_measurement_age_s=0.50,
    min_command_interval_s=0.12,
    one_shot=False,
    post_move_click=True,
    post_move_delay_s=0.10,
    post_move_remeasure_attempts=4,
    post_move_remeasure_interval_s=0.08,
    debug_out=None,
):
    """Closed-loop click using *cursor tracking* on the NDI stream.

    This is the correct approach when Makcu is physically attached to a different PC than this Python process.
    We detect the cursor position in the NDI frames, compute (dx,dy) to the target in pixels, and send relative
    `km.move` (or `move_smooth`) commands until within tolerance, then click.
    """

    # Allow a single template or a list/tuple of templates (for arrow/hand/etc).
    if cursor_template_gray is None:
        print('[HELPER CURSOR] No cursor template provided; cannot do cursor feedback click')
        return False

    if isinstance(cursor_template_gray, (list, tuple)):
        cursor_templates = [t for t in cursor_template_gray if t is not None]
    else:
        cursor_templates = [cursor_template_gray]

    if not cursor_templates:
        print('[HELPER CURSOR] No valid cursor templates; cannot do cursor feedback click')
        return False

    # We receive raw NDI frames. For robust cursor tracking, prefer matching on the raw frame
    # within a small ROI around the predicted cursor position.
    scale = preview_width / float(max(1, v_xres))

    # Build edge templates at raw + preview scale.
    raw_edges_list = []
    small_edges_list = []
    for templ in cursor_templates:
        raw_edges_list.append(cv2.Canny(templ, 60, 160))
        th, tw = templ.shape[:2]
        tw_s = max(1, int(tw * scale))
        th_s = max(1, int(th * scale))
        t_small = cv2.resize(templ, (tw_s, th_s), interpolation=cv2.INTER_AREA)
        small_edges_list.append(cv2.Canny(t_small, 60, 160))

    # Internal cursor estimate in NDI pixels. Vision (template match) corrects this estimate when available.
    est_cursor = None  # (ndix, ndiy)
    last_move = (0, 0)  # last commanded move in NDI px (used for ROI prediction)
    last_dist = None
    stale_count = 0
    last_meas_raw = None  # (cur_ndix, cur_ndiy) without offsets
    lost_after_lock = 0
    last_meas_ts = 0.0
    last_nonzero_cmd = (0, 0)

    last_cmd_sent_ts = 0.0
    last_cmd_sent_meas_raw = None

    last_recover_ts = 0.0
    recover_attempts = 0
    not_found_count = 0

    async def _recover_to_topleft():
        """Best-effort recovery: slam cursor toward top-left so it's visible again."""
        dx_cnt, dy_cnt = int(recover_move_counts[0]), int(recover_move_counts[1])
        try:
            transport = getattr(makcu, 'transport', None)
            if transport and transport.is_connected():
                transport.send_command(f"km.move({dx_cnt},{dy_cnt})", expect_response=False)
                return True
        except Exception:
            pass

        # Fallback to move_smooth if available.
        try:
            if hasattr(makcu, 'move_smooth'):
                segs = 40
                await makcu.move_smooth(dx_cnt, dy_cnt, segments=segs)
                return True
        except Exception:
            pass
        return False

    # Seed estimate from caller (e.g. continuously detected cursor in the main loop).
    if initial_cursor_ndix is not None and initial_cursor_ndiy is not None:
        try:
            est_cursor = (int(initial_cursor_ndix), int(initial_cursor_ndiy))
            last_meas_ts = time.time()
        except Exception:
            est_cursor = None

    def _roi_from_pred_raw(ndix, ndiy, frame_w, frame_h, radius_ndipx=520):
        # Compute ROI bounds directly in raw NDI pixel space.
        px = int(ndix)
        py = int(ndiy)
        r = int(max(32, radius_ndipx))
        x0 = max(0, px - r)
        y0 = max(0, py - r)
        x1 = min(int(frame_w), px + r)
        y1 = min(int(frame_h), py + r)
        return x0, y0, x1, y1

    for attempt in range(1, int(max_attempts) + 1):
        frame_bgr, fx, fy = _capture_frame_bgr(ndi_recv, timeout_ms=250)
        if frame_bgr is None:
            await asyncio.sleep(settle_s)
            continue

        # Keep targets inside a safe box so we never intentionally drive into corners.
        # (If user clicks near edges, we will stop at the nearest safe pixel.)
        margin = int(max(0, safe_margin))
        safe_min_x = margin
        safe_min_y = margin
        safe_max_x = int(max(0, fx - 1 - margin))
        safe_max_y = int(max(0, fy - 1 - margin))
        tgt_x = int(_clip(int(target_ndix), safe_min_x, safe_max_x))
        tgt_y = int(_clip(int(target_ndiy), safe_min_y, safe_max_y))

        # Raw edges for highest-fidelity cursor matching.
        gray_raw = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        edges_raw = cv2.Canny(gray_raw, 60, 160)

        # Prefer raw ROI search around predicted cursor location to avoid false positives and blur.
        cur_score = 0.0
        cur_ndix = None
        cur_ndiy = None

        # First acquisition: if user clicked a target and the cursor is already near it,
        # search *near the target first* to avoid latching onto a random static pattern elsewhere.
        if est_cursor is None:
            x0, y0, x1, y1 = _roi_from_pred_raw(tgt_x, tgt_y, fx, fy, radius_ndipx=360)
            roi = edges_raw[y0:y1, x0:x1]
            best0 = (0.0, None, None)
            for templ_edges in raw_edges_list:
                if roi.shape[0] < templ_edges.shape[0] or roi.shape[1] < templ_edges.shape[1]:
                    continue
                s0, cx0, cy0 = _match_center(roi, templ_edges)
                if cx0 is not None and s0 > best0[0]:
                    best0 = (float(s0), int(cx0), int(cy0))
            if best0[1] is not None:
                cur_score = float(best0[0])
                cur_ndix = int(best0[1] + x0)
                cur_ndiy = int(best0[2] + y0)

        if est_cursor is not None:
            pred_ndix = int(est_cursor[0] + last_move[0])
            pred_ndiy = int(est_cursor[1] + last_move[1])
            x0, y0, x1, y1 = _roi_from_pred_raw(pred_ndix, pred_ndiy, fx, fy)
            roi = edges_raw[y0:y1, x0:x1]
            # Evaluate all templates and take best.
            best = (0.0, None, None)
            for templ_edges in raw_edges_list:
                if roi.shape[0] < templ_edges.shape[0] or roi.shape[1] < templ_edges.shape[1]:
                    continue
                s, cx, cy = _match_center(roi, templ_edges)
                if cx is not None and s > best[0]:
                    best = (float(s), int(cx), int(cy))
            if best[1] is not None:
                cur_score = float(best[0])
                cur_ndix = int(best[1] + x0)
                cur_ndiy = int(best[2] + y0)

        # If we don't have an estimate yet, do a coarse preview-space acquisition,
        # then refine in raw around that coarse location.
        if est_cursor is None and (cur_ndix is None or cur_score < float(cursor_match_min)):
            small_h = int(preview_width * fy / max(1, fx))
            small = cv2.resize(frame_bgr, (preview_width, max(1, small_h)))
            gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            edges_small = cv2.Canny(gray_small, 60, 160)
            best2 = (0.0, None, None)
            for templ_edges_s in small_edges_list:
                if edges_small.shape[0] < templ_edges_s.shape[0] or edges_small.shape[1] < templ_edges_s.shape[1]:
                    continue
                s2, cx_s, cy_s = _match_center(edges_small, templ_edges_s)
                if cx_s is not None and s2 > best2[0]:
                    best2 = (float(s2), int(cx_s), int(cy_s))

            if best2[1] is not None:
                coarse_ndix = int(best2[1] / max(1e-6, scale))
                coarse_ndiy = int(best2[2] / max(1e-6, scale))
                # refine around coarse estimate in raw using best raw template match
                x0, y0, x1, y1 = _roi_from_pred_raw(coarse_ndix, coarse_ndiy, fx, fy, radius_ndipx=420)
                roi = edges_raw[y0:y1, x0:x1]
                best3 = (0.0, None, None)
                for templ_edges in raw_edges_list:
                    if roi.shape[0] < templ_edges.shape[0] or roi.shape[1] < templ_edges.shape[1]:
                        continue
                    s3, cx3, cy3 = _match_center(roi, templ_edges)
                    if cx3 is not None and s3 > best3[0]:
                        best3 = (float(s3), int(cx3), int(cy3))

                if best3[1] is not None:
                    cur_score = float(best3[0])
                    cur_ndix = int(best3[1] + x0)
                    cur_ndiy = int(best3[2] + y0)
                else:
                    cur_score = float(best2[0])
                    cur_ndix = coarse_ndix
                    cur_ndiy = coarse_ndiy
        have_meas = (cur_ndix is not None and cur_score >= float(cursor_match_min))

        if not have_meas:
            not_found_count += 1
            if debug_out is not None:
                debug_out.update({"cursor_score": float(cur_score), "cursor_ndix": None, "cursor_ndiy": None})
            if est_cursor is None:
                print(f"[HELPER CURSOR] Attempt {attempt}: cursor not found (score={cur_score:.3f} < {cursor_match_min})")
                # If we cannot find the cursor at all for several attempts, force recovery.
                now = time.time()
                can_recover = (
                    bool(recover_to_topleft_on_lost)
                    and (not_found_count >= int(max_lost_after_lock))
                    and (recover_attempts < int(max_recover_attempts))
                    and (now - last_recover_ts >= float(recover_cooldown_s))
                )
                if can_recover:
                    print(f"[HELPER CURSOR] Cursor not found for {not_found_count} attempts; recovering with km.move{tuple(recover_move_counts)}")
                    ok = await _recover_to_topleft()
                    last_recover_ts = now
                    recover_attempts += 1
                    not_found_count = 0
                    est_cursor = (int(safe_min_x), int(safe_min_y))
                    last_move = (0, 0)
                    stale_count = 0
                    if not ok:
                        print('[HELPER CURSOR] Recovery command failed; aborting')
                        return False
                    await asyncio.sleep(float(settle_s) + 0.10)
                    continue

                await asyncio.sleep(settle_s)
                continue
            # We have an estimate; keep moving toward target using the estimate, but stop after too many misses.
            lost_after_lock += 1
            print(
                f"[HELPER CURSOR] Attempt {attempt}: cursor not detected; using estimate estNDI=({est_cursor[0]},{est_cursor[1]}) "
                f"lost_count={lost_after_lock}/{max_lost_after_lock}"
            )
            if lost_after_lock >= int(max_lost_after_lock):
                now = time.time()
                can_recover = bool(recover_to_topleft_on_lost) and (recover_attempts < int(max_recover_attempts)) and (now - last_recover_ts >= float(recover_cooldown_s))
                if can_recover:
                    print(f"[HELPER CURSOR] Cursor lost too long; recovering with km.move{tuple(recover_move_counts)}")
                    ok = await _recover_to_topleft()
                    last_recover_ts = now
                    recover_attempts += 1
                    lost_after_lock = 0
                    stale_count = 0
                    last_move = (0, 0)

                    # Bias estimate to safe top-left after the slam.
                    est_cursor = (int(safe_min_x), int(safe_min_y))
                    if not ok:
                        print('[HELPER CURSOR] Recovery command failed; aborting')
                        return False

                    await asyncio.sleep(float(settle_s) + 0.08)
                    continue

                return False
        else:
            not_found_count = 0
            lost_after_lock = 0
            # Successful measurement: allow future recoveries.
            recover_attempts = 0

            # Reject wild jumps when we already have an estimate.
            if est_cursor is not None:
                pred_ndix = int(est_cursor[0] + last_move[0])
                pred_ndiy = int(est_cursor[1] + last_move[1])
                jump = math.hypot(cur_ndix - pred_ndix, cur_ndiy - pred_ndiy)
                if jump > 900 and cur_score < float(cursor_match_min) + 0.15:
                    print(
                        f"[HELPER CURSOR] Attempt {attempt}: rejected jump (jump={jump:.1f}px score={cur_score:.3f}); using estimate"
                    )
                    have_meas = False
                else:
                    # Accept measurement and correct estimate.
                    est_cursor = (int(cur_ndix) + int(cursor_offset_x), int(cur_ndiy) + int(cursor_offset_y))
                    last_meas_ts = time.time()

            if est_cursor is None:
                est_cursor = (int(cur_ndix) + int(cursor_offset_x), int(cur_ndiy) + int(cursor_offset_y))
                last_meas_ts = time.time()

            if debug_out is not None:
                debug_out.update({"cursor_score": float(cur_score), "cursor_ndix": int(cur_ndix), "cursor_ndiy": int(cur_ndiy)})

        # At this point we must have an estimate.
        if est_cursor is None:
            return False

        # If the detected cursor is outside safe bounds, treat it as invalid (likely false match).
        # IMPORTANT: After we have a track, do not "re-acquire globally" (that can jump to corners).
        est_x, est_y = int(est_cursor[0]), int(est_cursor[1])
        if est_x < safe_min_x or est_x > safe_max_x or est_y < safe_min_y or est_y > safe_max_y:
            # Clamp estimate back into safe box (prevents runaway if estimate drifts).
            est_x = int(_clip(est_x, safe_min_x, safe_max_x))
            est_y = int(_clip(est_y, safe_min_y, safe_max_y))
            est_cursor = (est_x, est_y)

        err_x = float(tgt_x - est_x)
        err_y = float(tgt_y - est_y)
        dist = math.hypot(err_x, err_y)
        meas_age = (time.time() - last_meas_ts) if last_meas_ts > 0 else float('inf')
        meas_tag = "meas" if have_meas else f"est-only age={meas_age:.2f}s"
        print(
            f"[HELPER CURSOR] Attempt {attempt}: score={cur_score:.3f} [{meas_tag}] estNDI=({est_x},{est_y}) tgtNDI=({tgt_x},{tgt_y}) "
            f"err=({err_x:.0f},{err_y:.0f}) dist={dist:.1f}px"
        )

        # Critical safety: never click based on estimate-only state.
        # If the estimate claims we're already (very) close but we can't actually see the cursor,
        # force a re-acquisition instead of declaring success.
        if (not have_meas) and dist <= (float(pixel_tolerance) + 2.5):
            print('[HELPER CURSOR] Near target by estimate only; forcing cursor re-acquisition (no click)')
            est_cursor = None
            last_move = (0, 0)
            stale_count = 0
            await asyncio.sleep(settle_s)
            continue

        # If the caller provides a target hitbox (half extents in pixels), prefer that.
        # This avoids needing the cursor to reach the exact center.
        if have_meas and target_hitbox_half_w is not None and target_hitbox_half_h is not None:
            try:
                hw = float(target_hitbox_half_w)
                hh = float(target_hitbox_half_h)
                if abs(err_x) <= hw and abs(err_y) <= hh:
                    print('[HELPER CURSOR] Inside target hitbox — clicking')
                    await makcu.click(makcu_mouse_button())
                    return True
            except Exception:
                pass

        # Stale detection (hardened): only consider it "stale" if the *raw* cursor measurement
        # is literally unchanged across iterations while we are issuing non-zero moves and
        # the distance is not improving. This avoids false aborts when close to the target
        # or when using cursor offsets.
        if have_meas and cur_ndix is not None and cur_ndiy is not None:
            cur_raw = (int(cur_ndix), int(cur_ndiy))
            cmd_mag = math.hypot(float(last_nonzero_cmd[0]), float(last_nonzero_cmd[1]))
            if last_meas_raw is not None and cur_raw == last_meas_raw and cmd_mag >= 3.0:
                if last_dist is not None and dist >= (last_dist - 1.0):
                    stale_count += 1
                else:
                    stale_count = 0
            else:
                stale_count = 0
            last_meas_raw = cur_raw
        else:
            stale_count = 0

        last_dist = dist

        # Latency guard: if we just sent a move, but the NDI measurement hasn't updated yet,
        # do NOT send another move based on the same cursor position.
        if have_meas and cur_ndix is not None and cur_ndiy is not None and last_cmd_sent_meas_raw is not None:
            cur_raw = (int(cur_ndix), int(cur_ndiy))
            if cur_raw == last_cmd_sent_meas_raw and (time.time() - float(last_cmd_sent_ts)) < float(min_command_interval_s):
                print('[HELPER CURSOR] Measurement not updated yet; waiting (latency guard)')
                await asyncio.sleep(max(float(settle_s), 0.03))
                continue

        # Only abort if we're clearly not near the target and truly stuck.
        if stale_count >= 3 and dist > max(float(pixel_tolerance) * 3.0, 45.0):
            print(
                f"[HELPER CURSOR] Cursor match appears stale after move commands (last_cmd={last_nonzero_cmd}); aborting to avoid runaway"
            )
            return False

        # Deadband: if we're extremely close and we have a measurement, prefer click over tiny corrective moves.
        close_enough = dist <= (float(pixel_tolerance) + 2.5)
        if close_enough and have_meas:
            print('[HELPER CURSOR] Close enough (deadband) — clicking')
            await makcu_left_click(makcu)
            return True

        if dist <= float(pixel_tolerance):
            # Never click based on estimate-only state.
            if not have_meas:
                print('[HELPER CURSOR] Within tolerance by estimate only, but cursor not measured — not clicking')
            elif require_recent_measurement_for_click and meas_age > float(max_click_measurement_age_s):
                print('[HELPER CURSOR] Within tolerance, but cursor measurement too old — not clicking')
            else:
                print('[HELPER CURSOR] Within tolerance — clicking')
                await makcu_left_click(makcu)
                return True

        # If we're near the safe box edge and cursor tracking is currently lost, pull toward center first.
        center_x = int((safe_min_x + safe_max_x) / 2)
        center_y = int((safe_min_y + safe_max_y) / 2)
        near_edge = (est_x <= safe_min_x + 4) or (est_x >= safe_max_x - 4) or (est_y <= safe_min_y + 4) or (est_y >= safe_max_y - 4)
        if (not have_meas) and near_edge:
            pull_x = float(center_x - est_x)
            pull_y = float(center_y - est_y)
            # Use a conservative pull to re-enter a trackable region.
            move_ndix = int(_clip(int(round(pull_x * 0.5)), -max_step, max_step))
            move_ndiy = int(_clip(int(round(pull_y * 0.5)), -max_step, max_step))
            if move_ndix == 0 and move_ndiy == 0:
                move_ndix = int(_clip(center_x - est_x, -max_step, max_step))
                move_ndiy = int(_clip(center_y - est_y, -max_step, max_step))
            print(f"[HELPER CURSOR] Tracking lost near edge; pulling toward center move_ndi=({move_ndix},{move_ndiy})")
        else:
            # Apply proportional correction and clamp.
            eff_gain = float(gain) * (0.65 if not have_meas else 1.0)
            move_ndix = int(_clip(int(round(err_x * eff_gain)), -max_step, max_step))
            move_ndiy = int(_clip(int(round(err_y * eff_gain)), -max_step, max_step))

        # Clamp predicted next cursor position to stay inside safe box.
        pred_x = int(est_x + move_ndix)
        pred_y = int(est_y + move_ndiy)
        if pred_x < safe_min_x:
            move_ndix += int(safe_min_x - pred_x)
        elif pred_x > safe_max_x:
            move_ndix -= int(pred_x - safe_max_x)
        if pred_y < safe_min_y:
            move_ndiy += int(safe_min_y - pred_y)
        elif pred_y > safe_max_y:
            move_ndiy -= int(pred_y - safe_max_y)

        move_ndix = int(_clip(move_ndix, -max_step, max_step))
        move_ndiy = int(_clip(move_ndiy, -max_step, max_step))

        last_move = (int(move_ndix), int(move_ndiy))

        # Integrate move into estimate (open-loop update). Vision corrections will snap it back later.
        est_cursor = (int(_clip(est_x + move_ndix, safe_min_x, safe_max_x)), int(_clip(est_y + move_ndiy, safe_min_y, safe_max_y)))

        # Convert NDI delta -> km.move counts.
        move_sx, move_sy = map_ndidelta_to_km(move_ndix, move_ndiy, fx, fy, km_calibration=km_calibration)
        move_sx = int(_clip(move_sx, -max_step, max_step))
        move_sy = int(_clip(move_sy, -max_step, max_step))

        print(f"[HELPER CURSOR] move_ndi=({move_ndix},{move_ndiy}) move_cmd=({move_sx},{move_sy})")

        # IMPORTANT: keep the internal estimate consistent with what we actually send.
        # When km calibration is present, rounding can make move_cmd != move_ndi, and
        # updating the estimate with move_ndi causes drift/overshoot when vision is lost.
        if km_calibration:
            try:
                px_x = float(km_calibration.get('px_per_count_x', 0.0))
                px_y = float(km_calibration.get('px_per_count_y', 0.0))
                sx = 1 if int(km_calibration.get('sign_x', 1)) >= 0 else -1
                sy = 1 if int(km_calibration.get('sign_y', 1)) >= 0 else -1
                if px_x > 0 and px_y > 0:
                    # Invert the sign application done in map_ndidelta_to_km.
                    cmd_ndi_x = int(round(int(move_sx) * px_x * sx))
                    cmd_ndi_y = int(round(int(move_sy) * px_y * sy))
                    est_cursor = (
                        int(_clip(est_x + cmd_ndi_x, safe_min_x, safe_max_x)),
                        int(_clip(est_y + cmd_ndi_y, safe_min_y, safe_max_y)),
                    )
            except Exception:
                pass

        # Send relative move.
        try:
            transport = getattr(makcu, 'transport', None)
            if transport and transport.is_connected():
                transport.send_command(f"km.move({int(move_sx)},{int(move_sy)})", expect_response=False)
            elif hasattr(makcu, 'move_smooth'):
                segs = int(max(6, min(40, math.hypot(move_sx, move_sy) / 6)))
                await makcu.move_smooth(int(move_sx), int(move_sy), segments=segs)
            else:
                print('[HELPER CURSOR] No move method available on makcu')
                return False
        except Exception as e:
            print(f"[HELPER CURSOR] move failed: {e}")
            return False

        # One-shot mode: exactly one move command, then click.
        # (We still require a real cursor measurement to avoid clicking based on an estimate.)
        if one_shot:
            if not have_meas:
                print('[HELPER CURSOR] one_shot requested but cursor not measured — refusing to click')
                return False
            await asyncio.sleep(float(post_move_delay_s))

            if not post_move_click:
                return True

            # Safety: re-measure cursor after the move and only click if we are truly inside the hitbox.
            # Under NDI latency, the first frame after a move can still show the old cursor position.
            cx2 = cy2 = None
            score2 = 0.0
            for i in range(1, int(max(1, post_move_remeasure_attempts)) + 1):
                frame2_bgr, _, _ = _capture_frame_bgr(ndi_recv, timeout_ms=250)
                if frame2_bgr is None:
                    await asyncio.sleep(float(post_move_remeasure_interval_s))
                    continue

                score2, cx2, cy2 = detect_cursor_in_frame(frame2_bgr, cursor_templates, match_min=float(cursor_match_min))
                if cx2 is not None:
                    break
                await asyncio.sleep(float(post_move_remeasure_interval_s))

            if cx2 is None:
                print(f"[HELPER CURSOR] one_shot: cursor not detected after move (score={float(score2):.3f}) — not clicking")
                return False

            cur2_x = int(cx2) + int(cursor_offset_x)
            cur2_y = int(cy2) + int(cursor_offset_y)
            # Use original safe-clamped target.
            err2_x = float(tgt_x - cur2_x)
            err2_y = float(tgt_y - cur2_y)
            dist2 = math.hypot(err2_x, err2_y)
            print(
                f"[HELPER CURSOR] one_shot post-move: score={float(score2):.3f} curNDI=({cur2_x},{cur2_y}) "
                f"tgtNDI=({tgt_x},{tgt_y}) err=({err2_x:.0f},{err2_y:.0f}) dist={dist2:.1f}px"
            )

            inside = False
            if target_hitbox_half_w is not None and target_hitbox_half_h is not None:
                try:
                    inside = (abs(err2_x) <= float(target_hitbox_half_w)) and (abs(err2_y) <= float(target_hitbox_half_h))
                except Exception:
                    inside = False
            else:
                inside = dist2 <= float(pixel_tolerance)

            if not inside:
                print('[HELPER CURSOR] one_shot: not inside hitbox after move — not clicking')
                return False

            print('[HELPER CURSOR] one_shot: inside hitbox — clicking')
            ok_click = await makcu_left_click(makcu)
            return bool(ok_click)

        if have_meas and cur_ndix is not None and cur_ndiy is not None and (move_sx != 0 or move_sy != 0):
            last_cmd_sent_ts = time.time()
            last_cmd_sent_meas_raw = (int(cur_ndix), int(cur_ndiy))

        if move_sx != 0 or move_sy != 0:
            last_nonzero_cmd = (int(move_sx), int(move_sy))

        await asyncio.sleep(settle_s)

    print('[HELPER CURSOR] Failed to reach target within attempts')
    return False


def send_small_steps_via_transport(makcu, target_x, target_y, step_px=8, delay=0.02, max_steps=500):
    """Fallback: send small raw `km.move(dx,dy)` commands via the makcu.transport.
    Use when `move_abs` didn't visibly move the local cursor. Works only if `makcu.transport` exists and is connected.
    Returns True if local cursor reached (or commands were sent), False otherwise.
    """
    try:
        transport = getattr(makcu, 'transport', None)
        if transport is None:
            print('[HELPER] No transport available on makcu object')
            return False
        if not transport.is_connected():
            print('[HELPER] Transport not connected')
            return False

        # poll local cursor and send relative steps until near target or max_steps
        def _get():
            try:
                return get_cursor_pos()
            except Exception:
                return None

        steps = 0
        while steps < max_steps:
            cur = _get()
            if cur is None:
                break
            dx = target_x - cur[0]
            dy = target_y - cur[1]
            if abs(dx) <= 1 and abs(dy) <= 1:
                return True

            mvx = int(math.copysign(min(step_px, abs(dx)), dx)) if dx != 0 else 0
            mvy = int(math.copysign(min(step_px, abs(dy)), dy)) if dy != 0 else 0

            # clip to safe range
            mvx = _clip(mvx, -32767, 32767)
            mvy = _clip(mvy, -32767, 32767)

            try:
                transport.send_command(f"km.move({mvx},{mvy})", expect_response=False)
            except Exception as e:
                print(f"[HELPER] transport.send_command failed: {e}")
                return False

            time.sleep(delay)
            steps += 1

        print('[HELPER] send_small_steps_via_transport finished (max_steps or no movement)')
        return False
    except Exception as e:
        print('[HELPER] Exception in send_small_steps_via_transport:', e)
        return False


def _aggressive_transport_steps(makcu, corr_sx, corr_sy, step_px=8, batches=40, delay=0.03):
    """Send repeated small raw km.move steps in the requested direction without relying on local cursor polling.
    This is used when vision shows the observed target is not changing between iterations.
    """
    transport = getattr(makcu, 'transport', None)
    if transport is None or not transport.is_connected():
        print('[HELPER VISION] No transport available for aggressive steps')
        return False

    # Determine per-step x/y signs and remaining magnitude
    rem_x = corr_sx
    rem_y = corr_sy
    for b in range(batches):
        if abs(rem_x) <= 0 and abs(rem_y) <= 0:
            return True
        mvx = int(math.copysign(min(step_px, abs(rem_x)), rem_x)) if rem_x != 0 else 0
        mvy = int(math.copysign(min(step_px, abs(rem_y)), rem_y)) if rem_y != 0 else 0
        # send the small step
        try:
            transport.send_command(f"km.move({mvx},{mvy})", expect_response=False)
        except Exception as e:
            print(f"[HELPER VISION] aggressive transport send failed: {e}")
            return False
        rem_x -= mvx
        rem_y -= mvy
        time.sleep(delay)
    return True
