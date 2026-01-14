import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cv2


@dataclass
class ActionEvent:
    ts: float
    kind: str
    payload: Dict[str, Any]


class DatasetLogger:
    """Very small JSONL + optional frame-dump logger.

    Designed to be safe to leave enabled during runs without affecting behavior.
    """

    def __init__(
        self,
        log_dir: str,
        *,
        enabled: bool = True,
        save_frames: bool = False,
    ):
        self.enabled = enabled and bool(log_dir)
        self.save_frames = save_frames
        self.log_dir = os.path.abspath(log_dir) if log_dir else ""
        self._fp = None
        self._run_id = time.strftime("%Y%m%d-%H%M%S")

        if not self.enabled:
            return

        os.makedirs(self.log_dir, exist_ok=True)
        self._frames_dir = os.path.join(self.log_dir, "frames")
        if self.save_frames:
            os.makedirs(self._frames_dir, exist_ok=True)

        self._jsonl_path = os.path.join(self.log_dir, f"events_{self._run_id}.jsonl")
        self._fp = open(self._jsonl_path, "a", encoding="utf-8")

    def close(self):
        if self._fp is not None:
            try:
                self._fp.flush()
            finally:
                self._fp.close()
            self._fp = None

    def _write(self, event: ActionEvent):
        if not self.enabled or self._fp is None:
            return
        self._fp.write(json.dumps(event.__dict__, ensure_ascii=False) + "\n")
        self._fp.flush()

    def log_match(
        self,
        *,
        template: str,
        match_val: float,
        match_loc: Tuple[int, int],
        ndi_res: Tuple[int, int],
    ):
        self._write(
            ActionEvent(
                ts=time.time(),
                kind="match",
                payload={
                    "template": template,
                    "match_val": float(match_val),
                    "match_loc": [int(match_loc[0]), int(match_loc[1])],
                    "ndi_res": [int(ndi_res[0]), int(ndi_res[1])],
                },
            )
        )

    def log_action(
        self,
        *,
        action: str,
        template: str,
        ndi_target: Tuple[int, int],
        ndi_res: Tuple[int, int],
        ok: Optional[bool] = None,
        error: Optional[str] = None,
        frame_bgr: Any = None,
    ):
        frame_path = None
        if self.enabled and self.save_frames and frame_bgr is not None:
            fname = f"frame_{int(time.time()*1000)}.jpg"
            frame_path = os.path.join("frames", fname)
            abs_path = os.path.join(self.log_dir, frame_path)
            try:
                cv2.imwrite(abs_path, frame_bgr)
            except Exception:
                frame_path = None

        self._write(
            ActionEvent(
                ts=time.time(),
                kind="action",
                payload={
                    "action": action,
                    "template": template,
                    "ndi_target": [int(ndi_target[0]), int(ndi_target[1])],
                    "ndi_res": [int(ndi_res[0]), int(ndi_res[1])],
                    "ok": ok,
                    "error": error,
                    "frame": frame_path,
                },
            )
        )
