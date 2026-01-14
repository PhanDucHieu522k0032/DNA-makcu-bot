import argparse
import glob
import os
import random
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def _list_images(folder: Path) -> list[Path]:
    out: list[Path] = []
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            out.append(p)
    return sorted(out)


def _imread(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    return img


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _paste_rgba(bg_bgr: np.ndarray, fg_rgba: np.ndarray, x: int, y: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Paste fg_rgba onto bg_bgr at top-left (x,y). Returns (out_img, bbox_xyxy)."""
    out = bg_bgr.copy()

    fg = fg_rgba
    if fg.ndim != 3:
        raise ValueError("fg must be HxWxC")

    if fg.shape[2] == 3:
        # No alpha channel; treat as opaque
        fg_bgr = fg
        alpha = np.ones((fg.shape[0], fg.shape[1]), dtype=np.float32)
    else:
        fg_bgr = fg[:, :, :3]
        alpha = fg[:, :, 3].astype(np.float32) / 255.0

    h, w = fg_bgr.shape[:2]
    H, W = out.shape[:2]

    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(W, x + w)
    y1 = min(H, y + h)

    if x1 <= x0 or y1 <= y0:
        # Off-screen
        return out, (0, 0, 0, 0)

    fx0 = x0 - x
    fy0 = y0 - y
    fx1 = fx0 + (x1 - x0)
    fy1 = fy0 + (y1 - y0)

    roi = out[y0:y1, x0:x1]
    fg_roi = fg_bgr[fy0:fy1, fx0:fx1]
    a_roi = alpha[fy0:fy1, fx0:fx1][:, :, None]

    roi[:] = (fg_roi.astype(np.float32) * a_roi + roi.astype(np.float32) * (1.0 - a_roi)).astype(np.uint8)

    # bbox on background (xyxy)
    return out, (x0, y0, x1, y1)


def _to_yolo_xywh(bbox_xyxy: tuple[int, int, int, int], img_w: int, img_h: int) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = bbox_xyxy
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    cx = x0 + bw / 2.0
    cy = y0 + bh / 2.0
    return (cx / img_w, cy / img_h, bw / img_w, bh / img_h)


def main() -> int:
    ap = argparse.ArgumentParser(description="Make a synthetic cursor detection dataset (no NDI capture).")
    ap.add_argument("--cursor-templates", type=str, default=str(Path(__file__).parent / "data" / "cursor_templates"), help="Folder containing cursor crops/templates")
    ap.add_argument("--backgrounds", type=str, default=str(Path(__file__).parent / "data" / "backgrounds"), help="Folder containing screenshots/background images")
    ap.add_argument("--out", type=str, required=True, help="Output dataset folder")
    ap.add_argument("--n", type=int, default=3000, help="Number of synthetic images to generate")
    ap.add_argument("--imgsz", type=int, default=640, help="Output square image size")
    ap.add_argument("--val-frac", type=float, default=0.15, help="Fraction to put in val split")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    random.seed(int(args.seed))

    cursor_dir = Path(args.cursor_templates)
    bg_dir = Path(args.backgrounds)
    out_dir = Path(args.out)

    cursors = _list_images(cursor_dir)
    bgs = _list_images(bg_dir)

    if not cursors:
        raise SystemExit(f"No cursor templates found in {cursor_dir}")
    if not bgs:
        raise SystemExit(
            f"No background screenshots found in {bg_dir}.\n"
            "Add some screenshots manually (no NDI capture needed)."
        )

    images_train = out_dir / "images" / "train"
    images_val = out_dir / "images" / "val"
    labels_train = out_dir / "labels" / "train"
    labels_val = out_dir / "labels" / "val"

    for p in (images_train, images_val, labels_train, labels_val):
        _ensure_dir(p)

    n = int(args.n)
    imgsz = int(args.imgsz)
    val_frac = float(args.val_frac)

    for i in range(n):
        bg_path = random.choice(bgs)
        cur_path = random.choice(cursors)

        bg = _imread(bg_path)
        fg = _imread(cur_path)
        if bg is None or fg is None:
            continue

        # Normalize to BGR background
        if bg.ndim == 2:
            bg = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
        elif bg.shape[2] == 4:
            bg = bg[:, :, :3]

        # Resize background to square (letterbox-ish via center crop)
        H, W = bg.shape[:2]
        if H <= 0 or W <= 0:
            continue

        # scale smallest side to imgsz, then center crop
        scale = imgsz / float(min(H, W))
        bg2 = cv2.resize(bg, (int(round(W * scale)), int(round(H * scale))), interpolation=cv2.INTER_AREA)
        H2, W2 = bg2.shape[:2]
        x0 = max(0, (W2 - imgsz) // 2)
        y0 = max(0, (H2 - imgsz) // 2)
        bg3 = bg2[y0 : y0 + imgsz, x0 : x0 + imgsz]
        if bg3.shape[0] != imgsz or bg3.shape[1] != imgsz:
            bg3 = cv2.resize(bg3, (imgsz, imgsz), interpolation=cv2.INTER_AREA)

        # Randomly scale cursor
        if fg.ndim == 2:
            fg = cv2.cvtColor(fg, cv2.COLOR_GRAY2BGRA)
        if fg.shape[2] == 3:
            fg = cv2.cvtColor(fg, cv2.COLOR_BGR2BGRA)

        # scale factor tuned to typical cursor sizes; adjust as needed
        s = random.uniform(0.6, 1.6)
        fh, fw = fg.shape[:2]
        fg2 = cv2.resize(fg, (max(4, int(round(fw * s))), max(4, int(round(fh * s)))), interpolation=cv2.INTER_AREA)

        # random position, keep mostly on-screen
        max_x = max(1, imgsz - fg2.shape[1])
        max_y = max(1, imgsz - fg2.shape[0])
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)

        out_img, bbox = _paste_rgba(bg3, fg2, x, y)
        x0b, y0b, x1b, y1b = bbox
        if x1b <= x0b or y1b <= y0b:
            continue

        # Optional: add mild noise/blur to simulate compression
        if random.random() < 0.25:
            out_img = cv2.GaussianBlur(out_img, (3, 3), 0)
        if random.random() < 0.25:
            noise = np.random.normal(0, 3.0, out_img.shape).astype(np.float32)
            out_img = np.clip(out_img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        split_is_val = random.random() < val_frac
        img_name = f"img_{i:06d}.jpg"

        if split_is_val:
            img_out = images_val / img_name
            lbl_out = labels_val / (Path(img_name).stem + ".txt")
        else:
            img_out = images_train / img_name
            lbl_out = labels_train / (Path(img_name).stem + ".txt")

        cv2.imwrite(str(img_out), out_img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

        x, y, w, h = _to_yolo_xywh(bbox, imgsz, imgsz)
        # class 0 = cursor
        lbl_out.write_text(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n", encoding="utf-8")

    # dataset yaml
    yaml_path = out_dir / "dataset.yaml"
    yaml_path.write_text(
        """path: .
train: images/train
val: images/val
names:
  0: cursor
""",
        encoding="utf-8",
    )

    print(f"Wrote dataset to: {out_dir}")
    print(f"- train images: {len(list(images_train.glob('*.jpg')))}")
    print(f"- val images:   {len(list(images_val.glob('*.jpg')))}")
    print(f"dataset.yaml: {yaml_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
