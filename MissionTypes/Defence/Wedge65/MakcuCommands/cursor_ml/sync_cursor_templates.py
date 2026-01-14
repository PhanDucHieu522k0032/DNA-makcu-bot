"""Sync cursor templates into cursor_ml/data/cursor_templates.

No NDI capture. Just copies your existing cursor crops so the training pipeline has a stable input path.

Usage:
  python cursor_ml/sync_cursor_templates.py \
    --src "C:\\okDNA\\DNA-AI\\MissionTypes\\Defence\\Wedge65\\MakcuCommands\\cursor_templates"
"""

import argparse
import shutil
from pathlib import Path

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, required=True, help="Source folder with cursor crops")
    ap.add_argument(
        "--dst",
        type=str,
        default=str(Path(__file__).parent / "data" / "cursor_templates"),
        help="Destination folder (default: cursor_ml/data/cursor_templates)",
    )
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    if not src.exists() or not src.is_dir():
        raise SystemExit(f"Source folder not found: {src}")

    copied = 0
    for p in src.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMG_EXTS:
            continue
        out = dst / p.name
        shutil.copy2(str(p), str(out))
        copied += 1

    print(f"Copied {copied} files -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
