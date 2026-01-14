# Cursor ML (No NDI capture)

You said: **no capturing NDI frames for training**.

This folder provides an *offline* training workflow that only uses files you put on disk.

## Reality check (important)
A true "cursor detector" needs examples of the cursor **inside full images** (screenshots/frames) so it learns background variation.

Your current folder `cursor_templates/` contains **cursor-only crops**. That is not enough by itself to train a detector that finds the cursor in a full game frame.

So we provide two options:

### Option A (recommended): You provide screenshots manually
1. Put full-frame screenshots (or cropped HUD regions) in:
   - `cursor_ml/data/backgrounds/`

2. Keep your cursor crops in:
   - `cursor_ml/data/cursor_templates/`
   (we can symlink/copy from your existing `cursor_templates/`)

3. Run dataset synthesis (pastes cursor crops onto screenshots at random positions) to create a labeled YOLO dataset:
   - `python cursor_ml/make_synth_dataset.py --out cursor_ml/out --n 4000`

This **does not capture anything from NDI**. It only reads your on-disk images.

### Option B (not recommended): Train on cursor crops only
With only cursor crops, you can train a *patch classifier* (“does this patch contain cursor?”), but you still need a scanning strategy and it will be slower/less reliable than a detector.

## What gets generated
- `cursor_ml/out/images/train/*.jpg`
- `cursor_ml/out/images/val/*.jpg`
- `cursor_ml/out/labels/train/*.txt` (YOLO format)
- `cursor_ml/out/labels/val/*.txt`
- `cursor_ml/out/dataset.yaml`

## Training (Ultralytics YOLO)
1. Install training deps into your venv:
   - `pip install ultralytics==8.*`

2. Train:
   - `yolo detect train data=cursor_ml/out/dataset.yaml model=yolov8n.pt imgsz=640 epochs=60 batch=16`

3. Export to ONNX:
   - `yolo export model=runs/detect/train/weights/best.pt format=onnx opset=12`

## Next step
If you want the model to be useful in-game, please drop **some actual screenshots** into `cursor_ml/data/backgrounds/` (even 30-100 is enough to start). The synthesis script will do the labeling for you.
