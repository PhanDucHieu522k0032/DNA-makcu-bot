# Bot-control-Makcu

Automation bot that uses:
- NDI video stream for vision (PC2 reads frames)
- Makcu USB HID for mouse control (physically attached to PC1)

## Quickstart
- Install deps: `pip install -r requirements.txt`
- Ensure you have NDI runtime installed and an NDI source running.
- Set Makcu lib path (if needed): `setx MAKCU_PY_LIB "C:\\path\\to\\makcu-py-lib-main"`
- Run: `python main.py`

## Layout
- `main.py`: bot entrypoint
- `MissionTypes/Defence/Wedge65/MakcuCommands/`: Makcu helpers, scripts, cursor templates, and offline training tools
