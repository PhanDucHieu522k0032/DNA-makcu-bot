## Disclaimer
This software is an open-source and free external tool, intended for learning and communication purposes only. It aims to simplify the gameplay of Double Helix through simulation.

- How it works : The bot receive NDI stream from PC1 and then send commands to Makcu to play game on PC1.
- Purpose of use : Intended to provide convenience to users, and not intended to disrupt game balance or provide any unfair advantage.
- Legal responsibility : This project and the development team are not responsible for any problems or consequences arising from the use of this software. The development team reserves the right to the final interpretation of this project.

All issues and consequences arising from the use of this software are not related to this project or its development team. The development team reserves the final right of interpretation for this project. If you encounter vendors using this software for services and charging a fee, this may cover their costs for equipment and time; any resulting problems or consequences are not associated with this software.

## Requirements

- 2 PCs on the same LAN
- Makcu - https://www.makcu.com/en
- NDI and Duet Night Abyss (DNA) installed on PC1 - https://ndi.video/

## DNA-bot-makcu

Automation bot that uses:
- NDI video stream for vision (PC2 reads frames)
- Makcu USB HID for mouse control (physically attached to PC1&2 + mouse)

## Quickstart
- Install deps: `pip install -r requirements.txt`
- Ensure you have NDI runtime installed and an NDI source running.
- Set Makcu lib path (if needed): `setx MAKCU_PY_LIB "C:\\path\\to\\makcu-py-lib-main"`
- Run: `python main.py`

## Layout
- `main.py`: bot entrypoint
- `MissionTypes/Defence/Wedge65/MakcuCommands/`: Makcu helpers, scripts, cursor templates, and offline training tools

## Project status
This project is **under active development**.

Expect breaking changes, incomplete features, and occasional instability while the core loop and detection logic are evolving.

## Known issues / limitations
- **Bot view can be laggy**: the vision loop depends on the NDI stream + decoding + CV processing, which can introduce noticeable latency.
- Detection is currently optimized for specific UI patterns/templates; results can degrade when the game UI, resolution, or background changes significantly.

## Roadmap
- Add more mission types and play modes beyond the current Wedge/Defence flow.
- Improve robustness of UI/cursor detection across different scenes and backgrounds.
- Better configuration + profiles per mode (templates, timings, calibration).

## Demo

https://youtu.be/gapBsAfUgH0
