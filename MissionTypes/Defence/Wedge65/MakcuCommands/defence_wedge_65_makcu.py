"""Makcu script for Defence Wedge 65.

This module is intended to be called from `testing.recovered.py` once `combat_ui` is detected.

You can edit `run_combat()` to implement the in-combat "doing" sequence using raw Makcu
transport commands (e.g. `km.move(dx,dy)`, `km.left(1)`, `km.left(0)`, etc.).

Expected integration style:
- `await run_combat(makcu)` should perform the combat actions and return.
"""

from __future__ import annotations

import argparse
import asyncio
from typing import Optional

import makcu_helper as mh


async def _sleep_s(seconds: float) -> None:
    if seconds and seconds > 0:
        await asyncio.sleep(float(seconds))


async def run_combat(
    makcu,
    *,
    cancel_event: Optional[asyncio.Event] = None,
) -> None:
    """Run the in-combat Makcu sequence.

    Args:
        makcu: Connected Makcu controller instance (same object used by `testing.recovered.py`).
        cancel_event: Optional asyncio.Event; if set, stop early.

    Notes:
        - Keep this routine non-blocking (use `await asyncio.sleep(...)` between actions).
        - Prefer sending commands through the controller helpers (if available) or transport.
    """

    print("[DOING] combat script start")

    # Example of a raw movement command (relative):
    # NOTE: Makcu uses km.move(dx,dy) (there is no km.moveto).
    try:
        transport = getattr(makcu, 'transport', None)
        if transport is not None:
            transport.send_command("km.move(-200,0)", expect_response=False)
    except Exception:
        pass

    # Tunables for scripting.
    hold_s = 1.0
    repeats = 3
    # Gap between holds. Use 0 for "as fast as possible"; keep a tiny gap if the game needs it.
    gap_s = 0.0

    for i in range(int(repeats)):
        if cancel_event and cancel_event.is_set():
            print("[DOING] combat script cancelled")
            return

        ok = await mh.makcu_left_click(makcu, hold_s=float(hold_s))
        print(f"[DOING] charged attack {i+1}/{int(repeats)} hold={float(hold_s):.2f}s ok={ok}")
        await _sleep_s(float(gap_s))
    
    try:
        transport = getattr(makcu, 'transport', None)
        if transport is not None:
            transport.send_command("km.move(-308,0)", expect_response=False)
    except Exception:
        pass

    try:
        transport = getattr(makcu, 'transport', None)
        if transport is not None:
            transport.send_command("km.move(0,100)", expect_response=False)
    except Exception:
        pass

    # Tunables for scripting.
    hold_s = 1.0
    repeats = 5
    # Gap between holds. Use 0 for "as fast as possible"; keep a tiny gap if the game needs it.
    gap_s = 0.0

    for i in range(int(repeats)):
        if cancel_event and cancel_event.is_set():
            print("[DOING] combat script cancelled")
            return

        ok = await mh.makcu_left_click(makcu, hold_s=float(hold_s))
        print(f"[DOING] charged attack {i+1}/{int(repeats)} hold={float(hold_s):.2f}s ok={ok}")
        await _sleep_s(float(gap_s))

    try:
        transport = getattr(makcu, 'transport', None)
        if transport is not None:
            transport.send_command("km.move(0,-270)", expect_response=False)
    except Exception:
        pass

    hold_s = 1.0
    repeats = 2
    # Gap between holds. Use 0 for "as fast as possible"; keep a tiny gap if the game needs it.
    gap_s = 0.0

    for i in range(int(repeats)):
        if cancel_event and cancel_event.is_set():
            print("[DOING] combat script cancelled")
            return

        ok = await mh.makcu_left_click(makcu, hold_s=float(hold_s))
        print(f"[DOING] charged attack {i+1}/{int(repeats)} hold={float(hold_s):.2f}s ok={ok}")
        await _sleep_s(float(gap_s))

    try:
        transport = getattr(makcu, 'transport', None)
        if transport is not None:
            transport.send_command("km.move(0,-50)", expect_response=False)
    except Exception:
        pass

    hold_s = 1.0
    repeats = 1
    # Gap between holds. Use 0 for "as fast as possible"; keep a tiny gap if the game needs it.
    gap_s = 0.0

    for i in range(int(repeats)):
        if cancel_event and cancel_event.is_set():
            print("[DOING] combat script cancelled")
            return

        ok = await mh.makcu_left_click(makcu, hold_s=float(hold_s))
        print(f"[DOING] charged attack {i+1}/{int(repeats)} hold={float(hold_s):.2f}s ok={ok}")
        await _sleep_s(float(gap_s))

    print("[DOING] combat script done")


__all__ = ["run_combat"]


async def _main() -> None:
    parser = argparse.ArgumentParser(description="Run defence_wedge_65_makcu script standalone")
    parser.add_argument("--no-hw", action="store_true", help="Don't connect to Makcu; print commands only")
    parser.add_argument("--com-port", type=str, default="COM7", help="Makcu COM port (default: COM7)")
    args = parser.parse_args()

    if args.no_hw:
        class _DummyTransport:
            def is_connected(self):
                return True

            def send_command(self, cmd: str, expect_response: bool = False, timeout: float = 0.1):
                print(f"[DRYRUN] send_command: {cmd}")
                return ""

        class _DummyMakcu:
            def __init__(self):
                self.transport = _DummyTransport()

        makcu = _DummyMakcu()
        await run_combat(makcu)
        return

    # Real hardware
    import sys
    sys.path.insert(0, r"C:\okDNA\makcu-py-lib-main")
    from makcu import create_async_controller

    makcu = await create_async_controller(fallback_com_port=str(args.com_port), auto_reconnect=True)
    await run_combat(makcu)


if __name__ == "__main__":
    asyncio.run(_main())
