Makcu RPC Agent
================

Purpose
-------
Lightweight HTTP agent to run on the machine with the Makcu HID attached. It exposes simple endpoints to move the cursor and click.

Quick start (local test, no hardware)
-----------------------------------
1. Start the agent in no-hardware mode (uses DummyMakcu):

```powershell
python makcu_agent.py --host 127.0.0.1 --port 8080 --no-hw
```

2. In another shell, send a move:

```powershell
python makcu_client.py move --x 1000 --y 500
```

3. Or click:

```powershell
python makcu_client.py click --button left
```

Deployment
----------
- Run `makcu_agent.py` on the PC that has the Makcu device attached (omit `--no-hw`).
- From the remote PC, use `makcu_client.py` or any HTTP client to POST to `/move` and `/click`.

Security
--------
This agent is intentionally minimal. If exposing on a network, place it behind a firewall or add an API key and TLS.
