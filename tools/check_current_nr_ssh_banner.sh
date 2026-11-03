#!/usr/bin/env bash
set -euo pipefail
/root/autodl-tmp/PV-Ground/env/bin/python - <<'PY'
import datetime
import json
import socket

host, port = "connect.nmb1.seetacloud.com", 32749
print(json.dumps({
    "observed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "endpoint": f"{host}:{port}",
    "scope": "SSH banner only from the Scan server; no authentication or training access.",
}), flush=True)
with socket.create_connection((host, port), timeout=20) as connection:
    banner = connection.recv(256).decode("ascii", errors="replace").strip()
    assert banner.startswith("SSH-")
    print(json.dumps({"ssh_banner": banner, "reachable": True}))
PY
