"""Is a python from this checkout's venv running? (install.bat step 1)

Runs on the *system* python (the venv may not exist yet). Never kills.
Exit codes: 0 not running, 1 running (listed), 2 could not check.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    venv = Path(sys.argv[1] if len(sys.argv) > 1 else ".venv").resolve()
    if not venv.exists():
        return 0
    ps = (
        "Get-CimInstance Win32_Process | Where-Object { $_.ExecutablePath } | "
        "Select-Object ProcessId, ExecutablePath, CommandLine | ConvertTo-Json -Compress"
    )
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, timeout=60).stdout
        procs = json.loads(out.decode("utf-8", errors="replace") or "[]")
    except (OSError, subprocess.SubprocessError, ValueError) as e:
        print(f"  could not list processes: {e}")
        return 2
    if isinstance(procs, dict):
        procs = [procs]
    prefix = str(venv).lower() + "\\"
    hits = [p for p in procs if str(p.get("ExecutablePath", "")).lower().startswith(prefix)]
    for p in hits:
        print(f"  PID {p.get('ProcessId')}: {p.get('CommandLine')}")
    return 1 if hits else 0


if __name__ == "__main__":
    raise SystemExit(main())
