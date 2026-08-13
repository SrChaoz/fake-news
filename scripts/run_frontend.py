#!/usr/bin/env python3
"""Ejecuta el frontend Streamlit usando el mismo intérprete de Python."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_APP = PROJECT_ROOT / "app" / "frontend" / "app.py"


def main() -> int:
    """Delega en Streamlit y propaga su código de salida."""
    command = [sys.executable, "-m", "streamlit", "run", str(FRONTEND_APP), *sys.argv[1:]]
    return subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
