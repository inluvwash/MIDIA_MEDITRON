from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start Streamlit UI or FastAPI service")
    parser.add_argument("--api", action="store_true", help="start FastAPI instead of Streamlit")
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    if args.api:
        cmd = [sys.executable, "-m", "uvicorn", "api:app", "--host", "127.0.0.1", "--port", str(args.port or 8000), "--no-access-log"]
    else:
        cmd = [sys.executable, "-m", "streamlit", "run", "HealthScreening.py", "--server.address=127.0.0.1", "--server.port=" + str(args.port or 8501), "--browser.gatherUsageStats=false"]
    raise SystemExit(subprocess.call(cmd, cwd=ROOT))
