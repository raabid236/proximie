from pathlib import Path
import subprocess, sys

ROOT = Path(__file__).resolve().parents[2]
subprocess.run([sys.executable, str(ROOT / "scripts/run_pipeline.py")], check=True)
