import datetime
import json
import os
from pathlib import Path
import subprocess

root = Path(__file__).parent
runtime = Path("/root/autodl-tmp/mcln_pvground_runtime_20260908_v1")
environment = os.environ.copy()
environment.update(json.loads((runtime / "env_spec.json").read_bytes())["env"])
environment.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
(root / "controller.pid").write_text(str(os.getpid()) + "\n")
with (root / "audit.log").open("xb") as log:
    result = subprocess.run([str(runtime / "venv/bin/python"), "-u", str(root / "audit.py"), "--spec", str(root / "spec.json")], env=environment, stdout=log, stderr=subprocess.STDOUT)
(root / "audit.exit").write_text(str(result.returncode) + "\n")
(root / "controller.exit").write_text(str(result.returncode) + "\n")
raise SystemExit(result.returncode)
