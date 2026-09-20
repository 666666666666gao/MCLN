"""Offline terminal analysis; never starts, restarts, or alters model jobs."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


root = Path(__file__).resolve().parent
spec = json.loads((root / 'analysis_spec.json').read_text())
assert hashlib.sha256((root / 'spec.json').read_bytes()).hexdigest() == spec['training_spec_sha256']
for name, digest in spec['scripts'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
while not (root / 'controller.exit').exists():
    time.sleep(300)
assert (root / 'controller.exit').read_text().strip() == '0'
for name, command in [
    ('paired_rec', [sys.executable, '-u', str(root / 'analyze_eg3dvg_task_read.py'), '--root', str(root)]),
    ('candidate_analysis', [sys.executable, '-u', str(root / 'analyze_eg3dvg_nr3d_candidates.py'),
                            '--root', str(root / 'evaluation')]),
]:
    with (root / (name + '.log')).open('xb') as log:
        code = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT)
    (root / (name + '.exit')).write_text(str(code) + '\n')
    if code:
        raise SystemExit(code)
