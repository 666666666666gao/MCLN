"""Run native endpoint REC after the existing paired training and CPU audit."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    root = parser.parse_args().root.resolve()
    sources = json.loads((root / 'native_evaluation_sources.json').read_text())
    for name, digest in sources.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
    while not (root / 'audit_queue.exit').exists():
        time.sleep(240)
    code = int((root / 'audit_queue.exit').read_text())
    if code != 0:
        print(json.dumps(dict(status='prior_audit_failed', audit_exit=code)), flush=True)
        return code
    command = ['flock', '-n', '/root/autodl-tmp/mcln_v99_backbone_gpu0.lock', sys.executable,
               str(root / 'scripts/evaluate_scanrefer_object_appearance_native.py'),
               '--manifest', str(root / 'input_manifest.json'), '--output', str(root / 'native_evaluation')]
    with (root / 'native_evaluation.log').open('xb') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    print(json.dumps(dict(status='native_evaluation_process_finished', exit_code=result.returncode)), flush=True)
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
