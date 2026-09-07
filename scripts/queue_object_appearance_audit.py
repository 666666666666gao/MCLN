"""Wait for the existing paired job, then audit its saved endpoints on CPU."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    root = parser.parse_args().root.resolve()
    while not (root / 'controller.exit').exists():
        time.sleep(240)
    code = int((root / 'controller.exit').read_text())
    if code != 0:
        print(json.dumps(dict(status='training_process_failed', controller_exit=code)), flush=True)
        return code
    with (root / 'audit.log').open('xb') as log:
        result = subprocess.run([sys.executable, str(root / 'scripts/audit_scanrefer_object_appearance_pair.py'), '--root', str(root)], stdout=log, stderr=subprocess.STDOUT)
    print(json.dumps(dict(status='audit_finished', exit_code=result.returncode)), flush=True)
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
