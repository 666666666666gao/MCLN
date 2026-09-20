"""Analyze the first complete ScanRefer pair after each existing formal audit."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--controller-pid', type=int, required=True)
    args = parser.parse_args()
    specification = json.loads((args.root / 'candidate_analysis_spec.json').read_text())
    for name, digest in specification['scripts'].items():
        assert hashlib.sha256((args.root / name).read_bytes()).hexdigest() == digest
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1')
    for arm in ['task', 'native']:
        evaluation = args.root / arm / 'evaluation_01'
        while not (evaluation / 'formal/audit.json').exists():
            assert not (args.root / 'controller.exit').exists(), 'Campaign ended before expected formal audit'
            command = Path('/proc/%d/cmdline' % args.controller_pid).read_bytes().split(b'\0')
            assert str(args.root / 'controller.py').encode() in command
            print('WAIT_FORMAL_AUDIT ' + str(evaluation), flush=True)
            time.sleep(300)
        subprocess.check_call([sys.executable, '-u', str(args.root / 'analyze_eg3dvg_scan_candidates.py'),
                               '--root', str(evaluation)], env=env)
    report = dict(status='complete', epoch=1, arms=['task', 'native'],
                  model_forwards=0, optimizer_steps=0, training_modified=False)
    with (args.root / 'candidate_analysis_receipt.json').open('x') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
