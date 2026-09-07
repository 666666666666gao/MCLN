"""Run a bounded model probe only after qualified real cache completion."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', type=Path, required=True)
    path = p.parse_args().plan.resolve()
    root = path.parent
    plan = json.loads(path.read_text())
    for name, digest in plan['files'].items():
        assert sha(root / name) == digest, name
    previous = Path(plan['cache_queue_root'])
    assert sha(previous / 'plan.json') == plan['cache_queue_plan_sha256']
    while not (previous / 'queue.exit').exists():
        time.sleep(240)
    code = int((previous / 'queue.exit').read_text())
    if code != 0:
        write(root / 'decision.json', dict(status='cache_queue_failed', exit_code=code, model_forwards=0))
        return code
    decision = json.loads((previous / 'decision.json').read_text())
    if decision['status'] != 'train_cache_complete':
        write(root / 'decision.json', dict(status='scan_not_qualified', cache_status=decision['status'], model_forwards=0))
        return 0
    cache = previous / 'cache'
    assert sha(cache / 'receipt.json') == decision['cache_receipt_sha256']
    manifest = dict(plan['probe_spec'], files=plan['files'], cache_root=str(cache),
                    cache_receipt_sha256=decision['cache_receipt_sha256'])
    write(root / 'probe_manifest.json', manifest)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    with (root / 'probe.log').open('xb') as log:
        result = subprocess.run(['flock', '-n', '/root/autodl-tmp/mcln_v99_backbone_gpu0.lock',
            sys.executable, str(root / 'scripts/probe_referit3d_pretrained_appearance.py'), '--manifest', str(root / 'probe_manifest.json')],
            env=env, stdout=log, stderr=subprocess.STDOUT)
    (root / 'probe.exit').write_text(str(result.returncode) + '\n')
    if result.returncode != 0:
        return result.returncode
    receipt = json.loads((root / 'receipt.json').read_text())
    assert receipt['status'] == 'pass' and receipt['checkpoint_writes'] == 0
    assert receipt['model_forwards'] == 12 and receipt['disposable_optimizer_steps'] == 4
    write(root / 'decision.json', dict(status='real_model_probe_passed', receipt_sha256=sha(root / 'receipt.json'),
                                     full_nr3d_sr3d_training_started=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
