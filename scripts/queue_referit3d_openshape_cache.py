"""Encode only after the existing ScanRefer formal queue has qualified."""
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


def write(path, data):
    with path.open('x') as stream:
        json.dump(data, stream, indent=2, sort_keys=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', type=Path, required=True)
    path = p.parse_args().plan.resolve()
    root = path.parent
    plan = json.loads(path.read_text())
    for name, digest in plan['files'].items():
        assert sha(root/name) == digest, name
    formal = Path(plan['scan_formal_root'])
    assert sha(formal/'plan.json') == plan['scan_formal_plan_sha256']
    while not (formal/'queue.exit').exists():
        time.sleep(240)
    code = int((formal/'queue.exit').read_text())
    if code != 0:
        write(root/'decision.json', dict(status='scan_formal_queue_failed', exit_code=code, gpu_forwards=0))
        return code
    decision = json.loads((formal/'decision.json').read_text())
    if decision['status'] != 'formal_evaluated_and_audited':
        write(root/'decision.json', dict(status='scan_not_qualified', scan_status=decision['status'], gpu_forwards=0))
        return 0
    audited_path = formal/'formal/independent_audit.json'
    receipt_path = formal/'formal/result/receipt.json'
    assert sha(audited_path) == decision['audit_sha256']
    assert sha(receipt_path) == decision['formal_receipt_sha256']
    audit = json.loads(audited_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    assert audit['promotion'] == receipt['promotion'] == decision['promotion']
    assert receipt['status'] == 'complete' and receipt['formal_rows'] == 9508
    if not decision['promotion']['advance_to_nr3d_sr3d_rec']:
        write(root/'decision.json', dict(status='scan_not_qualified', promotion=decision['promotion'], gpu_forwards=0))
        return 0
    cache = root/'cache'
    cache.mkdir()
    (cache/'features').mkdir()
    spec = dict(plan['cache_spec'])
    cache_script = root/'scripts/cache_referit3d_openshape_objects.py'
    spec['script_sha256'] = plan['files']['scripts/cache_referit3d_openshape_objects.py']
    spec['qualification'] = dict(decision=str(formal/'decision.json'),
        files={str(q):sha(q) for q in [formal/'decision.json', audited_path, receipt_path]})
    write(cache/'manifest.json', spec)
    runtime = Path(spec['runtime_root'])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
        OPENBLAS_NUM_THREADS='4', DGLBACKEND='pytorch', PYTHONPATH=str(runtime/'vendor')+':'+str(runtime/'deps'))
    with (cache/'cache.log').open('xb') as log:
        result = subprocess.run(['flock','-n','/root/autodl-tmp/mcln_v99_backbone_gpu0.lock',
            sys.executable,str(cache_script),'--manifest',str(cache/'manifest.json')],
            env=env,stdout=log,stderr=subprocess.STDOUT)
    (cache/'cache.exit').write_text(str(result.returncode)+'\n')
    if result.returncode != 0:
        return result.returncode
    cached = json.loads((cache/'receipt.json').read_text())
    assert cached['status'] == 'complete' and cached['scene_count'] == 1200
    write(root/'decision.json',dict(status='train_cache_complete',cache_receipt_sha256=sha(cache/'receipt.json'),
        model_loading_and_gradients_pending=True,nr3d_sr3d_training_started=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
