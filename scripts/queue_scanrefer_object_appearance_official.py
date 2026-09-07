"""Prepare frozen val features and evaluate once only after the fixed train screen."""
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
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    plan_path = args.plan.resolve()
    root = plan_path.parent
    plan = json.loads(plan_path.read_text())
    for name, digest in plan['files'].items():
        assert sha(root / name) == digest, name
    training = Path(plan['training_root'])
    assert sha(training / 'input_manifest.json') == plan['training_manifest_sha256']
    if not args.execute:
        while not (training / 'native_queue.exit').exists():
            time.sleep(240)
        code = int((training / 'native_queue.exit').read_text())
        if code != 0:
            write(root / 'decision.json', dict(status='native_followup_failed', exit_code=code, formal_rows=0))
            return code
        audit = json.loads((training / 'audit.json').read_text())
        assert audit['integrity_pass']
        if not audit['module_rec_screen_pass']:
            write(root / 'decision.json', dict(status='module_screen_rejected', formal_rows=0,
                  reason='Fixed appearance endpoint regresses versus baseline or control; no validation run.'))
            return 0
        with (root / 'execution.log').open('xb') as log:
            result = subprocess.run(['flock', '-n', '/root/autodl-tmp/mcln_v99_backbone_gpu0.lock',
                sys.executable, str(Path(__file__).resolve()), '--plan', str(plan_path), '--execute'],
                stdout=log, stderr=subprocess.STDOUT)
        return result.returncode

    # Recheck eligibility inside the actual exclusive GPU execution.
    audit = json.loads((training / 'audit.json').read_text())
    assert audit['integrity_pass'] and audit['module_rec_screen_pass']
    assert (training / 'native_queue.exit').read_text().strip() == '0'
    train = json.loads((training / 'input_manifest.json').read_text())
    endpoint = json.loads((training / 'receipt.json').read_text())
    assert endpoint['development_dual_rec_nonregression'] and endpoint['steps_per_arm'] == 2482
    assert sha(Path(train['cache_root']) / 'manifest.json') == plan['train_cache_manifest_sha256']
    cache_spec = json.loads((Path(train['cache_root']) / 'manifest.json').read_text())
    assert sha(plan['val_scene_list']) == plan['val_scene_list_sha256']
    assert len(set(Path(plan['val_scene_list']).read_text().split())) == 141
    cache = root / 'validation_cache'
    cache.mkdir()
    (cache / 'features').mkdir()
    cache_script = root / 'scripts/cache_scanrefer_openshape_validation_objects.py'
    cache_spec.update(scene_list=plan['val_scene_list'], scene_list_sha256=plan['val_scene_list_sha256'],
                      scene_count=141, script_sha256=sha(cache_script))
    write(cache / 'manifest.json', cache_spec)
    runtime = Path(cache_spec['runtime_root'])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    cache_env = dict(env, PYTHONPATH=str(runtime/'vendor')+':'+str(runtime/'deps'), DGLBACKEND='pytorch')
    with (cache / 'cache.log').open('xb') as log:
        result = subprocess.run([sys.executable, str(cache_script), '--manifest', str(cache/'manifest.json')],
                                env=cache_env, stdout=log, stderr=subprocess.STDOUT)
    (cache / 'cache.exit').write_text(str(result.returncode)+'\n')
    if result.returncode != 0:
        return result.returncode
    cache_receipt = json.loads((cache / 'receipt.json').read_text())
    assert cache_receipt['status'] == 'complete' and cache_receipt['scene_count'] == 141
    formal = root / 'formal'
    formal.mkdir()
    (formal / 'scripts').mkdir()
    sources = {}
    for name, digest in plan['files'].items():
        if name.startswith('scripts/'):
            data = (root/name).read_bytes()
            (formal/name).write_bytes(data)
            sources[name] = digest
    manifest = dict(schema='mcln-scanrefer-object-appearance-official-input-v1',
        training_directory=str(training), training_receipt_sha256=sha(training/'receipt.json'),
        training_audit_sha256=sha(training/'audit.json'), native_receipt_sha256=sha(training/'native_evaluation/receipt.json'),
        trained_checkpoint=endpoint['checkpoints']['appearance'], files=sources,
        data_root=train['data_root'], val_superpoint_files=train['superpoint_files']['val'],
        cache_root=str(cache), cache_receipt_sha256=sha(cache/'receipt.json'),
        formal_rows=9508, optimizer_steps=0, scan_rec_historical_floor_hits=[5572,4797],
        scan_mask_paper_floor_percent=[58.70,50.70,44.72], nr3d_sr3d_mask_gate=False,
        decision='Fixed2482 appearance endpoint after module screen; no endpoint/threshold selection.')
    write(formal/'input_manifest.json', manifest)
    formal_env = dict(env, PYTHONPATH=str(formal))
    for script, arguments, logfile, exitfile in [
        ('evaluate_scanrefer_object_appearance_official.py', ['--manifest', str(formal/'input_manifest.json')], 'controller.log', 'controller.exit'),
        ('audit_scanrefer_object_appearance_official.py', [str(formal), str(formal/'independent_audit.json')], 'audit.log', 'audit.exit')]:
        with (formal/logfile).open('xb') as log:
            result = subprocess.run([sys.executable, str(formal/'scripts'/script)] + arguments,
                                    env=formal_env, stdout=log, stderr=subprocess.STDOUT)
        (formal/exitfile).write_text(str(result.returncode)+'\n')
        if result.returncode != 0:
            return result.returncode
    final = json.loads((formal/'independent_audit.json').read_text())
    write(root/'decision.json', dict(status='formal_evaluated_and_audited', formal_rows=9508,
          formal_receipt_sha256=sha(formal/'result/receipt.json'), audit_sha256=sha(formal/'independent_audit.json'),
          promotion=final['promotion'], nr3d_sr3d_training_started=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
