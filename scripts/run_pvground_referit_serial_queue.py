"""Run fixed Nr3D then Sr3D stages after the existing ScanRefer REC decision."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def record(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def wait_dependency(root, pid):
    root = Path(root)
    while not (root / 'controller.exit').is_file():
        process = Path('/proc') / str(pid) / 'cmdline'
        assert process.is_file() and (str(root) + '/controller.py').encode() in process.read_bytes(), str(root)
        time.sleep(300)
    assert (root / 'controller.exit').read_text().strip() == '0', str(root)


def scan_pass(root):
    root = Path(root)
    decision = read(root / 'decision.json')
    if decision['status'] == 'skipped_primary_rec_regression':
        return False
    assert decision['status'] == 'launching_fixed_formal'
    audit = read(root / 'audit.json')
    assert audit['integrity_pass'] and audit['formal_rows'] == 9508
    assert audit['scanrefer_mask_gate'] is False
    assert audit['receipt_sha256'] == sha(root / 'receipt.json')
    assert audit['advance_to_nr3d_sr3d_rec'] == all(audit['checks'].values())
    return audit['advance_to_nr3d_sr3d_rec']


def run_stage(command, root, exit_name, log_name, environment):
    root = Path(root)
    assert not (root / exit_name).exists()
    with (root / log_name).open('x') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
    with (root / exit_name).open('x') as stream:
        stream.write(str(result.returncode) + '\n')
    assert result.returncode == 0, str(root / log_name)


def run_dataset(item, python, environment, gpu_lock):
    training, endpoint, formal = [Path(item[key]) for key in ['training_root', 'audit_root', 'formal_root']]
    assert shutil.disk_usage(training).free >= item['training_free_bytes']
    # The training entry owns the GPU lock; a second outer flock would deadlock.
    run_stage([python, '-u', str(training / 'train.py'), '--spec', str(training / 'spec.json')],
              training, 'controller.exit', 'controller.log', environment)
    cpu_environment = dict(environment, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1')
    run_stage([python, '-u', str(endpoint / 'audit.py'), '--root', str(training), '--out', str(endpoint / 'audit.json')],
              endpoint, 'controller.exit', 'controller.log', cpu_environment)
    audit = read(endpoint / 'audit.json')
    receipt = read(training / 'receipt.json')
    assert audit['integrity_pass'] and audit['rec_competition_verified']
    assert audit['receipt_sha256'] == sha(training / 'receipt.json')
    assert receipt['terminal_sha256'] == sha(training / 'terminal.pth')
    latest = training / 'latest.pth'
    assert latest.resolve().parent == training.resolve() and latest.is_file()
    cleanup = dict(path=str(latest), bytes=latest.stat().st_size, sha256=sha(latest),
                   retained_terminal_sha256=receipt['terminal_sha256'])
    record(endpoint / 'cleanup_planned.json', cleanup)
    latest.unlink()
    record(endpoint / 'cleanup_receipt.json', dict(cleanup, deleted=not latest.exists()))
    if not audit['primary_rec_nonregression']:
        record(formal / 'decision.json', dict(status='skipped_primary_rec_regression', formal_rows=0,
                                             transitions=audit['transitions']['bbs']))
        return dict(dataset=item['dataset'], status='module_rec_regression', formal_rows=0)
    assert shutil.disk_usage(formal).free >= item['formal_free_bytes']
    record(formal / 'decision.json', dict(status='launching_fixed_formal',
                                         training_audit_sha256=sha(endpoint / 'audit.json')))
    run_stage(['flock', gpu_lock, python, '-u', str(formal / 'evaluate.py'), '--spec', str(formal / 'spec.json')],
              formal, 'evaluation.exit', 'evaluation.log', environment)
    run_stage([python, '-u', str(formal / 'audit.py'), '--root', str(formal), '--out', str(formal / 'audit.json')],
              formal, 'controller.exit', 'audit.log', cpu_environment)
    result = read(formal / 'audit.json')
    assert result['integrity_pass'] and result['mask_gate'] is False
    return dict(dataset=item['dataset'], status='formal_complete', formal_rows=result['formal_rows'],
                rec_target_pass=result['rec_target_pass'], audit_sha256=sha(formal / 'audit.json'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', type=Path, required=True)
    args = parser.parse_args()
    spec = read(args.spec)
    root = args.spec.parent
    for path, digest in spec['files'].items():
        assert sha(path) == digest, path
    assert [item['dataset'] for item in spec['datasets']] == ['nr3d', 'sr3d']
    runtime = Path(spec['runtime'])
    env_spec = read(runtime / 'env_spec.json')
    assert hashlib.sha256(json.dumps(env_spec, sort_keys=True, separators=(',', ':')).encode()).hexdigest() == spec['env_spec_sha256']
    time.sleep(max(0, datetime.datetime.fromisoformat(spec['first_check_cst']).timestamp() - time.time()))
    wait_dependency(spec['scan_formal_root'], spec['scan_formal_pid'])
    if not scan_pass(spec['scan_formal_root']):
        record(root / 'decision.json', dict(status='skipped_scanrefer_rec', training_jobs_launched=0))
        return
    # Both original single-batch checks finish before either full fit takes the GPU.
    for item in spec['datasets']:
        wait_dependency(item['probe_root'], item['probe_pid'])
        probe = read(Path(item['probe_root']) / 'actual_probe/receipt.json')
        assert probe['status'] == 'pass' and probe['full_state_restored']
        assert probe['optimizer_steps'] == 0 and probe['model_forwards'] == 1
    environment = dict(os.environ, **env_spec['env'])
    results = []
    for item in spec['datasets']:
        result = run_dataset(item, str(runtime / 'venv/bin/python'), environment, spec['gpu_lock'])
        results.append(result)
        record(root / (item['dataset'] + '_result.json'), result)
    record(root / 'decision.json', dict(status='complete', results=results))


if __name__ == '__main__':
    main()
