"""Complete the fixed frozen-readout compatibility experiment without choosing epochs or validation arms."""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


ZONE = datetime.timezone(datetime.timedelta(hours=8))
PYTHON = '/root/miniconda3/envs/bdetr/bin/python'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    directory = args.manifest.parent
    spec = read(args.manifest)
    assert sha(Path(__file__)) == spec['queue_script_sha256']
    training, formal = Path(spec['training_directory']), Path(spec['formal_directory'])
    assert sha(training / 'input_manifest.json') == spec['training_manifest_sha256']
    launch = read(training / 'launch.json')
    assert launch['manifest_sha256'] == spec['training_manifest_sha256']
    screen_pid = int(launch['screen_session'][0].split('.')[0])
    first = datetime.datetime.fromisoformat(launch['time_cst']) + datetime.timedelta(seconds=600)
    write(directory / 'observation_schedule.json', {'time_cst': datetime.datetime.now(ZONE).isoformat(),
        'training_screen_pid': screen_pid, 'first_check_cst': first.isoformat(), 'interval_seconds': 240,
        'basis': 'Previous matched preflight spent about7 minutes in native dataset/text preparation;then observe every240s until fixed terminal.'})
    time.sleep(max(0, (first - datetime.datetime.now(ZONE)).total_seconds()))
    while not (training / 'controller.exit').exists():
        process = subprocess.run(['ps', '-p', str(screen_pid), '-o', 'pid,stat,etime,args'], stdout=subprocess.PIPE)
        assert process.returncode == 0 and b'mcln_frozen_readout_pair_v1' in process.stdout, process.stdout.decode()
        with (training / 'run.log').open('rb') as stream:
            stream.seek(max(0, (training / 'run.log').stat().st_size - 64000))
            lines = stream.read().decode().splitlines()
        progress = {}
        for line in lines:
            for label in ['SCANREFER FROZEN READOUT EVAL ', 'SCANREFER FROZEN READOUT EVAL COMPLETE ', 'SCANREFER FROZEN READOUT TRAIN ']:
                if line.startswith(label + '{'):
                    progress[label.strip()] = json.loads(line[len(label):])
        value = {'time_cst': datetime.datetime.now(ZONE).isoformat(), 'screen_pid': screen_pid,
                 'screen_live': True, 'progress': progress}
        with (directory / 'training_observations.jsonl').open('a') as stream:
            stream.write(json.dumps(value, sort_keys=True) + '\n')
        time.sleep(240)
    assert (training / 'controller.exit').read_text().strip() == '0'
    receipt = read(training / 'receipt.json')
    assert receipt['status'] == 'complete' and receipt['steps_per_arm'] == 2482
    assert receipt['manifest_sha256'] == spec['training_manifest_sha256']
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
                       PYTHONPATH=str(training))
    audit_path = training / 'independent_audit.json'
    with (directory / 'training_audit.txt').open('xb') as stream:
        process = subprocess.run([PYTHON, '-m', 'scripts.audit_scanrefer_frozen_readout_pair', str(training), str(audit_path)],
                                 cwd=str(training), env=environment, stdout=stream, stderr=subprocess.STDOUT)
    assert process.returncode == 0
    audit = read(audit_path)
    assert audit['integrity_pass'] and audit['receipt_sha256'] == sha(training / 'receipt.json')
    training_manifest = read(training / 'input_manifest.json')
    if not audit['eligible_for_fixed_terminal_formal_evaluation']:
        decision = {'time_cst': datetime.datetime.now(ZONE).isoformat(),
            'status': 'module_rec_screen_not_passed', 'training_audit_sha256': sha(audit_path),
            'training_receipt_sha256': sha(training / 'receipt.json'), 'metrics': audit['metrics'],
            'comparisons': audit['comparisons'], 'formal_evaluation_count': 0,
            'nr3d_sr3d_training_started': False, 'no_epoch_or_arm_selected_on_validation': True}
        write(directory / 'decision.json', decision)
        print(json.dumps(decision), flush=True)
        return
    formal.mkdir()
    for subdir in ['scripts']:
        (formal / subdir).mkdir()
    for name, expected in training_manifest['files'].items():
        assert sha(training / name) == expected, name
        (formal / name).write_bytes((training / name).read_bytes())
        assert sha(formal / name) == expected
    formal_manifest = {
        'schema': 'mcln-scanrefer-frozen-readout-official-input-v1', 'training_directory': str(training),
        'training_receipt_sha256': sha(training / 'receipt.json'), 'training_audit_sha256': sha(audit_path),
        'trained_checkpoints': receipt['checkpoints'], 'files': training_manifest['files'],
        'data_root': training_manifest['data_root'], 'val_superpoint_files': training_manifest['val_superpoint_files'],
        'formal_rows': 9508, 'scan_rec_historical_floor_hits': [5572, 4797],
        'scan_mask_paper_floor_percent': [58.70, 50.70, 44.72], 'nr3d_sr3d_mask_gate': False,
        'candidate_predeclared': 'frozen_gt_v99', 'arms': ['protected_v99', 'native_only_v99', 'frozen_gt_v99'],
        'one_fixed_endpoint_no_validation_selection': True,
    }
    write(formal / 'input_manifest.json', formal_manifest)
    environment.update(CUDA_VISIBLE_DEVICES='0', PYTHONPATH=str(formal), MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
    command = ['flock', '-x', '/root/autodl-tmp/mcln_v99_backbone_gpu0.lock', PYTHON, '-u',
               'scripts/evaluate_scanrefer_frozen_readout_official.py', '--manifest', str(formal / 'input_manifest.json')]
    with (formal / 'run.log').open('xb') as stream:
        process = subprocess.Popen(command, cwd=str(formal), env=environment, stdout=stream, stderr=subprocess.STDOUT)
        write(formal / 'launch.json', {'time_cst': datetime.datetime.now(ZONE).isoformat(), 'process_pid': process.pid,
            'parent_queue_pid': os.getpid(), 'command': command, 'manifest_sha256': sha(formal / 'input_manifest.json'),
            'formal_rows': 9508, 'arms': formal_manifest['arms']})
        code = process.wait()
    (formal / 'controller.exit').write_text(str(code) + '\n')
    assert code == 0
    environment.update(CUDA_VISIBLE_DEVICES='')
    final_audit = formal / 'result/independent_audit.json'
    with (directory / 'formal_audit.txt').open('xb') as stream:
        process = subprocess.run([PYTHON, '-m', 'scripts.audit_scanrefer_frozen_readout_official', str(formal), str(final_audit)],
                                 cwd=str(formal), env=environment, stdout=stream, stderr=subprocess.STDOUT)
    assert process.returncode == 0
    final = read(final_audit)
    assert final['integrity_pass'] and final['formal_rows'] == 9508
    decision = {'time_cst': datetime.datetime.now(ZONE).isoformat(), 'training_audit_sha256': sha(audit_path),
        'formal_audit_sha256': sha(final_audit), 'formal_receipt_sha256': sha(formal / 'result/receipt.json'),
        'metrics': final['metrics'], 'promotion': final['promotion'],
        'nr3d_sr3d_rec_preflight_required': final['promotion']['advance_to_nr3d_sr3d_rec'],
        'nr3d_sr3d_training_started': False, 'formal_evaluation_count': 1,
        'no_epoch_or_arm_selected_on_validation': True}
    write(directory / 'decision.json', decision)
    print(json.dumps(decision), flush=True)


if __name__ == '__main__':
    main()
