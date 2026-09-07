"""Continue the fixed native-box pair, without changing training or choosing a validation arm."""
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
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def validated_module_pass(receipt, audit):
    assert receipt['schema'] == 'mcln-scanrefer-native-box-transfer-pair-v1'
    assert receipt['status'] == 'complete' and receipt['formal_rows'] == 0
    assert receipt['steps_per_arm'] == 2482 and receipt['holdout_rows'] == 6887
    assert audit['status'] == 'pass' and audit['formal_rows'] == 0
    candidate = receipt['terminal_metrics']['gt_teacher_box']
    references = [receipt['baseline_metrics']['gt_teacher_box'], receipt['terminal_metrics']['gt_only']]
    passed = all(candidate['rec_hits' + suffix] >= reference['rec_hits' + suffix]
                 for reference in references for suffix in ['025', '050'])
    assert passed == receipt['eligible_for_fixed_terminal_formal_evaluation']
    assert passed == audit['eligible_for_fixed_terminal_formal_evaluation']
    return passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    option = parser.parse_args()
    directory = option.manifest.parent
    spec = read(option.manifest)
    assert spec['schema'] == 'mcln-native-box-transfer-posttraining-queue-v1'
    assert sha(__file__) == spec['queue_script_sha256']
    assert spec['interval_seconds'] == 240 and spec['candidate_predeclared'] == 'gt_teacher_box'
    training = Path(spec['training_directory'])
    preparation = Path(spec['formal_preparation_directory'])
    formal = Path(spec['formal_directory'])
    assert sha(training / 'input_manifest.json') == spec['training_manifest_sha256']
    training_manifest = read(training / 'input_manifest.json')
    assert training_manifest['data_root'] == spec['data_root']
    assert sha(preparation / 'preparation.json') == spec['formal_preparation_sha256']
    prep = read(preparation / 'preparation.json')
    assert prep['candidate_predeclared'] == 'gt_teacher_box_v99' and prep['formal_rows_executed'] == 0
    for name, digest in prep['files'].items():
        assert sha(preparation / name) == digest, name
    screen_pid = spec['training_screen_pid']
    first = datetime.datetime.fromisoformat(spec['first_check_cst'])
    write(directory / 'observation_schedule.json', {'time_cst': datetime.datetime.now(ZONE).isoformat(),
        'training_screen_pid': screen_pid, 'first_check_cst': first.isoformat(), 'interval_seconds': 240,
        'basis': 'First check near estimated baseline completion; thereafter lightweight remote process/log observations every240s.'})
    time.sleep(max(0., (first - datetime.datetime.now(ZONE)).total_seconds()))
    while not (training / 'controller.exit').exists():
        process = subprocess.run(['ps', '-p', str(screen_pid), '-o', 'pid,stat,etime,args'], stdout=subprocess.PIPE)
        assert process.returncode == 0 and b'mcln_native_box_transfer_pair_v1' in process.stdout, process.stdout.decode()
        log = training / 'run.log'
        with log.open('rb') as stream:
            stream.seek(max(0, log.stat().st_size - 128000))
            lines = stream.read().decode().splitlines()
        progress = {}
        for line in lines:
            for label in ['SCANREFER NATIVE BOX TRANSFER EVAL ', 'SCANREFER NATIVE BOX TRANSFER EVAL COMPLETE ',
                          'SCANREFER NATIVE BOX TRANSFER TRAIN ', 'SCANREFER NATIVE BOX TRANSFER PAIR COMPLETE ']:
                if line.startswith(label + '{'):
                    progress[label.strip()] = json.loads(line[len(label):])
        value = {'time_cst': datetime.datetime.now(ZONE).isoformat(), 'screen_pid': screen_pid,
                 'live_process': process.stdout.decode(), 'progress': progress}
        with (directory / 'training_observations.jsonl').open('a') as stream:
            stream.write(json.dumps(value, sort_keys=True) + '\n')
        print('NATIVE BOX TRANSFER QUEUE OBSERVED', json.dumps({'time_cst': value['time_cst'],
              'screen_pid': screen_pid, 'stages': list(progress)}), flush=True)
        time.sleep(240)
    assert (training / 'training.exit').read_text().strip() == '0'
    assert (training / 'controller.exit').read_text().strip() == '0'
    receipt = read(training / 'receipt.json')
    audit_path = training / 'independent_audit.json'
    audit = read(audit_path)
    assert receipt['manifest_sha256'] == spec['training_manifest_sha256']
    assert audit['manifest_sha256'] == spec['training_manifest_sha256']
    assert audit['receipt_sha256'] == sha(training / 'receipt.json')
    # The training controller already executed its independent CPU audit exactly once.
    # Do not call that exclusive-output audit again from this queue.
    if not validated_module_pass(receipt, audit):
        decision = {'time_cst': datetime.datetime.now(ZONE).isoformat(), 'status': 'module_rec_screen_not_passed',
            'training_receipt_sha256': sha(training / 'receipt.json'), 'training_audit_sha256': sha(audit_path),
            'metrics': receipt['terminal_metrics'], 'native_rec_effects': receipt['native_rec_effects'],
            'system_rec_effects': receipt['system_rec_effects'], 'formal_evaluation_count': 0,
            'nr3d_sr3d_training_started': False, 'fixed_candidate_not_replaced_by_control': True}
        write(directory / 'decision.json', decision)
        print(json.dumps(decision), flush=True)
        return
    for item in receipt['checkpoints'].values():
        assert sha(item['path']) == item['sha256'] and Path(item['path']).stat().st_size == item['bytes']
    formal.mkdir()
    for name, digest in prep['files'].items():
        assert sha(preparation / name) == digest
        path = formal / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((preparation / name).read_bytes())
        assert sha(path) == digest
    formal_manifest = {
        'schema': 'mcln-scanrefer-native-box-transfer-official-input-v1', 'training_directory': str(training),
        'training_receipt_sha256': sha(training / 'receipt.json'), 'training_audit_sha256': sha(audit_path),
        'trained_checkpoints': receipt['checkpoints'], 'files': prep['files'],
        'data_root': spec['data_root'], 'val_superpoint_files': spec['val_superpoint_files'],
        'formal_rows': 9508, 'scan_rec_historical_floor_hits': [5572, 4797],
        'scan_mask_paper_floor_percent': [58.70, 50.70, 44.72], 'nr3d_sr3d_mask_gate': False,
        'candidate_predeclared': 'gt_teacher_box_v99', 'arms': ['protected_v99', 'gt_only_v99', 'gt_teacher_box_v99'],
        'one_fixed_endpoint_no_validation_selection': True,
    }
    write(formal / 'input_manifest.json', formal_manifest)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='0', PYTHONPATH=str(formal),
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false')
    command = ['flock', '-x', '/root/autodl-tmp/mcln_v99_backbone_gpu0.lock', PYTHON, '-u',
               'scripts/evaluate_scanrefer_native_box_transfer_official.py', '--manifest', str(formal / 'input_manifest.json')]
    with (formal / 'run.log').open('xb') as stream:
        process = subprocess.Popen(command, cwd=str(formal), env=environment, stdout=stream, stderr=subprocess.STDOUT)
        write(formal / 'launch.json', {'time_cst': datetime.datetime.now(ZONE).isoformat(),
            'process_pid': process.pid, 'parent_queue_pid': os.getpid(), 'command': command,
            'manifest_sha256': sha(formal / 'input_manifest.json'), 'formal_rows': 9508, 'arms': formal_manifest['arms']})
        code = process.wait()
    (formal / 'controller.exit').write_text(str(code) + '\n')
    assert code == 0
    environment.update(CUDA_VISIBLE_DEVICES='')
    final_audit = formal / 'result/independent_audit.json'
    with (directory / 'formal_audit.txt').open('xb') as stream:
        process = subprocess.run([PYTHON, '-m', 'scripts.audit_scanrefer_native_box_transfer_official',
                                  str(formal), str(final_audit)], cwd=str(formal), env=environment,
                                 stdout=stream, stderr=subprocess.STDOUT)
    assert process.returncode == 0
    final = read(final_audit)
    assert final['integrity_pass'] and final['formal_rows'] == 9508
    decision = {'time_cst': datetime.datetime.now(ZONE).isoformat(),
        'status': 'scanrefer_promoted' if final['promotion']['advance_to_nr3d_sr3d_rec'] else 'formal_scanrefer_not_promoted',
        'training_audit_sha256': sha(audit_path), 'formal_audit_sha256': sha(final_audit),
        'formal_receipt_sha256': sha(formal / 'result/receipt.json'), 'metrics': final['metrics'],
        'native_rec_metrics': final['native_rec_metrics'], 'promotion': final['promotion'],
        'nr3d_sr3d_rec_preflight_required': final['promotion']['advance_to_nr3d_sr3d_rec'],
        'nr3d_sr3d_training_started': False, 'formal_evaluation_count': 1,
        'fixed_candidate_not_replaced_by_control': True}
    write(directory / 'decision.json', decision)
    print(json.dumps(decision), flush=True)


if __name__ == '__main__':
    main()
