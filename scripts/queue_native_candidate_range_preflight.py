"""Wait for the fixed ScanRefer decision, then run the prepared native GPU probe."""

import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time


PYTHON = '/root/miniconda3/envs/bdetr/bin/python'
ZONE = datetime.timezone(datetime.timedelta(hours=8))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def validate_formal(formal, decision, evaluation):
    assert (formal / 'controller.exit').read_text().strip() == '0'
    receipt_path, audit_path = formal / 'result/receipt.json', formal / 'result/independent_audit.json'
    receipt, audit = read(receipt_path), read(audit_path)
    assert decision['formal_receipt_sha256'] == sha(receipt_path)
    assert decision['formal_audit_sha256'] == sha(audit_path)
    assert receipt['schema'] == 'mcln-scanrefer-range-official-v1'
    assert audit['schema'] == 'mcln-scanrefer-range-official-audit-v1'
    assert receipt['status'] == 'complete' and receipt['formal_rows'] == audit['formal_rows'] == 9508
    assert audit['integrity_pass'] and audit['receipt_sha256'] == sha(receipt_path)
    assert receipt['optimizer_steps'] == receipt['checkpoint_writes'] == 0
    assert receipt['all_model_states_unchanged'] and receipt['native_evaluators_match_row_metrics']
    assert receipt['manifest_sha256'] == sha(formal / 'input_manifest.json')
    assert read(formal / 'input_manifest.json')['candidate_predeclared'] == 'local_v99'
    rows_path = formal / 'result/rows.json'
    assert receipt['rows_sha256'] == sha(rows_path)
    rows = read(rows_path)
    metrics = {}
    for arm in ['protected_v99', 'center_v99', 'local_v99']:
        metrics[arm] = evaluation.row_metrics(rows[arm])
        assert len(rows[arm]) == len({row['row_id'] for row in rows[arm]}) == 9508
        for key, value in metrics[arm].items():
            for reported in [receipt['metrics'][arm], audit['metrics'][arm], decision['metrics'][arm]]:
                if key == 'mask_miou':
                    assert abs(value - reported[key]) < 1e-8
                else:
                    assert value == reported[key], (arm, key)
    for arm in ['center_v99', 'local_v99']:
        for before, after in zip(rows['protected_v99'], rows[arm]):
            assert all(before[key] == after[key] for key in ['row_id', 'scan_id', 'physical_space', 'point_sha256'])
    promotion = evaluation.promotion_check(metrics['protected_v99'], metrics['local_v99'])
    assert promotion == receipt['promotion'] == audit['promotion'] == decision['promotion']
    assert decision['native_range_preflight_required'] == promotion['advance_to_nr3d_sr3d_rec']
    return promotion, metrics, receipt['data_root']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True, type=Path)
    args = parser.parse_args()
    directory = args.manifest.parent
    manifest = read(args.manifest)
    assert manifest['schema'] == 'mcln-native-range-preflight-queue-v1'
    assert sha(Path(__file__)) == manifest['queue_script_sha256']
    assert manifest['interval_seconds'] == 240
    upstream, formal, prep, native = [Path(manifest[name]) for name in
        ['upstream_directory', 'formal_directory', 'preparation_directory', 'native_preflight_directory']]
    assert sha(upstream / 'input_manifest.json') == manifest['upstream_manifest_sha256']
    assert sha(prep / 'receipt.json') == manifest['preparation_receipt_sha256']
    assert sha(prep / 'expected.json') == manifest['preparation_expected_sha256']
    prepared, expected = read(prep / 'receipt.json'), read(prep / 'expected.json')
    assert prepared['status'] == 'pass' and prepared['reader_variant'] == 'extent'
    assert prepared['gpu_forwards'] == prepared['native_model_optimizer_updates'] == prepared['checkpoint_writes'] == 0
    upstream_pid = manifest['upstream_screen_pid']
    while not (upstream / 'controller.exit').exists():
        process = subprocess.run(['ps', '-p', str(upstream_pid), '-o', 'pid,stat,etime,args'], stdout=subprocess.PIPE)
        assert process.returncode == 0 and b'mcln_scanrefer_range_posttraining_v1' in process.stdout
        with (directory / 'upstream_observations.jsonl').open('a') as stream:
            stream.write(json.dumps({'time_cst': datetime.datetime.now(ZONE).isoformat(),
                'upstream_screen_pid': upstream_pid, 'process': process.stdout.decode(), 'native_gpu_started': False}) + '\n')
        time.sleep(240)
    assert (upstream / 'controller.exit').read_text().strip() == '0'
    decision = read(upstream / 'decision.json')
    evaluator_path = formal / 'scripts/evaluate_scanrefer_range_official.py'
    assert sha(evaluator_path) == manifest['evaluation_script_sha256']
    spec = importlib.util.spec_from_file_location('fixed_range_evaluation', str(evaluator_path))
    evaluation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluation)
    promotion, metrics, data_root = validate_formal(formal, decision, evaluation)
    if not promotion['advance_to_nr3d_sr3d_rec']:
        write(directory / 'decision.json', {'status': 'scanrefer_not_promoted', 'promotion': promotion,
            'metrics': metrics, 'native_preflight_started': False, 'nr3d_sr3d_training_started': False,
            'time_cst': datetime.datetime.now(ZONE).isoformat()})
        return
    assert data_root == prepared['data_root']
    process = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], stdout=subprocess.PIPE, check=True)
    assert not process.stdout.strip(), process.stdout.decode()
    native.mkdir()
    file_names = ['annotation_receipt.json', 'preflight_rows.json', 'nr_contract.json', 'data_inputs.json',
                  'run_native_candidate_range_preflight.py']
    for name in file_names:
        assert sha(prep / name) == expected['prepared_files'][name], name
        (native / name).write_bytes((prep / name).read_bytes())
    source = prep / 'model_source'
    assert sha(source / 'native_source_manifest.json') == prepared['source_manifest_sha256']
    specification = {'schema': 'mcln-native-range-gpu-preflight-input-v1', 'model_source': str(source),
        'source_manifest_sha256': prepared['source_manifest_sha256'],
        'annotation_source_manifest_sha256': expected['parent_manifest_sha256'],
        'candidate_local_visual_variant': 'extent', 'checkpoint': expected['checkpoint'],
        'checkpoint_sha256': expected['checkpoint_sha256'], 'data_root': data_root,
        'scan_formal_receipt': str(formal / 'result/receipt.json'),
        'scan_formal_receipt_sha256': decision['formal_receipt_sha256'],
        'scan_formal_audit': str(formal / 'result/independent_audit.json'),
        'scan_formal_audit_sha256': decision['formal_audit_sha256'],
        'files': {name: sha(native / name) for name in file_names},
        'rows_per_dataset': 16, 'optimizer_steps_per_dataset': 2,
        'core_learning_rate': 1e-6, 'local_learning_rate': 1e-4, 'checkpoint_writes': 0,
        'pretraining_scope': 'Protected Nr weights for both compatible protocols;Sr historical best not restored.'}
    write(native / 'input_manifest.json', specification)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1')
    command = ['flock', '-x', '/root/autodl-tmp/mcln_v99_backbone_gpu0.lock', PYTHON, '-u',
               'run_native_candidate_range_preflight.py', '--manifest', str(native / 'input_manifest.json')]
    with (native / 'run.log').open('xb') as stream:
        process = subprocess.Popen(command, cwd=str(native), env=environment, stdout=stream, stderr=subprocess.STDOUT)
        write(native / 'launch.json', {'time_cst': datetime.datetime.now(ZONE).isoformat(), 'process_pid': process.pid,
            'parent_queue_pid': os.getpid(), 'command': command, 'manifest_sha256': sha(native / 'input_manifest.json')})
        code = process.wait()
    (native / 'controller.exit').write_text(str(code) + '\n')
    assert code == 0
    result = read(native / 'receipt.json')
    assert result['schema'] == 'mcln-native-candidate-range-gpu-preflight-v1' and result['status'] == 'pass'
    assert result['manifest_sha256'] == sha(native / 'input_manifest.json')
    assert result['checkpoint_writes'] == result['formal_rows'] == 0
    assert result['gpu_forwards'] == 14 and result['disposable_optimizer_steps'] == 4
    write(directory / 'decision.json', {'status': 'native_preflight_passed', 'native_preflight_started': True,
        'native_preflight_receipt_sha256': sha(native / 'receipt.json'), 'nr3d_sr3d_training_started': False,
        'time_cst': datetime.datetime.now(ZONE).isoformat()})


if __name__ == '__main__':
    main()
