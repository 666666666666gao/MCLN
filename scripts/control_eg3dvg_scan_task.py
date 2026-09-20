"""Run the fixed ScanRefer task/native campaign; never dispatch Nr3D or Sr3D."""
import fcntl
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parent
    campaign = json.loads((root / 'campaign.json').read_text())
    assert sha(__file__) == campaign['controller_sha256']
    for name, digest in campaign['scripts'].items():
        assert sha(root / name) == digest, name
    waiting = Path(campaign['wait_for'])
    while not (waiting / 'controller.exit').exists():
        time.sleep(300)
    lock = open('/root/autodl-tmp/mcln_v99_backbone_gpu0.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader']).decode().strip()
    assert shutil.disk_usage('/root').free > 4_000_000_000

    def run(name, argv, cwd):
        with (root / (name + '.log')).open('xb') as stream:
            code = subprocess.call([sys.executable, '-u'] + argv, cwd=cwd,
                                   stdout=stream, stderr=subprocess.STDOUT)
        (root / (name + '.exit')).write_text(str(code) + '\n')
        if code:
            (root / 'controller.exit').write_text(str(code) + '\n')
            raise SystemExit(code)

    specs = {arm: json.loads((root / arm / 'spec.json').read_text()) for arm in ['task', 'native']}
    source = specs['task']['source']
    run('initial_equality', [str(root / 'initial.py'), '--spec', str(root / 'task/spec.json')], source)
    assert json.loads((root / 'task/initial_equality.json').read_text())['status'] == 'pass'
    for arm in ['task', 'native']:
        run(arm + '_preflight', [str(root / 'train.py'), '--spec', str(root / arm / 'spec.json'),
                                 '--stage', 'preflight'], source)
    reference = Path(campaign['reference_root'])
    with gzip.open(reference / 'formal/rows.jsonl.gz', 'rt') as f:
        original = [json.loads(line) for line in f]
    metrics = []
    for epoch in range(1, campaign['max_epochs'] + 1):
        outputs = {}
        for arm in ['task', 'native']:
            label = '%s_epoch_%02d' % (arm, epoch)
            spec = specs[arm]
            run(label, [str(root / 'train.py'), '--spec', str(root / arm / 'spec.json'),
                        '--stage', 'fit', '--epoch', str(epoch)], source)
            fit = json.loads((root / arm / ('epoch_%02d' % epoch) / 'receipt.json').read_text())
            assert fit['optimizer_steps'] == spec['fit_updates'] and fit['rows'] == spec['fit_rows']
            evaluation = root / arm / ('evaluation_%02d' % epoch)
            evaluation.mkdir()
            es = json.loads((reference / 'spec.json').read_text())
            es.update(checkpoint=fit['checkpoint_path'], checkpoint_sha256=fit['checkpoint_sha256'],
                      checkpoint_optimizer_steps=fit['total_fit_steps'], task_read=spec['task_read'],
                      source=source, source_files=spec['source_files'],
                      training_spec_sha256=fit['spec_sha256'],
                      evaluator_sha256=sha(root / 'evaluate.py'), auditor_sha256=sha(root / 'audit.py'))
            (evaluation / 'spec.json').write_text(json.dumps(es, indent=2))
            shutil.copyfile(reference / 'annotation_manifest.json', evaluation / 'annotation_manifest.json')
            for stage in ['preflight', 'formal']:
                run(label + '_eval_' + stage, [str(root / 'evaluate.py'), '--spec', str(evaluation / 'spec.json'),
                                               '--stage', stage], source)
            run(label + '_audit', [str(root / 'audit.py'), '--root', str(evaluation)], source)
            receipt = json.loads((evaluation / 'formal/receipt.json').read_text())
            with gzip.open(evaluation / 'formal/rows.jsonl.gz', 'rt') as f:
                rows = [json.loads(line) for line in f]
            for left, right in zip(original, rows):
                for key in ['row_id', 'scan_id', 'target_id', 'utterance', 'point_sha256', 'gt_box']:
                    assert left[key] == right[key], (arm, epoch, key, left['row_id'])
            outputs[arm] = dict(receipt=receipt, rows=rows)
            print('SCAN_FORMAL_RESULT ' + json.dumps(dict(arm=arm, epoch=epoch, metrics=receipt['metrics'])), flush=True)
        # Identical augmented point inputs, not just the same seed/order.
        logs = []
        for arm in ['task', 'native']:
            logs.append([json.loads(line) for line in (root / arm / ('epoch_%02d' % epoch) / 'updates.jsonl').read_text().splitlines()])
        assert len(logs[0]) == len(logs[1]) == specs['task']['fit_updates']
        for a, b in zip(*logs):
            for key in ['step', 'rows', 'scan_ids', 'point_sha256']:
                assert a[key] == b[key], (epoch, a['step'], key)
        report = {'epoch': epoch, 'same_training_points': True, 'same_formal_inputs': True,
                  'task': outputs['task']['receipt']['metrics']['bbs'],
                  'native': outputs['native']['receipt']['metrics']['bbs'], 'paired': {}}
        for label, before in [('pretrained', original), ('native', outputs['native']['rows'])]:
            report['paired'][label] = {}
            for threshold in [.25, .5]:
                repairs = sum(a['bbs']['iou'] <= threshold < b['bbs']['iou'] for a, b in zip(before, outputs['task']['rows']))
                breaks = sum(b['bbs']['iou'] <= threshold < a['bbs']['iou'] for a, b in zip(before, outputs['task']['rows']))
                report['paired'][label][str(threshold)] = dict(repairs=repairs, breaks=breaks, net=repairs-breaks)
        report['beats_pretrained_both'] = report['task']['rec_hits25'] > 5542 and report['task']['rec_hits50'] > 4952
        report['beats_same_budget_native_both'] = all(report['task'][key] > report['native'][key] for key in ['rec_hits25','rec_hits50'])
        metrics.append(report)
        (root / 'epoch_results.json').write_text(json.dumps(metrics, indent=2))
        print('SCAN_PAIRED_RESULT ' + json.dumps(report), flush=True)
        if report['beats_pretrained_both']:
            break
    decision = {'status': 'complete', 'epochs': len(metrics), 'scanrefer_gate_pass': metrics[-1]['beats_pretrained_both'],
                'same_budget_innovation_gain': metrics[-1]['beats_same_budget_native_both'],
                'nr3d_sr3d_training_started': False,
                'next_action': 'review_scan_result_before_transfer' if metrics[-1]['beats_pretrained_both'] else 'continue_scan_research_with_recorded_negative_results'}
    (root / 'decision.json').write_text(json.dumps(decision, indent=2))
    (root / 'controller.exit').write_text('0\n')


if __name__ == '__main__':
    main()
