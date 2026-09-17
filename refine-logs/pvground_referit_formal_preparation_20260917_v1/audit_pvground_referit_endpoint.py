"""Independent fixed-budget ReferIt endpoint and mixed-row REC audit."""
import argparse
from collections import Counter
import datetime
import json
import math
from pathlib import Path

from pvground_native_output_recount import recount_native_rows, sha


def audit_training_rows(training, partition, spec):
    total = math.ceil(len(partition['fit_ids']) / spec['batch_size'])
    assert total == spec['total_steps']
    assert [row['step'] for row in training] == list(range(1, total + 1))
    seen = [index for row in training for index in row['rows']]
    assert Counter(seen) == Counter(partition['fit_ids'])
    assert not set(seen).intersection(partition['holdout_ids'])
    assert [len(row['rows']) for row in training] == [spec['batch_size']] * (total - 1) + [len(seen) - spec['batch_size'] * (total - 1)]
    counts = dict(eligible25=0, eligible50=0, active25=0, active50=0)
    language_rows = detection_rows = 0
    for row in training:
        assert row['total_steps'] == total
        expected = [spec['dataset'] if i < partition['original_language_rows'] else 'scannet' for i in row['rows']]
        assert row['sample_dataset'] == expected
        referring = expected.count(spec['dataset'])
        language_rows += referring
        detection_rows += expected.count('scannet')
        for key in ['loss', 'loss_native', 'loss_rec_competition', 'grad_norm', 'seconds',
                    'loss_bbox', 'loss_giou', 'loss_ce', 'loss_sem_align']:
            assert math.isfinite(row[key]), (row['step'], key)
        assert row['loss_rec_competition'] >= 0
        assert math.isclose(row['loss'], row['loss_native'] + row['loss_rec_competition'], rel_tol=1e-6, abs_tol=1e-6)
        for suffix in ['25', '50']:
            assert 0 <= row['active' + suffix] <= row['eligible' + suffix] <= referring
        if referring == 0:
            assert row['loss_rec_competition'] == 0
        for key in counts:counts[key] += row[key]
    assert counts['active25'] + counts['active50'] > 0
    assert language_rows == len(partition['language_fit_ids'])
    assert detection_rows == len(partition['detection_kept_base_ids']) * partition['detection_repeats']
    return dict(competition_pair_counts=counts, language_rows=language_rows, detection_rows=detection_rows)


def audit(root):
    assert (root/'controller.exit').read_text().strip() == '0'
    spec = json.loads((root/'spec.json').read_bytes())
    receipt = json.loads((root/'receipt.json').read_bytes())
    assert spec['dataset'] in ('nr3d', 'sr3d') and receipt['dataset'] == spec['dataset']
    assert sha(spec['partition']) == spec['partition_sha256'] == receipt['partition_sha256']
    partition = json.loads(Path(spec['partition']).read_bytes())
    assert partition['dataset'] == spec['dataset']
    assert spec['batch_size'] == 8 and spec['seed'] == 2027 and spec['fit_passes'] == 1
    assert spec['lr'] == spec['lr_backbone'] == 1e-5
    assert spec['rec_competition'] and spec['competition_weight'] == 1.0
    assert receipt['status'] == 'complete' and receipt['formal_rows'] == 0
    assert receipt['primary_mode'] == spec['primary_mode'] == 'bbs'
    assert receipt['training_steps'] == spec['total_steps']
    assert receipt['fit_rows'] == spec['fit_rows'] == len(partition['fit_ids'])
    assert receipt['holdout_rows'] == spec['holdout_rows'] == len(partition['holdout_ids'])
    assert sha(root/'spec.json') == receipt['spec_sha256']
    assert sha(root/'train.py') == receipt['script_sha256']
    for name, digest in spec['files'].items():assert sha(root/name) == digest, name
    assert sha(root/'terminal.pth') == receipt['terminal_sha256']
    assert sha(root/'train.jsonl') == receipt['train_log_sha256']
    assert receipt['checkpoint_sha256'] == spec['checkpoint_sha256'] == sha(spec['checkpoint']['path'])
    assert receipt['frozen_parameters_unchanged'] and receipt['fit_seen_exactly_once']
    assert receipt['added_state_tensors'] == 37 and receipt['new_parameters'] == 923616
    for key in ['task_read', 'observation_state', 'source_query_read', 'rec_competition']:
        assert receipt[key] is spec[key] is True
    for key in ['task_module_sha256', 'observation_module_sha256', 'source_query_module_sha256',
                'competition_module_sha256', 'source_port_sha256']:
        assert receipt[key] == spec[key]
    training = [json.loads(line) for line in (root/'train.jsonl').read_text().splitlines()]
    mixed = audit_training_rows(training, partition, spec)
    stages = {};outputs = {};metrics = {}
    for stage in ['initial', 'terminal']:
        stage_receipt = json.loads((root/stage/'receipt.json').read_bytes())
        assert stage_receipt['stage'] == stage and stage_receipt['formal_rows'] == 0
        rows, values, checks = recount_native_rows(root/stage)
        assert [row['row_id'] for row in rows] == partition['holdout_ids']
        assert len(rows) == spec['holdout_rows']
        assert values == receipt[stage]
        outputs[stage] = rows;metrics[stage] = values;stages[stage] = checks
    before, after = outputs['initial'], outputs['terminal']
    transitions = {};bands = {}
    for mode in ['bbs', 'bbf']:
        transitions[mode] = {};bands[mode] = [[0]*3 for _ in range(3)]
        for old, new in zip(before, after):
            assert all(old[key] == new[key] for key in ['row_id', 'scan_id', 'target_id', 'point_sha256', 'root_box'])
            bands[mode][int(old[mode]['iou']>.25)+int(old[mode]['iou']>.5)][int(new[mode]['iou']>.25)+int(new[mode]['iou']>.5)] += 1
        for threshold in [.25, .5]:
            fixes = sum(old[mode]['iou'] <= threshold < new[mode]['iou'] for old, new in zip(before, after))
            breaks = sum(new[mode]['iou'] <= threshold < old[mode]['iou'] for old, new in zip(before, after))
            transitions[mode][str(threshold)] = dict(fixes=fixes, breaks=breaks, net=fixes-breaks)
    assert transitions == receipt['transitions']
    nonregression = all(transitions['bbs'][str(t)]['net'] >= 0 for t in [.25, .5])
    assert nonregression == receipt['primary_rec_nonregression']
    return dict(integrity_pass=True, dataset=spec['dataset'], time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        formal_rows=0, fit_rows=receipt['fit_rows'], holdout_rows=receipt['holdout_rows'], training_steps=receipt['training_steps'],
        primary_mode='bbs', initial=metrics['initial'], terminal=metrics['terminal'], stages=stages,
        transitions=transitions, transition_counts=bands, rec_competition_verified=True, mixed_rows=mixed,
        primary_rec_nonregression=nonregression, candidate_for_fixed_formal_evaluation=nonregression,
        mask_gate=False, mask_audit_scope='recount exported per-row IoU; no binary mask reload',
        new_scene_generalization_claim=False, receipt_sha256=sha(root/'receipt.json'),
        auditor_sha256=sha(__file__), recount_sha256=sha(Path(__file__).parent/'pvground_native_output_recount.py'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root)
    with args.out.open('x') as f:json.dump(result, f, indent=2, allow_nan=False);f.write('\n')
    print(json.dumps(result), flush=True)
