"""Compare audited task-read REC to the pretrained reference and native control."""
import argparse
import hashlib
import json
from pathlib import Path

from analyze_eg3dvg_nr3d_adaptation import read_rows


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def paired(first, before, last, after):
    assert len(before) == len(after) == 7899
    for a, b in zip(before, after):
        for key in ['row_id', 'scan_id', 'target_id', 'utterance', 'point_sha256', 'gt_box', 'object_boxes']:
            assert a[key] == b[key], (a['row_id'], key)
    values = {}
    for mode in ['bbs', 'bbf', 'bbs_unfiltered', 'bbf_unfiltered']:
        values[mode] = {}
        for threshold, key in [(.25, 'rec_hits25'), (.5, 'rec_hits50')]:
            repaired = [a['row_id'] for a, b in zip(before, after) if a[mode]['iou'] <= threshold < b[mode]['iou']]
            broken = [a['row_id'] for a, b in zip(before, after) if b[mode]['iou'] <= threshold < a[mode]['iou']]
            first_hits, last_hits = first['metrics'][mode][key], last['metrics'][mode][key]
            assert last_hits - first_hits == len(repaired) - len(broken)
            values[mode][str(threshold)] = dict(reference_hits=first_hits, candidate_hits=last_hits,
                repairs=len(repaired), breaks=len(broken), net=last_hits-first_hits,
                repair_row_ids=repaired, break_row_ids=broken)
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    spec = json.loads((root / 'spec.json').read_text())
    control = Path(spec['control_root'])
    control_spec = json.loads((control / 'spec.json').read_text())
    for key in ['checkpoint_sha256', 'seed', 'batch_size', 'fit_rows', 'fit_updates',
                'order_path', 'input_hashes', 'lr', 'lr_backbone', 'text_encoder_lr',
                'schedule', 'clip_norm', 'weight_decay', 'environment_sha256']:
        assert spec[key] == control_spec[key], key
    for experiment in [root, control]:
        fit = json.loads((experiment / 'fit/receipt.json').read_text())
        result = json.loads((experiment / 'evaluation/formal/receipt.json').read_text())
        audit = json.loads((experiment / 'evaluation/formal/audit.json').read_text())
        assert fit['optimizer_steps'] == 5614 and fit['rows'] == 44909
        assert fit['parent_checkpoint_sha256'] == spec['checkpoint_sha256']
        assert fit['checkpoint_sha256'] == result['checkpoint_sha256']
        assert fit['spec_sha256'] == sha(experiment / 'spec.json')
        assert audit['receipt_sha256'] == sha(experiment / 'evaluation/formal/receipt.json')
    initial = json.loads((root / 'initial_equality.json').read_text())
    assert initial['status'] == 'pass' and initial['spec_sha256'] == sha(root / 'spec.json')
    a = [json.loads(line) for line in (control / 'fit/updates.jsonl').read_text().splitlines()]
    b = [json.loads(line) for line in (root / 'fit/updates.jsonl').read_text().splitlines()]
    assert len(a) == len(b) == 5614
    for left, right in zip(a, b):
        for key in ['step', 'rows', 'scan_ids']:
            assert left[key] == right[key], (left['step'], key)
    start, start_rows = read_rows(Path(spec['transfer_root']))
    native, native_rows = read_rows(control / 'evaluation')
    candidate, candidate_rows = read_rows(root / 'evaluation')
    assert start['checkpoint_sha256'] == spec['checkpoint_sha256']
    report = dict(status='complete', rows=7899, primary_mode='bbs', model_forwards=0,
                  optimizer_steps=0, same_training_recipe=True, same_training_scene_order=True,
                  identical_formal_inputs=True, candidate_checkpoint_sha256=candidate['checkpoint_sha256'],
                  native_checkpoint_sha256=native['checkpoint_sha256'],
                  versus_pretrained_reference=paired(start, start_rows, candidate, candidate_rows),
                  versus_native_control=paired(native, native_rows, candidate, candidate_rows))
    report['project_target_met'] = (candidate['metrics']['bbs']['rec_hits25'] >= 4726
                                    and candidate['metrics']['bbs']['rec_hits50'] >= 4059)
    report['claim_boundary'] = ('Only the native-control comparison isolates this fixed task-read adaptation. '
        'Pretrained-reference gains include ordinary training. Formal inputs match exactly; training '
        'checks bind recipe and scene order, not a per-step point hash. Single seed and Nr only. '
        'The protocol still uses GT scene object inputs; unfiltered is not GT-free.')
    with (root / 'paired_rec.json').open('x') as f:
        json.dump(report, f, indent=2)
    print(json.dumps({key: {threshold: {k: v for k, v in value.items() if not k.endswith('_ids')}
                           for threshold, value in report[key]['bbs'].items()}
                      for key in ['versus_pretrained_reference', 'versus_native_control']}), flush=True)


if __name__ == '__main__':
    main()
