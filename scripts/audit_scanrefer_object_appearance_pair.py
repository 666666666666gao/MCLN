"""Independently recount the fixed appearance endpoints and their protected state."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    root = parser.parse_args().root.resolve()
    config = json.loads((root / 'input_manifest.json').read_text())
    for item in config['artifacts'].values():
        assert sha(item['path']) == item['sha256']
    source_manifest = Path(config['model_source']) / 'appearance_source_manifest.json'
    assert sha(source_manifest) == config['source_manifest_sha256']
    for name, digest in json.loads(source_manifest.read_text())['files'].items():
        assert sha(Path(config['model_source']) / name) == digest, name
    assert (root / 'controller.exit').read_text().strip() == '0'
    receipt = json.loads((root / 'receipt.json').read_text())
    assert receipt['status'] == 'complete' and receipt['steps_per_arm'] == 2482
    assert receipt['manifest_sha256'] == sha(root / 'input_manifest.json')
    assert receipt['schema'] == 'mcln-scanrefer-object-appearance-pair-v1'
    for field, file in [('baseline_rows_sha256', 'baseline_rows.json'),
                        ('terminal_rows_sha256', 'terminal_rows.json'),
                        ('fit_batches_sha256', 'fit_point_batches.json')]:
        assert receipt[field] == sha(root / file)
    protocol = json.loads((root / 'protocol.json').read_text())
    assert sha(config['split_protocol']) == config['split_protocol_sha256']
    assert protocol['row_ids'] == json.loads(Path(config['split_protocol']).read_text())['row_ids']
    fit = protocol['row_ids']['fit']
    holdout = protocol['row_ids']['holdout']
    assert len(fit) == 29778 and len(holdout) == 6887 and not set(fit) & set(holdout)
    batches = json.loads((root / 'fit_point_batches.json').read_text())
    assert [v['step'] for v in batches] == list(range(1, 2483))
    assert Counter(i for v in batches for i in v['row_ids']) == Counter(fit)
    baseline = json.loads((root / 'baseline_rows.json').read_text())
    terminal = json.loads((root / 'terminal_rows.json').read_text())
    assert baseline['control'] == baseline['appearance']
    metrics = {}
    for stage, rows_by_arm in [('baseline', baseline), ('terminal', terminal)]:
        metrics[stage] = {}
        for arm, rows in rows_by_arm.items():
            assert [v['row_id'] for v in rows] == holdout
            metrics[stage][arm] = dict(rec025=sum(v['rec_iou'] > .25 for v in rows),
                                       rec050=sum(v['rec_iou'] > .5 for v in rows),
                                       mask025=sum(v['mask_iou'] > .25 for v in rows),
                                       mask050=sum(v['mask_iou'] > .5 for v in rows),
                                       mask_miou=sum(v['mask_iou'] for v in rows) / len(rows) * 100)
            for before, after in zip(baseline['control'], rows):
                for key in ['row_id', 'scan_id', 'physical_space', 'point_sha256']:
                    assert before[key] == after[key], key
    comparisons = {}
    for reference, before in [('baseline', baseline['control']), ('control', terminal['control'])]:
        comparisons[reference] = {}
        for threshold in [.25, .5]:
            repairs = sum(a['rec_iou'] > threshold and b['rec_iou'] <= threshold for b, a in zip(before, terminal['appearance']))
            breaks = sum(a['rec_iou'] <= threshold and b['rec_iou'] > threshold for b, a in zip(before, terminal['appearance']))
            value = dict(repair=repairs, damage=breaks, net=repairs-breaks)
            assert value == receipt['appearance_rec_effects'][reference][str(threshold)]
            comparisons[reference][str(threshold)] = value

    import torch
    sys.path.insert(0, config['model_source'])
    import scripts
    scripts.__path__ = [str(root / 'scripts')] + list(scripts.__path__)
    from scripts.audit_scanrefer_local_visual_pair import check_readouts
    initial = {k[7:]: v for k, v in torch.load(config['artifacts']['backbone']['path'], map_location='cpu')['model'].items()}
    assert len(initial) == 1144
    core_names = protocol['core_trainable_tensors']
    assert all(n.startswith(('cross_encoder.', 'decoder.', 'prediction_heads.')) for n in core_names)
    protected = {k: torch.load(v['path'], map_location='cpu') for k, v in config['artifacts'].items() if k != 'backbone'}
    states = {}
    for arm in ['control', 'appearance']:
        item = receipt['checkpoints'][arm]
        assert sha(item['path']) == item['sha256']
        checkpoint = torch.load(item['path'], map_location='cpu')
        assert checkpoint['steps'] == 2482 and checkpoint['manifest_sha256'] == sha(root / 'input_manifest.json')
        actual = checkpoint['model']
        extra = ['object_appearance.projection.weight'] if arm == 'appearance' else []
        assert set(actual) == set(initial) | set(extra)
        changes = []
        for name, value in actual.items():
            assert torch.isfinite(value).all(), name
            if name in extra:
                assert value.shape == (160, 1280) and value.abs().sum() > 0
            else:
                same = torch.equal(value, initial[name])
                if name not in core_names:
                    assert same, name
                elif not same:
                    changes.append(name)
        assert changes
        check_readouts(checkpoint['readout'], protected)
        optimizer = checkpoint['optimizer']
        assert [g['lr'] for g in optimizer['param_groups']] == ([1e-6, 1e-4] if extra else [1e-6])
        names = core_names + extra
        for index, state in optimizer['state'].items():
            assert 0 < float(state['step']) <= 2482
            for field in ['exp_avg', 'exp_avg_sq']:
                assert state[field].shape == actual[names[index]].shape and torch.isfinite(state[field]).all()
        states[arm] = dict(changed_original_tensors=len(changes), original_frozen_state_exact=True,
                           readouts_exact=True, optimizer_state_tensors=len(optimizer['state']))
    eligible = all(v['net'] >= 0 for effect in comparisons.values() for v in effect.values())
    assert eligible == receipt['development_dual_rec_nonregression']
    result = dict(integrity_pass=True, module_rec_screen_pass=eligible, comparisons=comparisons,
                  metrics=metrics, states=states, formal_rows=0,
                  previous_backbone_saw_module_holdout=True,
                  scope='Fixed module-holdout endpoint audit. Formal Scan qualification and Mask floors remain separate.')
    with (root / 'audit.json').open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
