"""Independently check a terminal R1 artifact against pinned inputs and CSV IDs."""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sys


def file_sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--addon', type=Path, required=True)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.addon / 'input_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    source = Path(manifest['model_source'])
    os.chdir(str(source))
    sys.path.insert(0, str(source))
    import torch
    import scripts
    scripts.__path__ = [str(args.addon / 'scripts')] + list(scripts.__path__)
    from scripts.nr3d_reference_memory_contract import ARMS, CONTRACT, decide, split_rows
    from scripts.nr3d_view_pair_contract import CHECKPOINT_SHA, digest_ids
    from scripts.run_nr3d_view_pair_role import read_train_rows

    assert (args.addon / 'train.exit').read_text().strip() == '0'
    assert manifest['contract'] == CONTRACT
    for relative, expected in manifest['files'].items():
        assert file_sha(args.addon / relative) == expected, relative
    source_manifest = source / 'g0_source_manifest.json'
    assert file_sha(source_manifest) == manifest['source_manifest_sha256']
    for relative, expected in json.loads(source_manifest.read_text())['files'].items():
        assert file_sha(source / relative) == expected, relative

    run = args.addon / 'results'
    receipt = json.loads((run / 'receipt.json').read_text())
    assert receipt['status'] == 'complete'
    assert receipt['checkpoint_sha256'] == CHECKPOINT_SHA
    assert receipt['input_manifest_sha256'] == file_sha(manifest_path)
    assert receipt['contract'] == CONTRACT and receipt['census'] == manifest['census']
    for flag in ['protected_evaluator_row_parity', 'protected_state_unchanged',
                 'backbone_gradients_absent', 'common_initial_state_equal']:
        assert receipt[flag], flag
    assert not receipt['formal_validation_dataset_constructed']
    assert file_sha(run / 'holdout_rows.jsonl') == receipt['holdout_rows_sha256']
    rows = [json.loads(line) for line in (run / 'holdout_rows.jsonl').read_text().splitlines()]
    raw_rows = read_train_rows(Path('/root/autodl-tmp/DATA_ROOT'))
    split = split_rows(raw_rows)
    assert [row['id'] for row in rows] == split['holdout']
    assert len(rows) == len({row['id'] for row in rows}) == 6172
    assert len({row['scan_id'] for row in rows}) == 98
    assert digest_ids(row['id'] for row in rows) == manifest['census']['holdout']['identity_sha256']
    for row in rows:
        csv_row = raw_rows[row['id']]
        assert row['scan_id'] == csv_row['scan_id']
        assert row['target_id'] == int(csv_row['target_id'])
        assert row['raw_token_count'] == len(ast.literal_eval(csv_row['tokens']))

    training = receipt['training']
    assert training == json.loads((run / 'training.json').read_text())
    assert training['sample_count'] == 26747
    assert training['sample_order_sha256'] == '285ea28b72d7a88a26251a1d92471b50aa726aa6eae725c8822fe0a26271ca7b'
    assert training['optimizer_steps'] == {mode: 6687 - training['skipped_batches'] for mode in ARMS}
    for mode, weights in training['weights'].items():
        path = run / weights['name']
        assert file_sha(path) == weights['sha256']
        state = torch.load(str(path), map_location='cpu')
        assert state['mode'] == mode and state['contract'] == CONTRACT
        assert state['backbone_sha256'] == CHECKPOINT_SHA
        assert all(torch.isfinite(value).all() for value in state['model'].values())
    assert set(training['weights']) == set(ARMS)
    decision = decide(rows)
    assert decision == receipt['decision'] == json.loads((run / 'decision.json').read_text())
    assert not decision['formal_promotion']
    assert not decision['control_substituted_for_failed_primary_candidate']
    analysis = json.loads(args.analysis.read_text())
    assert analysis['decision'] == decision
    assert analysis['holdout_rows_sha256'] == receipt['holdout_rows_sha256']
    for mode, values in analysis['groups']['overall']['scores'].items():
        for metric in ['rec_hits025', 'rec_hits050', 'mask_hits025', 'mask_hits050']:
            assert values[metric] == receipt['summary'][mode][metric]
        assert values['mask_iou_sum'] / len(rows) == receipt['summary'][mode]['mask_mean_iou']
    result = {'schema': 'mcln-r1-terminal-verification-v1', 'status': 'pass',
              'receipt_sha256': file_sha(run / 'receipt.json'),
              'analysis_sha256': file_sha(args.analysis),
              'holdout_rows_sha256': receipt['holdout_rows_sha256'],
              'csv_identity_order_and_raw_token_lengths_match': True,
              'fit_order_matches_fixed_P2_order': True, 'all_four_update_counts_match': True,
              'all_four_final_head_hashes_and_finiteness_pass': True,
              'source_and_addon_manifest_pass': True, 'formal_promotion': False,
              'new_gpu_forwards': 0, 'new_optimizer_steps': 0}
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
