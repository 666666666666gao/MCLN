"""Describe the existing same-input E0/E3 panel; never run the model."""
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np


root = Path(__file__).resolve().parent
panel = root / 'actual_panel/results'
original = json.loads((root / 'PANEL_RESULT.json').read_bytes())
receipt = json.loads((panel / 'PANEL_RECEIPT.json').read_bytes())
expected = {entry['name']: entry for entry in original['files']}
input_hashes = {}
for name in ('PANEL_RECEIPT.json', 'best_rows.json', 'latest_rows.json',
             'best_panel.npz', 'latest_panel.npz'):
    path = panel / name
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    assert len(raw) == expected['results/' + name]['bytes']
    assert digest == expected['results/' + name]['sha256']
    input_hashes[str(path)] = digest
assert receipt['same_cached_inputs'] and receipt['same_inference_random_states']
assert not receipt['formal_full9508_result']


def transitions(before, after, threshold):
    good_before = before > threshold
    good_after = after > threshold
    repaired = int((~good_before & good_after).sum())
    damaged = int((good_before & ~good_after).sum())
    return dict(before_hits=int(good_before.sum()), after_hits=int(good_after.sum()),
                repaired=repaired, damaged=damaged, net=repaired - damaged)


def distribution(values):
    return dict(mean=float(values.mean()),
                quantiles=dict(zip(('min', 'q25', 'median', 'q75', 'max'),
                                   map(float, np.quantile(values, [0, .25, .5, .75, 1])))),
                zero_count=int((values == 0).sum()),
                one_count=int((values == 1).sum()), total_values=int(values.size))


data = {}
summaries = []
for index, name in enumerate(('best', 'latest')):
    rows = json.loads((panel / (name + '_rows.json')).read_bytes())
    with np.load(panel / (name + '_panel.npz'), allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    assert len(rows) == 64
    assert all(value.shape[:2] == (64, 256) for value in arrays.values())
    selected = np.asarray([row['selected_query'] for row in rows], dtype=np.int64)
    selected_values = {key: value[np.arange(64), selected] for key, value in arrays.items()}
    for branch in ('native', 'reference', 'final'):
        np.testing.assert_array_equal(selected_values[branch + '_iou'],
                                      [row['selected_' + branch + '_iou'] for row in rows])
    np.testing.assert_array_equal(selected_values['axis_gate'],
                                  [row['selected_axis_gate'] for row in rows])
    assert (arrays['bbs_score'].max(1) == selected_values['bbs_score']).all()
    gate = arrays['axis_gate']
    assert gate.shape == (64, 256, 3) and ((gate >= 0) & (gate <= 1)).all()
    np.testing.assert_array_equal(gate, arrays['raw_axis_gate'].clip(0, 1))
    # Coupled center/size interpolation is the same interpolation of both faces.
    expected_box = ((1 - np.concatenate([gate, gate], -1)) * arrays['reference_box']
                    + np.concatenate([gate, gate], -1) * arrays['native_box'])
    maximum_reconstruction_error = float(np.abs(expected_box - arrays['final_box']).max())
    assert maximum_reconstruction_error < 1e-5
    selected_counts = {}
    for branch in ('native', 'reference', 'final'):
        selected_counts[branch] = {
            str(threshold): int((selected_values[branch + '_iou'] > threshold).sum())
            for threshold in (.25, .5)}
    assert selected_counts['final']['0.5'] == receipt['results'][index]['selected_hits050']
    own_good = selected_values['own_mask_iou'] > .5
    fused_good = selected_values['fused_mask_iou'] > .5
    final_good = selected_values['final_iou'] > .5
    reference_good = selected_values['reference_iou'] > .5
    summary = dict(checkpoint=name, saved_epoch=receipt['results'][index]['saved_epoch'],
        rows=64, selected_hits=selected_counts,
        same_selected_query_reference_to_final={str(t): transitions(
            selected_values['reference_iou'], selected_values['final_iou'], t) for t in (.25, .5)},
        same_selected_query_native_to_final={str(t): transitions(
            selected_values['native_iou'], selected_values['final_iou'], t) for t in (.25, .5)},
        selected_own_and_fused_mask_good_final_box_bad=int((own_good & fused_good & ~final_good).sum()),
        selected_fused_mask_good_reference_box_bad=int((fused_good & ~reference_good).sum()),
        full256_qualified_rows={branch: int((arrays[branch + '_iou'].max(1) > .5).sum())
                                for branch in ('native', 'reference', 'final')},
        selected_axis_gate=distribution(selected_values['axis_gate']),
        all_candidate_axis_gate=distribution(gate),
        all_candidate_raw_gate_negative_count=int((arrays['raw_axis_gate'] < 0).sum()),
        maximum_saved_box_reconstruction_error=maximum_reconstruction_error)
    summaries.append(summary)
    data[name] = dict(rows=rows, arrays=arrays, selected=selected, values=selected_values)

assert [(row['panel_index'], row['scan_id'], row['target_id'], row['utterance'])
        for row in data['best']['rows']] == [
        (row['panel_index'], row['scan_id'], row['target_id'], row['utterance'])
        for row in data['latest']['rows']]
paired = dict(selected_query_slot_changed_rows=int(
    (data['best']['selected'] != data['latest']['selected']).sum()),
    selected_final_transitions={str(t): transitions(
        data['best']['values']['final_iou'], data['latest']['values']['final_iou'], t)
        for t in (.25, .5)})
result = dict(status='EXISTING_NORMAL_E0_E3_PANEL_LOCAL_GEOMETRY_ANALYSIS_COMPLETE',
    recorded_cst=datetime.datetime.now().astimezone().isoformat(),
    input_hashes=input_hashes, checkpoints=summaries, paired_panel=paired,
    formal_full9508_result=False, fixed_panel_representative=False,
    equal_query_slot_proves_physical_instance_identity=False,
    reordering_or_counterfactual_predictions=False, actual_new_neural_forwards=0,
    new_optimizer_updates=0, GPU_calls=0, remote_queries=0,
    inference_or_active_source_changes=0,
    limitations=[
        'The fixed64 panel cannot explain the full9508 normal-training REC decline.',
        'Within-checkpoint reference-to-final comparisons fix the saved selected Query.',
        'Between-checkpoint choices can differ; equal Query slots do not establish identity.',
        'Gate changes are observations, not proof of a causal geometry-fusion failure.'])
output = root / 'CACHED_GEOMETRY_ANALYSIS_20261011.json'
output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(result, ensure_ascii=False))
