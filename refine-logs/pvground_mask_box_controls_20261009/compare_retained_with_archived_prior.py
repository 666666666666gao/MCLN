"""A row-aligned hypothesis check; this is explicitly NOT a same-forward result."""
import datetime
import json
from pathlib import Path

import numpy as np

from analyze_fixed_query_boxes import compare, digest, iou


ROOT = Path(__file__).resolve().parent
CURRENT = Path(r'C:\Users\gb\.codex\tmp\pvground_compressed_geometry_support_20261008')
OLD = Path(r'C:\Users\gb\.codex\tmp\pvground_mask_reference_20261006')
N = 9508


def main():
    source_path = CURRENT / 'complete_fit' / 'formal' / 'rows.jsonl'
    receipt_path = source_path.parent / 'receipt.json'
    receipt = json.loads(receipt_path.read_bytes())
    assert receipt['status'] == 'pass' and receipt['rows'] == receipt['formal_rows'] == N
    assert digest(source_path) == receipt['rows_sha256']
    old_spec = json.loads((OLD / 'fused_mask_reference_spec.json').read_bytes())
    current_spec = json.loads((CURRENT / 'pair_spec.json').read_bytes())
    for key in ('base_terminal_sha256', 'checkpoint_sha256', 'seed', 'batch_size', 'reference_mode'):
        assert old_spec[key] == current_spec[key]
    archived = [json.loads(line) for line in (ROOT / 'SELECTED_BOX_ROWS.jsonl').read_text(encoding='utf-8').splitlines()]
    current = [json.loads(line) for line in source_path.read_text(encoding='utf-8').splitlines()]
    assert len(current) == len(archived) == N
    for older, newer in zip(archived, current):
        for key in ('row_id', 'scan_id', 'target_id', 'query', 'root_box', 'point_sha256'):
            assert older[key] == newer[key]
        # The original uncorrected Mask extent agrees exactly in the two passes.
        assert older['boxes']['mask_reference'] == newer['parent_box']
    truth = np.asarray([row['root_box'] for row in current], dtype=np.float32)
    native = np.asarray([row['boxes']['native_regression'] for row in archived], dtype=np.float32)
    corrected = np.asarray([row['arms']['content']['box'] for row in current], dtype=np.float32)
    blend = (native + corrected) * np.float32(.5)
    corrected_iou = iou(corrected, truth, np.float32)
    blend_iou = iou(blend, truth, np.float32)
    corrected64 = iou(corrected, truth, np.float64)
    blend64 = iou(blend, truth, np.float64)
    assert [int((corrected_iou > t).sum()) for t in (.25, .5)] == [5599, 4859]
    result = dict(status='EXECUTED_ROW_ALIGNED_CROSS_FORWARD_GEOMETRY_HYPOTHESIS_ONLY',
        generated_cst=datetime.datetime.now().astimezone().isoformat(), rows=N,
        retained_checkpoint_sha256='6f9109cb690399880b8dc3bcf6e4b98d1053145732ee2a6b504bfaecbd1eaac2',
        source_current_rows_sha256=digest(source_path), source_current_receipt_sha256=digest(receipt_path),
        source_archived_control_rows_sha256=digest(ROOT / 'SELECTED_BOX_ROWS.jsonl'),
        source_script_sha256=digest(Path(__file__)),
        exact_row_input_gt_query_alignment=N, exact_uncorrected_mask_extent_alignment=N,
        current_native_coarse_box_not_saved=True, same_forward_comparison=False,
        conditions={}, comparison=compare(corrected_iou, blend_iou),
        coefficient_search=False, neural_forwards=0, optimizer_updates=0, ssh_queries=0,
        retained_best_changed=False, source_only_hypothesis_not_deployment_metric=True,
        full_goal_complete=False, three_effective_contributions=False,
        limits=[
            'The regression prior comes from October 6, while corrected Mask extents come from the October 8 retained state.',
            'Inputs, root GT, native selected Query and uncorrected Mask extents align exactly for all9508 rows.',
            'Exact agreement of saved Mask extents does not prove bitwise identity of unsaved current native coarse boxes.',
            'Frozen parent GPU repeat drift has already been observed in this project; this arithmetic cannot promote a new complete model.',
            'Any promising output requires a new serial same-forward full-validation execution after the active fit closes.',
            'The fixed 50/50 formula is a named prior control, not a novel contribution or a restored V99 ranking chain.'
        ])
    for name, values, values64 in (('retained_pure_mask', corrected_iou, corrected64),
                                   ('archived_prior_half_blend_hypothesis', blend_iou, blend64)):
        hits = [int((values > t).sum()) for t in (.25, .5)]
        result['conditions'][name] = dict(hits=hits, percentages=[h / N * 100 for h in hits],
            float64_hits=[int((values64 > t).sum()) for t in (.25, .5)],
            float32_float64_threshold_flips=[int(((values > t) != (values64 > t)).sum()) for t in (.25, .5)],
            numerical_gate_only=hits[0] >= 5658 and hits[1] >= 4850)
    output = ROOT / 'RETAINED_CROSS_FORWARD_HYPOTHESIS.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
