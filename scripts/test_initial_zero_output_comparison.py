"""Regression on the archived real failed E0, plus the enforced invariants."""
import copy
import json
from pathlib import Path
from initial_zero_output_comparison import compare_initial_rows

root=Path(__file__).parent
def read(path):return [json.loads(line) for line in path.read_text().splitlines()]
actual=read(root/'failed_tail_fused_initial/arm/initial/rows.jsonl')
reference=read(root/'complete_tail_raw/arm/initial/rows.jsonl')
result=compare_initial_rows(actual,reference)
assert not result['cross_process_rec_output_exact']
assert result['cross_process_differences']['bbs']['query_row_ids']==[26603]
assert result['cross_process_differences']['bbf']['query_row_ids']==[26641]
assert result['input_identities_exact'] and result['rec_threshold_decisions_exact']
for changed_key in ('input','zero_output','threshold'):
    changed=copy.deepcopy(actual)
    if changed_key=='input':changed[0]['point_sha256']='changed'
    elif changed_key=='zero_output':changed[0]['bbs']['coarse_box'][0]+=1
    else:
        value=0. if reference[0]['bbs']['iou']>.5 else 1.
        changed[0]['bbs']['iou']=changed[0]['bbs']['coarse_iou']=value
    rejected=False
    try:
        compare_initial_rows(changed,reference)
    except AssertionError:
        rejected=True
    assert rejected,changed_key
print(json.dumps(dict(status='pass',real_failed_rows=6887,no_numerical_tolerance=True,
    detects_changed_input=True,detects_nonzero_refinement=True,detects_changed_threshold=True,
    functional_output_bit_identity=False)))
