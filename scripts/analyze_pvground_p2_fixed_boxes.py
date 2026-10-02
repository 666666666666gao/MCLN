"""CPU recount of sealed fixed-frame outputs; no model forward or update."""
import gzip
import hashlib
import json
import math
from pathlib import Path

root = Path(r'C:\Users\gb\.codex\tmp\pvg_p2_fixed_boxes_20261002\complete')
assert (root/'INTAKE.json').is_file()
assert not (root/'SUMMARY.json').exists()
receipt = json.loads((root/'full/receipt.json').read_bytes())
reference_path = Path(r'C:\Users\gb\.codex\tmp\pvground_g_p2_20261002\complete\g_p2\formal\rows.jsonl')
reference = [json.loads(line) for line in reference_path.read_text().splitlines()]
assert len(reference) == receipt['rows'] == 9508
counts = {mode: {'hits25':0,'hits50':0,'oracle25':[0]*4,'oracle50':[0]*4}
          for mode in ('actual','bypass')}
transitions = {str(t): {'repairs':0,'damages':0,'net':0} for t in (.25,.5)}
changed = 0
max_iou_difference = 0.0
threshold_mismatches = {str(t):0 for t in (.25,.5)}
examples = []
row_identity = hashlib.sha256()

def iou_from_box(box, gt):
    overlap = [max(0.0, min(box[k]+box[k+3]/2, gt[k]+gt[k+3]/2)
                   -max(box[k]-box[k+3]/2, gt[k]-gt[k+3]/2)) for k in range(3)]
    intersection = math.prod(overlap)
    return intersection/(math.prod(box[3:])+math.prod(gt[3:])-intersection)

with gzip.open(root/'full/rows.jsonl.gz','rt',encoding='utf-8') as stream:
    for index, line in enumerate(stream):
        row = json.loads(line)
        old = reference[index]
        assert row['row_id'] == old['row_id'] == index
        assert row['point_sha256'] == old['point_sha256']
        assert row['root_box'] == old['root_box']
        assert len(row['boxes']) == len(row['ious']) == 256
        assert len(row['root_box']) == 6 and all(math.isfinite(v) for v in row['root_box'])
        assert all(v > 0 for v in row['root_box'][3:])
        row_identity.update((str(index)+':'+row['point_sha256']+'\n').encode())
        for box, saved_iou in zip(row['boxes'], row['ious']):
            assert len(box) == 6 and all(math.isfinite(v) for v in box)
            assert all(v > 0 for v in box[3:])
            assert math.isfinite(saved_iou) and 0 <= saved_iou <= 1
            recomputed = iou_from_box(box, row['root_box'])
            max_iou_difference = max(max_iou_difference, abs(recomputed-saved_iou))
            for t in (.25,.5):
                threshold_mismatches[str(t)] += int((recomputed > t) != (saved_iou > t))
        for mode in ('actual','bypass'):
            value = row[mode]
            assert len(value['scores']) == 256 and all(math.isfinite(v) for v in value['scores'])
            q = value['query']
            assert value['scores'][q] == max(value['scores'])
            assert value['iou'] == row['ious'][q]
            counts[mode]['hits25'] += int(value['iou'] > .25)
            counts[mode]['hits50'] += int(value['iou'] > .5)
            for threshold, key in ((.25,'oracle25'),(.5,'oracle50')):
                assert value[key][-1] == int(any(u > threshold for u in row['ious']))
                assert len(value[key]) == 4
                counts[mode][key] = [a+b for a,b in zip(counts[mode][key],value[key])]
        assert row['actual']['query'] == row['historical_actual_query'] == old['bbs']['query']
        for t in (.25,.5):
            assert (row['actual']['iou'] > t) == (old['bbs']['iou'] > t)
            actual = row['actual']['iou'] > t
            bypass = row['bypass']['iou'] > t
            transitions[str(t)]['repairs'] += int(actual and not bypass)
            transitions[str(t)]['damages'] += int(bypass and not actual)
        changed += int(row['actual']['query'] != row['bypass']['query'])
        if any((row['actual']['iou'] > t) != (row['bypass']['iou'] > t) for t in (.25,.5)):
            examples.append({k:row[k] for k in ('row_id','scan_id','target_id','text')} |
                            {mode:{k:row[mode][k] for k in ('query','iou')} for mode in ('actual','bypass')})
assert index+1 == 9508
for mode in counts:
    assert {key:counts[mode][key] for key in ('hits25','hits50')} == receipt['metrics'][mode]
    assert counts[mode]['oracle25'][-1] == counts['actual']['oracle25'][-1]
    assert counts[mode]['oracle50'][-1] == counts['actual']['oracle50'][-1]
for value in transitions.values():value['net'] = value['repairs']-value['damages']
assert transitions == receipt['transitions']
assert changed == receipt['changed_selected_queries'] == 99
assert counts['actual']['hits25'] == 5613 and counts['actual']['hits50'] == 4419
summary = {'status':'cpu_recount_complete','rows':9508,'primary_mode':'bbs',
           'metrics':counts,'transitions':transitions,'changed_selected_queries':changed,
           'row_identity_sha256':row_identity.hexdigest(),
           'candidate_iou_reconstruction':{'boxes':9508*256,'max_abs_difference':max_iou_difference,
                                           'threshold_mismatches':threshold_mismatches},
           'historical_actual_mismatches':0,'optimizer_updates':0,
           'checkpoint_sha256':receipt['checkpoint_sha256'],'scope':receipt['scope'],
           'raw_rows_sha256':receipt['rows_sha256'],'threshold_transition_examples':examples}
(root/'SUMMARY.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
report = f'''# PV-Ground P2 fixed-box semantic diagnostic

Full ScanRefer development validation: 9508 expressions, native last/bbs,
GroupFree predicted-object-assisted two-stage protocol, seed2027. The same
trained P2 step3723 checkpoint supplies every candidate box and Mask for both
branches. Only the final semantic residual from P2 is bypassed.

| Semantic branch | Hits@0.25 | Hits@0.50 | Acc@0.25 | Acc@0.50 |
|---|---:|---:|---:|---:|
| P2 actual | {counts['actual']['hits25']} | {counts['actual']['hits50']} | {100*counts['actual']['hits25']/9508:.4f}% | {100*counts['actual']['hits50']/9508:.4f}% |
| P2 semantic bypass | {counts['bypass']['hits25']} | {counts['bypass']['hits50']} | {100*counts['bypass']['hits25']/9508:.4f}% | {100*counts['bypass']['hits50']/9508:.4f}% |

Actual minus bypass: +1/+3 hits. At0.25:2 repairs/1 damage; at0.50:4 repairs/1
damage. Selected Query changes in99/9508 expressions. This is a small positive
direct forward effect at this checkpoint, with no independent repeat.

All256 boxes, both score arrays, GT and input identities are saved privately.
CPU recount agrees with the native/manual evaluator receipts. All2434048
candidate IoUs were reconstructed from saved boxes and real root GT;
maximum absolute floating-point difference {max_iou_difference:.10g},
threshold classification differences {threshold_mismatches}.
Historical actual selection/threshold replay mismatches:0.
Model buffers and checkpoint unchanged; optimizer updates:0.

This does **not** compare a trained model without P2. Upstream features, D/G
reader, semantic head and geometry were trained with P2. It cannot prove why
the new model lost original-G capability, nor establish gradient conflict.

The completed same-budget training comparison remains G5600/4452 versus
G+P25613/4419 (+13/-33). Original verified G5615/4495 remains the strong
starting point. P2 is not promoted and the strict4754 target remains unmet.
The present result supplies little evidence for expanding P2 or immediately
adding P3. Keep the existing G weights and distinguish expression-conditioned
selection learning from changes to the geometry path in the next experiment.

Evidence: full/receipt.json, full/rows.jsonl.gz, INTAKE.json, SUMMARY.json,
run.py and the original paired formal rows. Compressed rows are private;
public receipts and aggregate report retain provenance without checkpoint copies.
'''
(root/'REPORT.md').write_text(report,encoding='utf-8')
print(json.dumps({key:summary[key] for key in ('status','metrics','transitions','candidate_iou_reconstruction')}))
