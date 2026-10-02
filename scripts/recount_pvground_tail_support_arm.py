"""Recount one completed support arm, stored predictions and its paired fit order."""
from collections import Counter
import argparse
import datetime
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--arm', choices=('tail_raw', 'tail_fused'), required=True)
args = parser.parse_args()
local = Path(r'C:\Users\gb\.codex\tmp\pvground_fused_support_20261002')
root = local/('complete_'+args.arm)
if args.arm == 'tail_raw':
    control = Path(r'C:\Users\gb\.codex\tmp\pvground_g_p2_20261002\complete\g_control')
    control_role = 'historical_G_continuation'
else:
    control = local/'complete_tail_raw/arm'
    control_role = 'same_tail_raw'
read = lambda path: [json.loads(line) for line in path.read_text().splitlines()]
train = json.loads((root/'arm/receipt.json').read_bytes())
intake = json.loads((root/'INTAKE.json').read_bytes())
for name,item in intake['files'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest() == item['sha256'],name
assert json.loads((root/'status.json').read_bytes())['status'] == 'complete'
assert train['training_steps'] == 3723 and train['fit_seen_exactly_once']
assert train['support_arm'] == args.arm
assert train['fused_support'] == (args.arm == 'tail_fused')
training = read(root/'arm/train.jsonl')
reference = read(control/'train.jsonl')
partitions = json.loads((root/'source/split_protocol.json').read_bytes())['row_ids']
assert len(training) == len(reference) == 3723
seen = []
for index,(actual,prior) in enumerate(zip(training,reference),1):
    assert actual['step'] == prior['step'] == index
    assert actual['rows'] == prior['rows'],index
    seen.extend(actual['rows'])
assert len(seen) == 29778 and Counter(seen) == Counter(partitions['fit'])
assert not set(seen).intersection(partitions['holdout'])


def iou(box,target):
    intersection = 1.
    for axis in range(3):
        lo = max(box[axis]-box[axis+3]/2,target[axis]-target[axis+3]/2)
        hi = min(box[axis]+box[axis+3]/2,target[axis]+target[axis+3]/2)
        intersection *= max(0.,hi-lo)
    return intersection/(box[3]*box[4]*box[5]+target[3]*target[4]*target[5]-intersection)


results = {}
stage_rows = {}
for stage,n in (('initial',6887),('terminal',6887),('formal',9508)):
    rows = read(root/'arm'/stage/'rows.jsonl')
    receipt = json.loads((root/'arm'/stage/'receipt.json').read_bytes())
    assert receipt['status'] == 'pass' and len(rows) == receipt['rows'] == n
    assert len({r['row_id'] for r in rows}) == n
    if stage != 'formal':
        assert [r['row_id'] for r in rows] == partitions['holdout']
    stage_rows[stage] = rows
    results[stage] = {}
    for mode in ('bbs','bbf'):
        metric = dict(rec_hits25=sum(r[mode]['iou']>.25 for r in rows),
            rec_hits50=sum(r[mode]['iou']>.5 for r in rows),
            mask_hits25=sum(r[mode]['mask_iou']>.25 for r in rows),
            mask_hits50=sum(r[mode]['mask_iou']>.5 for r in rows),
            mask_iou_sum=sum(r[mode]['mask_iou'] for r in rows))
        metric['mask_miou'] = metric['mask_iou_sum']/n*100
        assert metric == receipt['metrics'][mode],(stage,mode)
        recomputed = [iou(r[mode]['box'],r['root_box']) for r in rows]
        coarse = [iou(r[mode]['coarse_box'],r['root_box']) for r in rows]
        metric.update(cpu_box_rec_hits25=sum(v>.25 for v in recomputed),
            cpu_box_rec_hits50=sum(v>.5 for v in recomputed),
            cpu_box_iou_max_abs_difference=max(abs(v-r[mode]['iou']) for v,r in zip(recomputed,rows)),
            cpu_coarse_iou_max_abs_difference=max(abs(v-r[mode]['coarse_iou']) for v,r in zip(coarse,rows)),
            acc25=metric['rec_hits25']/n*100,acc50=metric['rec_hits50']/n*100)
        assert metric['cpu_box_rec_hits25'] == metric['rec_hits25']
        assert metric['cpu_box_rec_hits50'] == metric['rec_hits50']
        for threshold in (.25,.5):
            assert all((value>threshold)==(row[mode]['iou']>threshold)
                       for value,row in zip(recomputed,rows))
            assert all((value>threshold)==(row[mode]['coarse_iou']>threshold)
                       for value,row in zip(coarse,rows))
        metric['cpu_each_row_thresholds_match'] = True
        metric['same_selected_query_refinement'] = {}
        metric['executed_gpu_coverage_flags'] = {}
        for threshold,key in ((.25,'25'),(.5,'50')):
            fixes=sum(a<=threshold<b for a,b in zip(coarse,recomputed))
            breaks=sum(b<=threshold<a for a,b in zip(coarse,recomputed))
            metric['same_selected_query_refinement'][key] = dict(fixes=fixes,breaks=breaks,
                net=fixes-breaks,coarse_hits=sum(v>threshold for v in coarse))
            assert fixes-breaks == sum(v>threshold for v in recomputed)-sum(v>threshold for v in coarse)
            coverage=sum(r[mode]['oracle'+key][-1] for r in rows)
            metric['executed_gpu_coverage_flags'][key] = dict(full256=coverage,
                top16=sum(r[mode]['oracle'+key][0] for r in rows),
                selected=metric['rec_hits'+key],good_box_not_selected=coverage-metric['rec_hits'+key])
            for row in rows:
                flags = row[mode]['oracle'+key]
                assert flags == sorted(flags) and all(v in (0,1) for v in flags)
                assert int(row[mode]['iou']>threshold) <= flags[0]
        results[stage][mode] = metric
initial_reference = read(control/'initial/rows.jsonl')
assert len(initial_reference) == len(stage_rows['initial']) == 6887
for actual,prior in zip(stage_rows['initial'],initial_reference):
    for key in ('row_id','scan_id','target_id','root_box','point_sha256'):
        assert actual[key] == prior[key]
    for mode in ('bbs','bbf'):
        for key in ('query','box','iou'):
            assert actual[mode][key] == prior[mode][key]
transitions = {}
for mode in ('bbs','bbf'):
    transitions[mode] = {}
    for threshold in (.25,.5):
        fixes=breaks=0
        for old,new in zip(stage_rows['initial'],stage_rows['terminal']):
            for key in ('row_id','scan_id','target_id','root_box','point_sha256'):
                assert old[key] == new[key]
            a=old[mode]['iou']>threshold;b=new[mode]['iou']>threshold
            fixes += not a and b; breaks += a and not b
        transitions[mode][str(threshold)] = dict(fixes=fixes,breaks=breaks,net=fixes-breaks)
assert transitions == train['transitions']
paired = {}
for stage in ('terminal','formal'):
    prior = read(control/stage/'rows.jsonl')
    assert len(prior) == len(stage_rows[stage])
    paired[stage] = {}
    for old,new in zip(prior,stage_rows[stage]):
        for key in ('row_id','scan_id','target_id','root_box','point_sha256'):
            assert old[key] == new[key],(stage,key)
    for mode in ('bbs','bbf'):
        paired[stage][mode] = {}
        for threshold in (.25,.5):
            fixes=sum(a[mode]['iou']<=threshold<b[mode]['iou'] for a,b in zip(prior,stage_rows[stage]))
            breaks=sum(b[mode]['iou']<=threshold<a[mode]['iou'] for a,b in zip(prior,stage_rows[stage]))
            paired[stage][mode][str(threshold)] = dict(fixes=fixes,breaks=breaks,net=fixes-breaks)
formal = results['formal']['bbs']
summary = dict(status='pass',time_cst=datetime.datetime.now().astimezone().isoformat(),
    results=results,full_fit_order_matches_control=True,fit_rows=29778,training_steps=3723,
    transitions=transitions,paired_control_role=control_role,paired_vs_control=paired,
    support_arm=args.arm,
    formal_vs_original_G_delta_hits=[formal['rec_hits25']-5615,formal['rec_hits50']-4495],
    primary_target50_pass=formal['rec_hits50']>=4754,
    evidence_limits='Selected and coarse boxes recalculated against saved GT. Mask summaries recounted from saved IoUs, not fresh raw Mask arrays. Full256 coverage is saved GPU flags because all256 boxes were not saved. E0 parity is limited to saved selected REC/input identities; initial Mask differences are retained. Same selected Query refinement is internal mechanism evidence. Only tail_fused versus completed tail_raw isolates predicted support under the shared tail protocol; historical G comparisons do not.')
(root/'CPU_RECOUNT.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({'status':'pass','training_steps':3723,'formal_bbs':[formal['rec_hits25'],formal['rec_hits50']],
                  'original_G_delta':summary['formal_vs_original_G_delta_hits'],
                  'arm':args.arm,'control_role':control_role,
                  'target50_pass':summary['primary_target50_pass']}))
