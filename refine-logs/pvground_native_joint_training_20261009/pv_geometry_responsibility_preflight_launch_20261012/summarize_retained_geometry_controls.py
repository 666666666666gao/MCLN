"""Summarize already evaluated geometry controls from one shared forward."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
path = Path('C:/Users/gb/.codex/tmp/pvground_extremal_span_evidence_20261009/runner_v1/complete_fit/formal/rows.jsonl')
output = root/'retained_geometry_controls_20261012'
assert not output.exists()
rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
assert len(rows) == 9508
assert all(row['row_id']==i and row['parent_forwards']==row['final_semantic_head_calls']==1
           for i,row in enumerate(rows))
ious = {name:[row['controls'][name]['iou'] for row in rows]
        for name in ('native','mask','fixed_half')}
ious.update({name:[row['arms'][name]['iou'] for row in rows]
             for name in ('whole_support','extremal_support')})
metrics = {name:dict(hits025=sum(value>.25 for value in values),
                     hits050=sum(value>.5 for value in values)) for name,values in ious.items()}
assert metrics['extremal_support'] == dict(hits025=5677,hits050=4920)
comparisons = {}
for before in ('native','mask','fixed_half','whole_support'):
    comparisons[before+'__to__extremal_support'] = {}
    for suffix,threshold in (('025',.25),('050',.5)):
        repairs=sum(a<=threshold<b for a,b in zip(ious[before],ious['extremal_support']))
        breaks=sum(b<=threshold<a for a,b in zip(ious[before],ious['extremal_support']))
        net=metrics['extremal_support']['hits'+suffix]-metrics[before]['hits'+suffix]
        assert repairs-breaks==net
        comparisons[before+'__to__extremal_support'][suffix] = dict(repairs=repairs,breaks=breaks,net=net)
result = dict(scope='REAL_RETAINED_9508_SAME_FORWARD_GEOMETRY_CONTROL_COUNTS',
    input_sha256={str(path.resolve()):hashlib.sha256(path.read_bytes()).hexdigest()},rows=9508,
    metrics=metrics,comparisons=comparisons,model_forwards=0,optimizer_updates=0,
    GPU_calls=0,remote_queries=0,retained_checkpoint_changed=False,
    interpretation='Previously evaluated outputs, same selected Query and shared parent forward. Learned axis mixing has a positive comparison to fixed-half geometry; extremal versus whole features has no strict advantage (-1) and cannot support a claim that extremal features independently improve accuracy. Historical model selection is unchanged; no multi-seed significance or new model accuracy is claimed.')
output.mkdir()
(output/'RESULT.json').write_bytes((json.dumps(result,indent=2)+'\n').encode())
print(json.dumps(result))
