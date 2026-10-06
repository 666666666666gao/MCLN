"""Count actual three-stage overlaps and face movements on saved selected boxes."""
import hashlib
import json
from collections import Counter
from pathlib import Path

root=Path(__file__).resolve().parents[1]
intake=json.loads((root/'complete/INTAKE.json').read_bytes())
output=root/'analysis/STAGE_OVERLAP.json'
assert not output.exists()


def edges(box):
    return [box[a]+direction*box[a+3]/2 for a in range(3) for direction in (-1,1)]


def quantile(values,fraction):
    ordered=sorted(values)
    location=(len(ordered)-1)*fraction
    low=int(location)
    high=min(low+1,len(ordered)-1)
    return ordered[low]+(ordered[high]-ordered[low])*(location-low)


systems={}
for arm in ('control','support_reference'):
    name=arm+'/formal/rows.jsonl'
    raw=(root/'complete'/name).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==intake['files'][name]['sha256']
    data=[json.loads(line) for line in raw.splitlines()]
    assert len(data)==9508
    counts=Counter(''.join(str(int(row['bbs'][key]>.5)) for key in ('coarse_iou','reference_iou','iou')) for row in data)
    shifts={};errors={}
    for label,before,after in (('coarse_to_reference','coarse_box','reference_box'),('reference_to_final','reference_box','box')):
        values=[max(abs(x-y) for x,y in zip(edges(row['bbs'][before]),edges(row['bbs'][after]))) for row in data]
        shifts[label]={'median_m':quantile(values,.5),'p90_m':quantile(values,.9),'maximum_m':max(values)}
    for key in ('coarse_box','reference_box','box'):
        values=[max(abs(x-y) for x,y in zip(edges(row['bbs'][key]),edges(row['root_box']))) for row in data]
        errors[key]={'median_m':quantile(values,.5),'p90_m':quantile(values,.9)}
    systems[arm]={'three_stage_strict_hit_bits':{format(i,'03b'):counts[format(i,'03b')] for i in range(8)},
        'reference_repairs_final_retained':counts['011'],'reference_repairs_final_lost':counts['010'],
        'reference_damages_final_recovered':counts['101'],'reference_damages_final_persisted':counts['100'],
        'final_only_repairs':counts['001'],'final_new_damages_after_both_earlier_correct':counts['110'],
        'face_shift_meters':shifts,'GT_max_face_error_meters':errors}
    assert sum(counts.values())==9508
record={'scope':'actual9508_same_selected_Query_internal_GT_threshold_strict_greater_than_0.5',
    'bit_order':['coarse','reference','final'],'quantiles':'linear interpolation at (n-1)*fraction',
    'systems':systems,'does_not_establish_independent_baseline_gain_or_failure_cause':True}
output.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record))
