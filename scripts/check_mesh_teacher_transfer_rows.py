"""Independent CPU recomputation of stored native and teacher box evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def iou(boxes, root):
    b=np.asarray(boxes,dtype=np.float64)
    r=np.asarray(root,dtype=np.float64)
    size=np.maximum(b[...,3:],1e-6);rs=np.maximum(r[3:],1e-6)
    inter=np.maximum(np.minimum(b[...,:3]+size/2,r[:3]+rs/2)-np.maximum(b[...,:3]-size/2,r[:3]-rs/2),0).prod(-1)
    return inter/np.maximum(size.prod(-1)+rs.prod()-inter,1e-6)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--current',type=Path,required=True)
    parser.add_argument('--historical',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    rows=json.loads((args.current/'rows.json').read_bytes())
    old=json.loads((args.historical/'rows.json').read_bytes())
    receipt=json.loads((args.current/'receipt.json').read_bytes())
    assert hashlib.sha256((args.current/'rows.json').read_bytes()).hexdigest()==receipt['rows_sha256']
    assert len(rows)==len(old)==512
    computed={name:[] for name in receipt['summary']['hits']}
    eligible=[];gains=[];selected_changes=0;raw_unchanged=0;variant_oracles=[]
    for row,prior in zip(rows,old):
        assert (row['row_id'],row['scan_id'])==(prior['row_id'],prior['scan_id'])
        native=iou(row['native_boxes'],row['root_box'])
        assert native.shape==(256,)
        variant=np.asarray(row['variant_boxes']);valid=np.asarray(row['variant_valid'],dtype=bool)
        assert variant.shape==(16,7,6) and valid.shape==(16,7)
        flat=row['teacher_selected_flat_index']
        assert valid.reshape(-1)[flat]
        assert np.array_equal(variant.reshape(-1,6)[flat],row['teacher_box'])
        teacher=float(iou(row['teacher_box'],row['root_box']))
        values={'native':float(native[row['native_query_index']]),'raw_default':float(native[row['native_query_index']]),
            'hungarian_root':float(native[row['hungarian_root_query_index']]),'native_best':float(native.max()),
            'teacher':teacher,'teacher_source_query':float(native[row['teacher_source_query_index']]),
            'corresponding_query':float(native[row['corresponding_query_index']])}
        assert row['teacher_source_query_index']==row['parent_query_indices'][flat//7]
        for name,value in values.items():
            assert abs(value-row['ious'][name])<1e-5,(row['row_id'],name)
            computed[name].append(value)
        ok=teacher>.25 and teacher>values['hungarian_root']
        eligible.append(ok)
        if ok:gains.append(teacher-values['hungarian_root'])
        if (row['teacher_source_query_index'],row['teacher_variant_index'])!=(prior['teacher_source_query_index'],prior['teacher_variant_index']):selected_changes+=1
        if all(abs(row['ious'][key]-prior['ious'][key])<1e-6 for key in ['native','hungarian_root','native_best','raw_default']):raw_unchanged+=1
        variant_oracles.append(float(iou(variant.reshape(-1,6),row['root_box'])[valid.reshape(-1)].max()))
    hits={name:{str(t):sum(x>t for x in values) for t in [.25,.5]} for name,values in computed.items()}
    assert hits==receipt['summary']['hits']
    result={'status':'pass','rows':512,'hits':hits,'eligible_gt_supported_teacher_rows':sum(eligible),
        'teacher_minus_hungarian_iou_gain':{'mean':float(np.mean(gains)),'median':float(np.median(gains)),'max':max(gains)},
        'all_recorded_boxes_recomputed':True,'same_fit_row_identities':True,'historical_native_record_rows_unchanged_at_1e6':raw_unchanged,
        'teacher_query_or_variant_selection_changed_rows':selected_changes,
        'train_only_16x7_geometry_oracle_hits':{str(t):sum(x>t for x in variant_oracles) for t in [.25,.5]},
        'quality_result':False,'formal_rows':0,'new_optimizer_steps':0,
        'current_rows_sha256':receipt['rows_sha256'],'historical_rows_sha256':hashlib.sha256((args.historical/'rows.json').read_bytes()).hexdigest()}
    with args.output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
