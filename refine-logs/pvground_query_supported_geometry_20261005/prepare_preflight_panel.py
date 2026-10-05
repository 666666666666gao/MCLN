import json
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parent
branch=root.parent/'pvground_mask_branch_responsibility_20261005'
prior=root.parent/'pvground_mask_geometry_responsibility_20261005'
summary=json.loads((branch/'analysis/SUMMARY.json').read_bytes())
assert summary['counts']['own_query_confirmed']==1090 and summary['prior_box_threshold_differences']==0
panels=[]
for index in range(8):
    new=dict(np.load(branch/'complete'/('batch_%02d.npz'%index)))
    old=dict(np.load(prior/'complete'/('batch_%02d.npz'%index)))
    pool=(new['query_iou']>.5)&(new['fused_iou']>.5)&(new['box_iou']<=.5)&(new['matched_slot']<0)
    coarse=old['coarse_boxes']
    truth=new['root_box'][:,None]
    target=np.concatenate([(coarse[...,:3]-(truth[...,:3]-truth[...,3:]/2))/coarse[...,3:]*4-2,
        ((truth[...,:3]+truth[...,3:]/2)-coarse[...,:3])/coarse[...,3:]*4-2],axis=-1)
    outside=(target<-4)|(target>4)
    panels.append(dict(batch_index=index,row_ids=new['row_id'].tolist(),extra_row_counts=pool.sum(1).tolist(),
        extra_outside_faces=int(outside[pool].sum())))
chosen=next(panel for panel in panels if 0 in panel['extra_row_counts'] and max(panel['extra_row_counts'])>0 and panel['extra_outside_faces']>0)
record=dict(selection='First actual fixed64 batch with both empty/nonempty additional eligibility and current clipped endpoint targets; engineering coverage only, not accuracy selection',
    chosen=chosen,all_panels=panels,accuracy_result=False)
(root/'PREFLIGHT_PANEL.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
for arm in ('control','query_supported'):
    path=root/(arm+'_spec.json')
    spec=json.loads(path.read_bytes());spec['preflight_batch_index']=chosen['batch_index']
    path.write_text(json.dumps(spec,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'preflight_panel':chosen}))
