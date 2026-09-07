import datetime
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np

repo=Path('C:/Users/gb/.codex_mcln_g0_20260905')
directory=repo/'refine-logs/scanrefer_instance_overlap_20260907_v1'
summary=json.loads((directory/'summary.json').read_bytes())
with gzip.open(directory/'overlap_rows.json.gz','rt') as f:result=json.load(f)
original=json.loads((repo/'refine-logs/scanrefer_stage_diagnostic_20260907_v1/diagnostic_result/stage_rows.json').read_bytes())
stage_summary=json.loads((repo/'refine-logs/scanrefer_stage_diagnostic_20260907_v1/diagnostic_result/stage_summary.json').read_bytes())
assert hashlib.sha256((directory/'overlap_rows.json.gz').read_bytes()).hexdigest()==summary['rows_sha256']
assert (directory/'controller.exit').read_text().strip()=='0'
checked=0
for arm,rows in result['arms'].items():
 assert len(rows)==9508
 for index,row in enumerate(rows):
  old=original[arm][index];scene=result['scene_geometry'][row['scan_id']]
  assert all(row[k]==old[k] for k in ['row_id','scan_id','target_id'])
  assert scene['point_sha256']==old['point_sha256']
  gt=np.asarray(scene['boxes'],dtype=np.float32);gt[row['target_id']]=old['root_box']
  gs=np.maximum(gt[:,3:],np.float32(1e-6));gl=gt[:,:3]-gs/2;gu=gt[:,:3]+gs/2
  for name,item in row['stages'].items():
   if name in old['stages']:box=old['stages'][name]['box']
   else:
    parent={'geometry_query_native':'geometry','final_query_native':'v99_final'}[name]
    q=old['stages'][parent]['query_index'];slot=old['top16_query_indices'].index(q)
    assert item['query_index']==q and old['top16_valid'][slot]
    box=old['top16_boxes'][slot]
   p=np.asarray(box,dtype=np.float32);size=np.maximum(p[3:],np.float32(1e-6));lo=p[:3]-size/2;hi=p[:3]+size/2
   d=np.maximum(np.minimum(hi,gu)-np.maximum(lo,gl),0)
   intersection=d[:,0]*d[:,1]*d[:,2]
   a=np.maximum(hi-lo,0);b=np.maximum(gu-gl,0)
   scores=intersection/np.maximum(a[0]*a[1]*a[2]+b[:,0]*b[:,1]*b[:,2]-intersection,np.float32(1e-6))
   best=int(np.argmax(scores));ties=np.flatnonzero(np.abs(scores-scores[best])<=1e-6).tolist()
   category=('no_overlap' if scores[best]==0 else 'tied_max' if len(ties)>1 else 'root_unique_max' if best==row['target_id'] else 'other_same_label_unique_max' if scene['labels'][best]==scene['labels'][row['target_id']] else 'other_different_label_unique_max')
   assert category==item['category'],(arm,index,name,category,item['category'])
   assert float(scores[row['target_id']])==item['root_iou']
   assert float(scores[best])==item['max_iou']
   if category not in ('tied_max','no_overlap'):assert item['best_object_id']==scene['object_ids'][best]
   if name in old['stages']:assert item['root_iou']==old['stages'][name]['rec_iou']
   checked+=1
 for name,reported in summary['arms'][arm]['stages'].items():
  values=[r['stages'][name] for r in rows]
  assert dict(Counter(v['category'] for v in values))==reported['categories']
  for suffix,t in [('025',.25),('050',.5)]:
   bad=[v for v in values if v['root_iou']<=t]
   assert reported[suffix]['hits']==9508-len(bad) and reported[suffix]['failures']==len(bad)
   for category,count in reported[suffix]['failure_categories'].items():assert sum(v['category']==category for v in bad)==count
   if name in stage_summary['arms'][arm]['metrics']:
    assert reported[suffix]['hits']==stage_summary['arms'][arm]['metrics'][name]['hits'+suffix]
 for name,source,target in [('same_query_geometry_effect','geometry_query_native','geometry'),('same_query_final_effect','final_query_native','v99_final')]:
  for suffix,t in [('025',.25),('050',.5)]:
   pairs=[(r['stages'][source],r['stages'][target]) for r in rows]
   for key,test in [('repair',lambda a,b:a<=t<b),('break',lambda a,b:b<=t<a)]:
    selected=[(a,b) for a,b in pairs if test(a['root_iou'],b['root_iou'])]
    expected=summary['arms'][arm][name][suffix]
    assert expected[key+'s']==len(selected)
    assert expected[key+'_categories']==dict(Counter(a['category']+'->'+b['category'] for a,b in selected))
for stage,reported in summary['paired_stages'].items():
 pairs=[(a['stages'][stage],b['stages'][stage]) for a,b in zip(result['arms']['protected_v99'],result['arms']['local_v99'])]
 for suffix,t in [('025',.25),('050',.5)]:
  repairs=[(a,b) for a,b in pairs if a['root_iou']<=t<b['root_iou']]
  breaks=[(a,b) for a,b in pairs if b['root_iou']<=t<a['root_iou']]
  assert reported[suffix]['repairs']==len(repairs) and reported[suffix]['breaks']==len(breaks)
  assert reported[suffix]['net']==len(repairs)-len(breaks)
  assert reported[suffix]['repair_categories']==dict(Counter(a['category']+'->'+b['category'] for a,b in repairs))
  assert reported[suffix]['break_categories']==dict(Counter(a['category']+'->'+b['category'] for a,b in breaks))
terminal=repo/'refine-logs/scanrefer_native_box_transfer_pair_20260907_v1'
new_rows=json.loads((terminal/'terminal_rows.json').read_bytes())
metrics=json.loads((terminal/'terminal_metrics.json').read_bytes())
native=json.loads((terminal/'terminal_native_metrics.json').read_bytes())
for arm,rows in new_rows.items():
 assert len(rows)==6887
 for suffix,t in [('025',.25),('050',.5)]:
  assert sum(r['rec_iou']>t for r in rows)==metrics[arm]['rec_hits'+suffix]
  assert sum(r['native_rec_iou']>t for r in rows)==native[arm]['rec_hits'+suffix]
  assert sum(r['mask_iou']>t for r in rows)==metrics[arm]['mask_hits'+suffix]
 assert abs(sum(r['mask_iou'] for r in rows)/6887*100-metrics[arm]['mask_miou'])<1e-10
audit={'status':'pass','time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
 'geometry_stage_records_recomputed':checked,'historical_rows_per_arm':9508,'historical_scenes':len(result['scene_geometry']),
 'native_box_terminal_rows_recounted_per_arm':6887,'threshold_counts_categories_and_transitions_verified':True,
 'semantic_identity_proven':False,'gpu_forwards':0,'new_formal_rows':0,
 'summary_sha256':hashlib.sha256((directory/'summary.json').read_bytes()).hexdigest(),
 'native_box_terminal_rows_sha256':hashlib.sha256((terminal/'terminal_rows.json').read_bytes()).hexdigest(),
 'scope':'Independently recomputes saved geometry tables and observed stage predictions; scene-pickle provenance is the bound remote execution evidence, not a second pickle load.'}
with (directory/'independent_recount.json').open('x') as f:json.dump(audit,f,indent=2,sort_keys=True)
print(json.dumps(audit),flush=True)
