import hashlib,json,os,pickle,random,shutil,sys,time
from pathlib import Path
r=Path(__file__).resolve().parent;src=r/'source';dest=r/'referit_input_source';assert not dest.exists();shutil.copytree(str(src),str(dest),ignore=shutil.ignore_patterns('__pycache__'))
p=dest/'src/joint_det_dataset.py';raw=p.read_text();old="assert self.split == 'val' and test_dataset == 'scanrefer'\n        with open('/root/autodl-tmp/mcln_eg3dvg_acceptance_20260920_v1/annotations.pkl', 'rb') as file:"
new="assert self.split == 'val' and test_dataset in ['nr3d', 'sr3d']\n        with open('/root/autodl-tmp/mcln_eg3dvg_acceptance_20260920_v1/referit_input_cache/' + test_dataset + '_annotations.pkl', 'rb') as file:"
assert raw.count(old)==1;p.write_text(raw.replace(old,new))
os.chdir(str(dest));sys.path.insert(0,str(dest));sys.path.insert(1,str(dest/'pointnet2'))
import numpy as np
import torch
from src.joint_det_dataset import Joint3DDataset
random.seed(2027);np.random.seed(2027);torch.manual_seed(2027)
report={'status':'complete','scope':'full CPU val Dataset construction plus eight fixed samples per dataset; no model inference','model_forwards':0,'training_steps':0,'active_scan_source_unchanged':True,'prepared_source':str(dest),'dataset_source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'datasets':{}}
for name,n in [('nr3d',7899),('sr3d',17726)]:
 t=time.time();ds=Joint3DDataset(dataset_dict={name:1},test_dataset=name,split='val',data_path='/root/autodl-tmp/DATA_ROOT_mcln_meshsp/',use_color=True,detect_intermediate=True,butd_cls=True)
 assert len(ds)==n and not ds.augment and ds.butd_cls and not ds.butd_gt and not ds.butd
 chosen=[0,n//7,n//3,n//2,2*n//3,3*n//4,n-2,n-1];samples=[]
 for i in chosen:
  x=ds[i];pts=x['point_clouds'];boxes=x['all_detected_boxes'];ids=x['all_detected_class_ids'];keep=x['all_detected_bbox_label_mask'].astype(bool)
  assert pts.shape==(50000,6) and boxes.shape==(132,6) and ids.shape==(132,)
  assert np.isfinite(pts).all() and np.isfinite(boxes).all()
  assert np.array_equal(boxes,x['all_bboxes']) and np.array_equal(keep,x['all_bbox_label_mask'])
  predicted=np.asarray(ds.cls_results[x['scan_ids']]);assert np.array_equal(ids[keep],predicted[predicted>-1])
  samples.append({'index':i,'scan_id':x['scan_ids'],'target_id':int(x['target_id']),'point_sha256':hashlib.sha256(pts.tobytes()).hexdigest(),'object_slots':int(keep.sum()),'text':x['utterances']})
 report['datasets'][name]={'rows':len(ds),'loaded_cache_scenes':len(ds.scans),'sample_count':len(samples),'samples':samples,'seconds':time.time()-t,'object_input':'GT scene instance boxes with author predicted class ids'}
 print('REFERIT_DATASET_COMPLETE '+name+' '+str(len(ds)),flush=True)
 del ds
(r/'referit_dataset_preflight.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
