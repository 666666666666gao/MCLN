import ast,csv,hashlib,json,os,sys,time
from pathlib import Path
r=Path(__file__).resolve().parent;source=r/'source';data=Path('/root/autodl-tmp/DATA_ROOT_mcln_meshsp');os.chdir(str(source));sys.path.insert(0,str(source));sys.path.insert(1,str(source/'pointnet2'))
import numpy as np
from src.joint_det_dataset import Joint3DDataset,unpickle_data
from data.scannet_utils import read_label_mapping
started=time.time();scans=list(unpickle_data(str(data/'val_v3scans.pkl')))[0]
obj=Joint3DDataset.__new__(Joint3DDataset);obj.label_map=read_label_mapping('data/meta_data/scannetv2-labels.combined.tsv',label_from='raw_category',label_to='id');obj.split='val';obj.augment=False
classes=json.loads((source/'data/cls_results.json').read_text());report={'status':'complete','model_forwards':0,'training_steps':0,'scope':'actual author _get_scene_objects and predicted-class slot assignment; no full language Dataset or model','datasets':{}}
for name in ['nr3d','sr3d']:
 scenes=set(ast.literal_eval((source/('data/meta_data/'+name+'_test_scans.txt')).read_text()))
 with (data/('refer_it_3d/'+name+'.csv')).open() as f:raw=list(csv.DictReader(f))
 field='correct_guess' if name=='nr3d' else 'mentions_target_class';rows=[a for a in raw if a['scan_id'] in scenes and a[field].lower()=='true']
 selected=sorted({a['scan_id'] for a in rows});sceneinfo={};excluded=[];empty=[];mismatch=[];zero_extent=0
 for scene in selected:
  scan=scans[scene];class_ids,boxes,keep=obj._get_scene_objects(scan);pred=np.asarray(classes[scene]);assigned=np.zeros(len(boxes),dtype=np.int64)
  assert len(pred[pred>-1])==int(keep.sum()),(scene,len(pred[pred>-1]),keep.sum())
  assigned[keep]=pred[pred>-1];assert np.isfinite(boxes).all() and (boxes[keep,3:]>=0).all()
  assert (assigned[keep]>=0).all() and (assigned[keep]<485).all()
  zero_extent+=int((boxes[keep,3:]<=0).any(axis=1).sum())
  sceneinfo[scene]={'object_slots':len(scan.three_d_objects),'kept_slots':int(keep.sum()),'keep':keep}
 for index,a in enumerate(rows):
  scan=scans[a['scan_id']];tid=int(a['target_id']);assert 0<=tid<len(scan.three_d_objects)
  box=np.asarray(scan.get_object_bbox(tid)).reshape(-1);assert box.shape==(6,) and np.isfinite(box).all()
  pts=scan.three_d_objects[tid]['points'];assert np.asarray(pts).min()>=0 and np.asarray(pts).max()<len(scan.pc)
  if len(pts)==0:empty.append(index)
  keep=sceneinfo[a['scan_id']]['keep']
  if tid>=len(keep) or not keep[tid]:excluded.append(index)
  if scan.get_object_instance_label(tid)!=a['instance_type']:mismatch.append(index)
 assert len(rows)=={'nr3d':7899,'sr3d':17726}[name]
 report['datasets'][name]={'rows':len(rows),'scenes':len(selected),'object_boxes_checked':sum(i['kept_slots'] for i in sceneinfo.values()),'all_predicted_class_slot_counts_match':True,'all_target_ids_valid':True,'all_target_point_indices_valid':True,'root_not_in_object_input_count':len(excluded),'root_not_in_object_input_rows':excluded,'target_label_string_mismatch_count':len(mismatch),'target_label_string_mismatch_rows':mismatch,'zero_extent_object_boxes':zero_extent,'empty_target_pointsets':empty}
report['elapsed_seconds']=time.time()-started
(r/'referit_object_slot_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
