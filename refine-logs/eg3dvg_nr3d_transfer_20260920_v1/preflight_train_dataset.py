import hashlib,json,os,random,shutil,sys,time
from pathlib import Path
r=Path(__file__).resolve().parent;source=r/'source';dest=r/'train_input_source';assert not dest.exists();shutil.copytree(str(source),str(dest),ignore=shutil.ignore_patterns('__pycache__'))
p=dest/'src/joint_det_dataset.py';text=p.read_text();old="assert self.split == 'val' and test_dataset in ['nr3d', 'sr3d']\n        with open('/root/autodl-tmp/mcln_eg3dvg_acceptance_20260920_v1/referit_input_cache/' + test_dataset + '_annotations.pkl', 'rb') as file:\n            self.annos = pickle.load(file)"
new="assert self.split == 'train' and test_dataset == 'nr3d'\n        self.annos = []\n        for name, count in dataset_dict.items():\n            with open('"+str(r/ 'train_input_cache')+"/' + name + '_annotations.pkl', 'rb') as file:\n                self.annos += pickle.load(file) * count"
assert text.count(old)==1;p.write_text(text.replace(old,new))
os.chdir(str(dest));sys.path.insert(0,str(dest));sys.path.insert(1,str(dest/'pointnet2'))
import numpy as np
import torch
from src.joint_det_dataset import Joint3DDataset
random.seed(2027);np.random.seed(2027);torch.manual_seed(2027)
receipt=json.loads((r/'train_input_cache/receipt.json').read_text());nr=receipt['datasets']['nr3d']['rows'];det=receipt['datasets']['scannet']['rows'];start=time.time()
ds=Joint3DDataset(dataset_dict={'nr3d':1,'scannet':10},test_dataset='nr3d',split='train',data_path='/root/autodl-tmp/DATA_ROOT_mcln_meshsp/',use_color=True,detect_intermediate=True,butd_cls=True)
assert len(ds)==nr+10*det and ds.augment and ds.butd_cls and not ds.butd and not ds.butd_gt
samples=[]
for idx in [0,nr//7,nr//3,nr//2,2*nr//3,3*nr//4,nr-2,nr-1,nr,nr+det//2,nr+det-1]:
 x=ds[idx];assert x['point_clouds'].shape==(50000,6) and x['all_detected_boxes'].shape==(132,6)
 assert np.isfinite(x['point_clouds']).all() and np.isfinite(x['all_detected_boxes']).all()
 assert np.array_equal(x['all_detected_boxes'],x['all_bboxes']) and np.array_equal(x['all_detected_bbox_label_mask'],x['all_bbox_label_mask'])
 assert np.isfinite(x['positive_map']).all() and np.isfinite(x['negative_positive_map']).all()
 samples.append({'index':idx,'dataset':ds.annos[idx]['dataset'],'scan_id':x['scan_ids'],'utterance':x['utterances'],'negative_utterance':x['negative_utterances'],'object_slots':int(x['all_detected_bbox_label_mask'].sum()),'positive_map_sum':float(x['positive_map'].sum()),'point_sha256':hashlib.sha256(x['point_clouds'].tobytes()).hexdigest()})
report={'status':'complete','model_forwards':0,'training_steps':0,'scope':'full augmented CPU training Dataset plus 8 Nr and 3 ScanNet fixed samples','rows':len(ds),'nr3d_rows':nr,'scannet_rows_before_repeat':det,'scannet_repeat':10,'seconds':time.time()-start,'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'active_eval_source_unchanged':True,'augmentation_changed':False,'samples':samples}
(r/'train_dataset_preflight.json').write_text(json.dumps(report,indent=2));print('EG_TRAIN_DATASET_COMPLETE '+json.dumps(report),flush=True)
