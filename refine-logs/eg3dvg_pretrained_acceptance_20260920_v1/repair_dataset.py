import datetime,hashlib,json,os,sys,time
from pathlib import Path
r=Path(__file__).resolve().parent;p=r/'source/src/joint_det_dataset.py'
raw=p.read_text();before=hashlib.sha256(p.read_bytes()).hexdigest()
old="        # Evaluate original annotations using the upstream parser; no private cache.\n        assert self.split == 'val'\n        self.annos = self.load_annos(test_dataset)\n"
assert raw.count(old)==1
raw=raw.replace(old,"        # Cache produced by this checkout's original load_annos/Scene_graph_parse.\n        assert self.split == 'val' and test_dataset == 'scanrefer'\n        with open("+repr(str(r/'annotations.pkl'))+", 'rb') as file:\n            self.annos = pickle.load(file)\n")
a=raw.index('        dataset = list(self.dataset_dict.keys())[0]\n        one_hot_dict')
b=raw.index('    def _get_scene_objects',a)
assert raw[a:b].count('point_class_label')==3
raw=raw[:a]+'        return bboxes, box_label_mask, point_instance_label, gt_masks\n\n\n'+raw[b:]
assert raw.count('gt_bboxes, box_label_mask, point_instance_label, gt_masks, point_class_label =')==1
raw=raw.replace('gt_bboxes, box_label_mask, point_instance_label, gt_masks, point_class_label =','gt_bboxes, box_label_mask, point_instance_label, gt_masks =')
assert 'point_class_label' not in raw
p.write_text(raw)
(r/'dataset_repair.json').write_text(json.dumps({'before_sha256':before,'after_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'reason':'Public DC omits one_hot_scanrefer; point_class_label is computed but never returned by __getitem__ or used by any model/evaluator. Remove only dead computation and return slot.','cache':'Original upstream parser already completed all annotations before the sample-label exception. Reuse its saved annotations exactly.','annotations_sha256':hashlib.sha256((r/'annotations.pkl').read_bytes()).hexdigest()},indent=2))
os.chdir(str(r/'source'));sys.path.insert(0,str(r/'source'));sys.path.insert(1,str(r/'source/pointnet2'))
from src.joint_det_dataset import Joint3DDataset
import numpy as np,torch,random
t=time.time();random.seed(2027);np.random.seed(2027);torch.manual_seed(2027)
ds=Joint3DDataset(dataset_dict={'scanrefer':1,'scannet':10},test_dataset='scanrefer',split='val',data_path='/root/autodl-tmp/DATA_ROOT_mcln_meshsp/',use_color=True,detect_intermediate=True,butd=True)
assert len(ds)==9508 and not ds.augment and ds.butd and not ds.butd_gt and not ds.butd_cls
sample=ds[0];assert sample['point_clouds'].shape==(50000,6) and len(sample['superpoint'])==50000
rec={'status':'complete','time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'rows':len(ds),'scenes':len(ds.scans),'seconds':time.time()-t,'annotations_sha256':hashlib.sha256((r/'annotations.pkl').read_bytes()).hexdigest(),'sample_shapes':{k:list(v.shape) for k,v in sample.items() if hasattr(v,'shape')},'sample_utterance':sample['utterances'],'model_forwards':0,'training_steps':0}
(r/'data_receipt.json').write_text(json.dumps(rec,indent=2));print('EG_DATA_READY '+json.dumps(rec),flush=True)
