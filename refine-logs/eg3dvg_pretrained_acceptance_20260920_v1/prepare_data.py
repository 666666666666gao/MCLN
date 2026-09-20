import datetime,hashlib,json,os,pickle,sys,time
from pathlib import Path
r=Path(__file__).resolve().parent
os.chdir(str(r/'source'));sys.path.insert(0,str(r/'source'));sys.path.insert(1,str(r/'source/pointnet2'))
from src.joint_det_dataset import Joint3DDataset
import numpy as np,torch,random
t=time.time();random.seed(2027);np.random.seed(2027);torch.manual_seed(2027)
ds=Joint3DDataset(dataset_dict={'scanrefer':1,'scannet':10},test_dataset='scanrefer',split='val',data_path='/root/autodl-tmp/DATA_ROOT_mcln_meshsp/',use_color=True,detect_intermediate=True,butd=True)
assert len(ds)==9508 and ds.augment is False and ds.butd and not ds.butd_gt and not ds.butd_cls
with (r/'annotations.pkl').open('xb') as f:pickle.dump(ds.annos,f)
rows=[{'scan_id':a['scan_id'],'target_id':a['target_id'],'utterance':a['utterance']} for a in ds.annos]
(r/'annotation_manifest.json').write_text(json.dumps(rows,indent=2))
sample=ds[0]
assert sample['point_clouds'].shape==(50000,6)
assert len(sample['superpoint'])==50000
rec={'status':'complete','time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'rows':len(ds),'scenes':len(ds.scans),'seconds':time.time()-t,'annotations_sha256':hashlib.sha256((r/'annotations.pkl').read_bytes()).hexdigest(),'sample_shapes':{k:list(v.shape) for k,v in sample.items() if hasattr(v,'shape')},'sample_utterance':sample['utterances'],'model_forwards':0,'training_steps':0}
(r/'data_receipt.json').write_text(json.dumps(rec,indent=2));print('EG_DATA_READY '+json.dumps(rec),flush=True)
