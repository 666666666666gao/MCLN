"""Exercise the full author Sr3D training Dataset and fixed augmented samples on CPU."""
import hashlib,json,os,random,shutil,sys,time
from pathlib import Path
r=Path(__file__).resolve().parent
source=Path('/root/autodl-tmp/mcln_eg3dvg_acceptance_20260920_v1/referit_input_source')
dest=r/'train_input_source'
assert not dest.exists()
shutil.copytree(str(source),str(dest),ignore=shutil.ignore_patterns('__pycache__'))
p=dest/'src/joint_det_dataset.py'
text=p.read_text()
old="assert self.split == 'val' and test_dataset in ['nr3d', 'sr3d']\n        with open('/root/autodl-tmp/mcln_eg3dvg_acceptance_20260920_v1/referit_input_cache/' + test_dataset + '_annotations.pkl', 'rb') as file:\n            self.annos = pickle.load(file)"
new="assert self.split == 'train' and test_dataset == 'sr3d'\n        assert dataset_dict == {'sr3d': 1, 'scannet': 10}\n        with open('"+str(r/'train_input_cache/sr3d_annotations.pkl')+"', 'rb') as file:\n            self.annos = pickle.load(file)\n        with open('/root/autodl-tmp/mcln_eg3dvg_nr3d_transfer_20260920_v1/train_input_cache/scannet_annotations.pkl', 'rb') as file:\n            self.annos += pickle.load(file) * 10"
assert text.count(old)==1
p.write_text(text.replace(old,new))
os.chdir(str(dest));sys.path.insert(0,str(dest));sys.path.insert(1,str(dest/'pointnet2'))
import numpy as np
import torch
from src.joint_det_dataset import Joint3DDataset
random.seed(2027);np.random.seed(2027);torch.manual_seed(2027)
receipt=json.loads((r/'train_input_cache/receipt.json').read_text())
sr=receipt['rows']
det=json.loads(Path('/root/autodl-tmp/mcln_eg3dvg_nr3d_transfer_20260920_v1/train_input_cache/receipt.json').read_text())['datasets']['scannet']['rows']
start=time.time()
dataset=Joint3DDataset(dataset_dict={'sr3d':1,'scannet':10},test_dataset='sr3d',split='train',data_path='/root/autodl-tmp/DATA_ROOT_mcln_meshsp/',use_color=True,detect_intermediate=True,butd_cls=True)
assert len(dataset)==sr+10*det and dataset.augment and dataset.butd_cls and not dataset.butd and not dataset.butd_gt
samples=[]
for index in [0,sr//7,sr//3,sr//2,2*sr//3,3*sr//4,sr-2,sr-1,sr,sr+det//2,sr+det-1]:
    item=dataset[index]
    assert item['point_clouds'].shape==(50000,6) and item['all_detected_boxes'].shape==(132,6)
    assert np.isfinite(item['point_clouds']).all() and np.isfinite(item['all_detected_boxes']).all()
    assert np.array_equal(item['all_detected_boxes'],item['all_bboxes'])
    assert np.array_equal(item['all_detected_bbox_label_mask'],item['all_bbox_label_mask'])
    assert np.isfinite(item['positive_map']).all() and np.isfinite(item['negative_positive_map']).all()
    samples.append({'index':index,'dataset':dataset.annos[index]['dataset'],'scan_id':item['scan_ids'],
                    'utterance':item['utterances'],'negative_utterance':item['negative_utterances'],
                    'object_slots':int(item['all_detected_bbox_label_mask'].sum()),
                    'point_sha256':hashlib.sha256(item['point_clouds'].tobytes()).hexdigest()})
report={'status':'complete','model_forwards':0,'optimizer_steps':0,'rows':len(dataset),'sr3d_rows':sr,
        'scannet_rows_before_repeat':det,'scannet_repeat':10,'seconds':time.time()-start,
        'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'samples':samples,
        'augmentation_changed':False,'scope':'Full author augmented Sr3D train Dataset plus fixed eight Sr and three ScanNet reads; no GPU model.'}
(r/'train_dataset_preflight.json').write_text(json.dumps(report,indent=2))
print('SR_TRAIN_DATASET_COMPLETE '+json.dumps(report),flush=True)
