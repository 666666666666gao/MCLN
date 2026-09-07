import hashlib,json,os,sys,time
from pathlib import Path
root=Path('/root/autodl-tmp/mcln_referit3d_pretrained_slots_20260908_v1')
plan=json.loads((root/'manifest.json').read_text())
source=Path(plan['model_source']);os.chdir(str(source));sys.path.insert(0,str(source))
import torch
torch.set_num_threads(1)
from src.joint_det_dataset import Joint3DDataset,unpickle_data,read_label_mapping
start=time.time()
dataset=object.__new__(Joint3DDataset)
dataset.split='train';dataset.augment=False
dataset.label_map=read_label_mapping('data/meta_data/scannetv2-labels.combined.tsv',label_from='raw_category',label_to='id')
dataset.scans=list(unpickle_data(str(Path(plan['data_root'])/'train_v3scans.pkl')))[0]
annos=dataset.load_scannet_annos()
slots=json.loads((root/'scene_slots.json').read_text())
language=set(x['scan_id'] for x in slots);detection=set(x['scan_id'] for x in annos)
src=source/'src/joint_det_dataset.py'
cls=json.loads((source/'data/cls_results.json').read_text())
print(json.dumps(dict(detection_rows=len(annos),detection_unique_scenes=len(detection),
 language_scenes=len(language),intersection=len(language & detection),
 missing_detection_scenes=sorted(detection-language),language_only_scenes=sorted(language-detection),
 union_scenes=len(language|detection),missing_predicted_class_scenes=sorted((language|detection)-set(cls)),
 native_source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
 scannet_train_list_sha256=hashlib.sha256((source/'data/meta_data/scannetv2_train.txt').read_bytes()).hexdigest(),
 elapsed_seconds=time.time()-start,model_forwards=0,optimizer_steps=0)))
