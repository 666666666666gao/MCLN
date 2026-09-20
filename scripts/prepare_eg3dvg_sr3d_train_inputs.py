"""Prepare unchanged author Sr3D training annotations on CPU; no model execution."""
import datetime
import hashlib
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parent
acceptance = Path('/root/autodl-tmp/mcln_eg3dvg_acceptance_20260920_v1')
source = acceptance / 'referit_input_source'
data = Path('/root/autodl-tmp/DATA_ROOT_mcln_meshsp')
output = root / 'train_input_cache'
output.mkdir()
os.chdir(str(source))
sys.path.insert(0, str(source))
sys.path.insert(1, str(source / 'pointnet2'))
import numpy as np
from src.joint_det_dataset import Joint3DDataset, unpickle_data, read_label_mapping

random.seed(2027)
np.random.seed(2027)
loader = Joint3DDataset.__new__(Joint3DDataset)
loader.split = 'train'
loader.overfit = False
loader.data_path = str(acceptance / 'referit_input_cache/data_view') + '/'
loader.scans = list(unpickle_data(str(data / 'train_v3scans.pkl')))[0]
loader.label_map = read_label_mapping('data/meta_data/scannetv2-labels.combined.tsv', label_from='raw_category', label_to='id')
started = time.time()
print('SR_TRAIN_ANNOTATIONS_BEGIN', flush=True)
annotations = loader.load_annos('sr3d')
scenes = sorted({a['scan_id'] for a in annotations})
assert len(annotations) > 0
assert all(s in loader.scans for s in scenes)
assert all((data / 'superpoints/train' / (s + '_superpoint.pth')).is_file() for s in scenes)
assert all(0 <= a['target_id'] < len(loader.scans[a['scan_id']].three_d_objects) for a in annotations)
with (acceptance / 'referit_input_cache/sr3d_annotations.pkl').open('rb') as f:
    validation = pickle.load(f)
val_scenes = {a['scan_id'] for a in validation}
assert not set(scenes) & val_scenes
path = output / 'sr3d_annotations.pkl'
with path.open('xb') as f:
    pickle.dump(annotations, f, protocol=4)
relation_counts = {}
for a in annotations:
    relation = loader._find_rel(a['utterance'])
    relation_counts[relation] = relation_counts.get(relation, 0) + 1
report = {'status': 'complete', 'dataset': 'sr3d', 'rows': len(annotations), 'scenes': len(scenes),
          'validation_rows': len(validation), 'train_validation_scenes_disjoint': True,
          'seconds': time.time()-started, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
          'source_sha256': hashlib.sha256((source/'src/joint_det_dataset.py').read_bytes()).hexdigest(),
          'relation_counts': relation_counts, 'model_forwards': 0, 'optimizer_steps': 0,
          'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
          'scope': 'Unchanged author load_annos and scene graph parser; no Sr model evaluation or training.'}
(output / 'receipt.json').write_text(json.dumps(report, indent=2))
print('SR_TRAIN_ANNOTATIONS_COMPLETE '+json.dumps(report), flush=True)
