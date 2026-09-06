"""Count actual native Nr/Sr annotations without constructing point samples or a model."""
import ast
import csv
import datetime
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

root = Path(sys.argv[1])
source = Path('/root/autodl-tmp/mcln_candidate_local_native_preparation_20260906_v1/model_source')
data = Path('/root/autodl-tmp/DATA_ROOT')
source_raw = (source / 'native_source_manifest.json').read_bytes()
assert hashlib.sha256(source_raw).hexdigest() == '4af68cad46b52c9e250de17872a193485d75529378db983ad037779232e500fc'
for name, digest in json.loads(source_raw)['files'].items():
    assert hashlib.sha256((source / name).read_bytes()).hexdigest() == digest, name
os.chdir(str(source))
sys.path.insert(0, str(source))
import torch
from src.joint_det_dataset import Joint3DDataset, unpickle_data
from data.scannet_utils import read_label_mapping
assert not torch.cuda.is_available()
torch.set_num_threads(1)
started = time.time()

class AnnotationOnly(Joint3DDataset):
    def _scene_graph_parse(self, annos):
        # Text graph construction changes fields; this audit only measures row inclusion.
        pass

protocols = {'nr3d': {}, 'sr3d': {}}
selection = {}
inputs = {}
for name in ['refer_it_3d/nr3d.csv', 'refer_it_3d/sr3d.csv']:
    p = data / name
    inputs[str(p)] = {'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
for name in ['nr3d_train_scans.txt','nr3d_test_scans.txt','sr3d_train_scans.txt','sr3d_test_scans.txt','scannetv2_train.txt','scannetv2_val.txt','scannetv2-labels.combined.tsv']:
    p = source / 'data/meta_data' / name
    inputs[str(p)] = {'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
scenes = {}
for split in ['train', 'val']:
    p = data / (split + '_v3scans.pkl')
    pickle_metadata = {'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns, 'hashed': False}
    scan_lookup = next(unpickle_data(str(p)))
    missing = sorted(scan for scan in scan_lookup if not (data / 'superpoints' / split / (scan + '_superpoint.pth')).is_file())
    active_scans = {name: scan for name, scan in scan_lookup.items() if name not in set(missing)}
    dataset = object.__new__(AnnotationOnly)
    dataset.scans = active_scans
    dataset.data_path = str(data) + '/'
    dataset.split = split
    dataset.overfit = False
    dataset.use_sacr_source = False
    dataset.skip_missing_superpoints = True
    dataset.label_map = read_label_mapping('data/meta_data/scannetv2-labels.combined.tsv', label_from='raw_category', label_to='id')
    detection = dataset.load_annos('scannet')
    scenes[split] = {'pickle_metadata': pickle_metadata, 'pickle_scenes': len(scan_lookup),
                     'active_scenes': len(active_scans), 'missing_superpoint_scenes': missing,
                     'detection_base_rows': len(detection)}
    for dset in ['nr3d', 'sr3d']:
        annos = dataset.load_annos(dset)
        scan_ids = sorted({a['scan_id'] for a in annos})
        full = annos + detection * 10 if split == 'train' else annos
        full_scan_ids = sorted({a['scan_id'] for a in full})
        record = {'language_rows': len(annos), 'language_scans': len(scan_ids), 'language_scan_ids': scan_ids,
                  'joint_det_rows': len(detection) * 10 if split == 'train' else 0,
                  'total_rows': len(full), 'total_scan_ids': full_scan_ids,
                  'batches_b12_world1_drop_last': len(full) // 12 if split == 'train' else None,
                  'dropped_rows_b12_world1': len(full) % 12 if split == 'train' else None}
        protocols[dset][split] = record
        if split == 'train':
            picked, used = [], set()
            for row_id, anno in enumerate(annos):
                if anno['scan_id'] not in used:
                    used.add(anno['scan_id'])
                    picked.append(dict(anno, raw_language_row_id=row_id, raw_native_row_id=row_id))
                if len(picked) == 12:
                    break
            for det_index, anno in enumerate(detection[:4]):
                picked.append(dict(anno, raw_detection_row_id=det_index,
                                   raw_native_row_id=len(annos) + det_index))
            assert len(picked) == 16 and len({r['raw_native_row_id'] for r in picked}) == 16
            selection[dset] = picked
        print(json.dumps({'dataset': dset, 'split': split, 'language_rows': len(annos),
                          'joint_rows': len(full), 'language_scans': len(scan_ids)}), flush=True)
        del annos, full
    del dataset, active_scans, scan_lookup, detection
    gc.collect()
for dset in protocols:
    train = protocols[dset]['train']
    val = protocols[dset]['val']
    train_scans = set(train['total_scan_ids'])
    val_scans = set(val['total_scan_ids'])
    protocols[dset]['native_train_val_scan_overlap'] = sorted(train_scans & val_scans)
    protocols[dset]['native_train_val_physical_space_overlap'] = sorted({s.split('_')[0] for s in train_scans} & {s.split('_')[0] for s in val_scans})
assert protocols['nr3d']['train']['language_rows'] == 32919
assert protocols['nr3d']['val']['language_rows'] == 7899
assert protocols['sr3d']['train']['language_rows'] == 65846
assert protocols['sr3d']['val']['language_rows'] == 17726
for dset in protocols:
    assert not protocols[dset]['native_train_val_scan_overlap']
selection_raw = (json.dumps(selection, indent=2, sort_keys=True) + '\n').encode()
with (root / 'preflight_rows.json').open('xb') as stream:
    stream.write(selection_raw)
result = {'schema': 'mcln-native-joint-annotation-audit-v1', 'status': 'pass',
          'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
          'elapsed_seconds': time.time() - started, 'model_source': str(source),
          'source_manifest_sha256': hashlib.sha256(source_raw).hexdigest(),
          'annotations_and_split_files': inputs, 'scans': scenes, 'protocols': protocols,
          'preflight_rows_sha256': hashlib.sha256(selection_raw).hexdigest(),
          'preflight_selection_rule': 'First language row from first 12 distinct scenes in native annotation order, then first 4 base detection rows; no performance-based selection.',
          'gpu_forwards': 0, 'point_samples_constructed': 0, 'model_updates': 0, 'checkpoint_writes': 0,
          'limits': 'Native annotation loaders and actual scan object labels; skips text graph field construction and per-point sampling. Counts are not training or validation performance. Pickle identity recorded by size/mtime, not a new full-file hash.'}
with (root / 'receipt.json').open('x') as stream:
    json.dump(result, stream, indent=2, sort_keys=True)
print(json.dumps({key: result[key] for key in ['status','time_cst','elapsed_seconds','gpu_forwards','point_samples_constructed','preflight_rows_sha256']}), flush=True)

