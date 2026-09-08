import datetime
import hashlib
import importlib.util
import json
from pathlib import Path

manifest_path = Path('/root/autodl-tmp/mcln_scanrefer_object_appearance_pair_20260908_v1/input_manifest.json')
manifest = json.loads(manifest_path.read_bytes())
source = Path(manifest['model_source'])
source_manifest = source/'appearance_source_manifest.json'
assert hashlib.sha256(source_manifest.read_bytes()).hexdigest() == manifest['source_manifest_sha256']
files = json.loads(source_manifest.read_bytes())['files']
for name in ['src/joint_det_dataset.py', 'scripts/scanrefer_data_contract.py']:
    assert hashlib.sha256((source/name).read_bytes()).hexdigest() == files[name]
module_spec = importlib.util.spec_from_file_location('input_contract', str(source/'scripts/scanrefer_data_contract.py'))
module = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(module)
contract = module.verify_scanrefer_superpoints(manifest['data_root'], 'val', manifest['superpoint_files']['val'])
data = Path(manifest['data_root'])
scanrefer = data/'ScanRefer'
if not scanrefer.is_dir():
    scanrefer = data/'scanrefer'
split_path = scanrefer/'ScanRefer_filtered_val.txt'
annotation_path = scanrefer/'ScanRefer_filtered_val.json'
scan_ids = set(split_path.read_text().splitlines())
annotations = json.loads(annotation_path.read_bytes())
selected = [row for row in annotations if row['scene_id'] in scan_ids]
assert len(selected) == 9508
identities = [(row['scene_id'], int(row['object_id']), str(row.get('ann_id','')), ' '.join(row['token'])) for row in selected]
result = {'status': 'pass', 'scope': 'input-contract-only; no model evaluation',
          'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
          'data_root': manifest['data_root'], 'dataset_source': str(source),
          'source_manifest_sha256': manifest['source_manifest_sha256'],
          'val_superpoints': contract, 'expected_formal_rows': len(identities),
          'expression_scene_count': len({row[0] for row in identities}), 'annotation_path': str(annotation_path),
          'annotation_sha256': hashlib.sha256(annotation_path.read_bytes()).hexdigest(),
          'scene_split_path': str(split_path), 'scene_split_sha256': hashlib.sha256(split_path.read_bytes()).hexdigest(),
          'raw_identity_order_sha256': hashlib.sha256(json.dumps(identities,ensure_ascii=False,separators=(',',':')).encode()).hexdigest(),
          'primary_mode': 'bbs', 'size_policy': 'author clamp min 1e-6', 'butd': True, 'butd_cls': False,
          'formal_rows_executed': 0, 'optimizer_steps': 0}
print(json.dumps(result, indent=2))
