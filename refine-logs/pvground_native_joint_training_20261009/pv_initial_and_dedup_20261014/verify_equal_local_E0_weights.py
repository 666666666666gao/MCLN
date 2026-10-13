"""Verify redundant closed E0 containers using actual CPU-loaded tensors."""
import datetime
import hashlib
import json
from pathlib import Path

import torch

root = Path('\\\\?\\' + str(Path(__file__).resolve().parent))
archives = Path('C:/Users/gb/.codex/archives')
protected = archives / 'pvg_native_normal_20261010' / 'best.pth'
protected_sha = 'ed8455ddc67e4d17018e9cf499197140e15e35db3f5daee55e0eec6846faf0c4'
assert hashlib.sha256(protected.read_bytes()).hexdigest() == protected_sha
state = torch.load(str(protected), map_location='cpu')
assert state['epoch'] == 0 and len(state['model']) == 1295

def identity(tensor):
    assert torch.is_tensor(tensor)
    array = tensor.detach().cpu().numpy()
    return {'dtype': str(array.dtype), 'shape': list(array.shape),
            'sha256': hashlib.sha256(array.tobytes()).hexdigest()}

manifest = {key: identity(value) for key, value in state['model'].items()}
verified = []
for folder in ('pvg_geometry_control_20261013_reconstructed',
               'pvg_selected_geometry_20261014_reconstructed'):
    directory = archives / folder
    receipt_path = directory / 'ARCHIVE_RECEIPT.json'
    receipt = json.loads(receipt_path.read_bytes())
    target = directory / 'best.pth'
    metadata = directory / 'original_non_model_state.pth'
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    assert digest == receipt['sha256']
    assert target.stat().st_size == receipt['bytes'] == 614925967
    metadata_sha = hashlib.sha256(metadata.read_bytes()).hexdigest()
    assert metadata_sha == receipt['original_non_model_sha256']
    assert receipt['original_model_manifest'] == manifest
    archived = torch.load(str(target), map_location='cpu')
    actual_manifest = {key: identity(value) for key, value in archived['model'].items()}
    assert actual_manifest == manifest
    assert list(archived['model']) == list(state['model'])
    payload = torch.load(str(metadata), map_location='cpu')
    assert set(payload) == {'non_model', 'model_metadata'}
    assert set(payload['non_model']) == set(archived) - {'model'}
    assert payload['model_metadata'] == archived['model']._metadata
    assert payload['non_model']['epoch'] == archived['epoch'] == 0
    assert not payload['non_model']['optimizer']['state']
    assert not archived['optimizer']['state']
    verified.append({'path': str(target), 'bytes': target.stat().st_size,
        'sha256': digest, 'model_tensors': 1295, 'all_actual_model_tensors_exact': True,
        'model_namespace_and_order_exact': True,
        'original_non_model_payload': str(metadata),
        'original_non_model_sha256': metadata_sha,
        'original_non_model_keys_preserved': sorted(payload['non_model']),
        'original_model_metadata_exact': True,
        'source_archive_receipt_sha256': hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        'normal_training_dependency': False})
    del archived, payload
assert hashlib.sha256(protected.read_bytes()).hexdigest() == protected_sha
result = {'status': 'ACTUAL_1295_MODEL_TENSORS_EXACT_TWO_REDUNDANT_CLOSED_E0_CONTAINERS',
    'verified_cst': datetime.datetime.now().astimezone().isoformat(),
    'protected_model': {'path': str(protected), 'bytes': protected.stat().st_size,
        'sha256': protected_sha, 'model_tensors': 1295, 'epoch': 0},
    'retirement_candidates': verified,
    'expected_released_bytes': sum(row['bytes'] for row in verified),
    'checkpoint_reconstruction': 'Use protected model OrderedDict, attach each original model_metadata, and combine original non_model payload; existing archive CPU reconstructors implement this.',
    'original_ZIP_byte_recovery_not_claimed': True,
    'cold_native_training_resume_verified': False,
    'neural_forwards': 0, 'model_factories_built': 0, 'optimizer_updates': 0,
    'remote_queries': 0, 'files_deleted_by_this_verifier': 0,
    'current_normal_training_changed': False, 'best_metric_changed': False,
    'full_goal_complete': False}
path = root / 'LOCAL_EQUAL_E0_WEIGHT_DEDUP_VERIFICATION_20261014.json'
assert not path.exists()
path.write_bytes((json.dumps(result, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
print(json.dumps({'status': result['status'], 'actual_model_files_loaded': 3,
    'tensor_count_per_model': 1295, 'metadata_payloads_preserved': 2,
    'retirement_candidates': len(verified), 'expected_released_bytes': result['expected_released_bytes'],
    'neural_forwards': 0, 'files_deleted': 0}))
