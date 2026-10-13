"""Record current availability after exact duplicate local weight retirement."""
import hashlib
import json
from pathlib import Path

root = Path('\\\\?\\' + str(Path(__file__).resolve().parent))
receipt_path = root / 'LOCAL_EQUAL_E0_WEIGHT_DEDUP_RETIREMENT_20261014.json'
receipt = json.loads(receipt_path.read_bytes())
assert receipt['status'] == 'TWO_VERIFIED_EQUAL_MODEL_WEIGHT_CONTAINERS_RETIRED_LOCAL_ONLY'
assert receipt['released_bytes'] == 1229851934 and len(receipt['removed']) == 2
availability = {'receipt': str(receipt_path),
    'receipt_sha256': hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
    'retained_shared_full_model': receipt['protected_model'],
    'retired_full_containers': receipt['removed'],
    'all1295_model_tensors_identical_actual_CPU_check': True,
    'per_experiment_original_non_model_payloads_preserved': True,
    'original_experiment_containers_currently_materialized': False,
    'rehydrate_original_metadata_before_historical_checkpoint_load': True,
    'original_ZIP_bytes_not_claimed': True,
    'cold_normal_resume_verified': False, 'current_training_dependency': False}
paths = [root / 'CURRENT_PROGRESS.json', root.parent / 'NORMAL_CONTINUATION_STATE.json',
    root.parent / 'face_native_gpu_v2_20261011' / 'CURRENT_PROGRESS.json',
    Path('\\\\?\\C:\\Users\\gb\\.codex\\tmp\\mcln_portable_best_transfer_20261012\\CURRENT_PROGRESS.json'),
    Path('\\\\?\\C:\\Users\\gb\\.codex\\tmp\\pvground_referit_mask_reference_20261006\\current_research_goals.json'),
    Path('\\\\?\\C:\\Users\\gb\\.codex\\tmp\\pvground_query_supported_geometry_20261005\\active_continuation_state.json')]
for path in paths:
    state = json.loads(path.read_bytes())
    assert state['current_normal_arm'] == 'lower_span_lr'
    assert state['current_normal_E0_complete_metrics']['hits050'] == 4920
    state['local_equal_E0_weight_current_availability'] = availability
    state['current_normal_observer_functions_cell'] = '457'
    state['full_goal_complete'] = False
    path.write_bytes((json.dumps(state, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
print(json.dumps({'status': 'CURRENT_AVAILABILITY_RECORDED', 'continuation_files': len(paths),
    'released_bytes': receipt['released_bytes'], 'current_training_changed': False,
    'next_observer_functions_cell': '457'}))
