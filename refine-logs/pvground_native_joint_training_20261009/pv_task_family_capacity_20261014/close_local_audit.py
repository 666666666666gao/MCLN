"""Record a completed local-only audit without touching the live experiment."""
import datetime
import hashlib
import json
from pathlib import Path

root = Path('\\\\?\\' + str(Path(__file__).resolve().parent))
audit = root / 'selected_box_family_audit_20261014'
seal_path = audit / 'REVIEW_COMPLETE.json'
seal_raw = seal_path.read_bytes()
assert len(seal_raw) == 26743
assert hashlib.sha256(seal_raw).hexdigest() == 'dfa0d1d2362df562f5bc70f0d754979a1806041572cadc714da372b194776d75'
seal = json.loads(seal_raw)
assert seal['status'] == 'REVIEW_COMPLETE' and seal['verdict'] == 'WARN'
assert seal['blocking_findings'] == []
assert seal['scoped_saved_geometry_recount'] == 'PASS'
assert seal['scoped_family_maximum'] == 'PASS'
assert seal['scoped_source_path_verification'] == 'PASS_SOURCE_ONLY'
assert seal['input_count'] == 47 and seal['output_count'] == 28
assert seal['numerical_checks'] == 142 and seal['exact_rational_rows'] == 9
assert seal['final_inputs_unchanged_at_seal'] and seal['all_scientific_inputs_unchanged']
assert not seal['all_initial_input_bytes_unchanged']
assert seal['actual_model'] == 'UNATTESTED'
assert seal['review_independence'] == 'same-family' and seal['acceptance_status'] == 'provisional'
assert seal['remote_queries'] == 0 and seal['GPU_queries'] == 0
assert seal['neural_forwards'] == 0 and seal['optimizer_updates'] == 0
for entry in seal['inputs'] + seal['outputs']:
    name = entry['path']
    path = Path(name if name.startswith('\\\\?\\') else '\\\\?\\' + name)
    raw = path.read_bytes()
    assert len(raw) == entry['bytes'], name
    assert hashlib.sha256(raw).hexdigest() == entry['sha256'], name

capacity = json.loads((root / 'selected_family_capacity_20261014' / 'RESULT.json').read_bytes())
assert [row['rows'] for row in capacity['epochs']] == [9508] * 3
assert [row['050']['GT_only_independent_axis_family_upper_bound_hits'] for row in capacity['epochs']] == [5379, 5335, 5208]
owner = json.loads((root / 'OBS_lower_span_lr_e1_near_complete_OWNER.json').read_bytes())
assert owner['local_pid'] == 26676 and owner['due_cst'] == '2026-10-14T08:15:00+08:00'
receipt = dict(
    status='LOCAL_SELECTED_FAMILY_AUDIT_SEALED_NO_NEW_MODEL_RESULT',
    closed_at_cst=datetime.datetime.now().astimezone().isoformat(),
    reviewer_completed_at=seal['completed_at'],
    seal_path=str(seal_path), seal_bytes=len(seal_raw),
    seal_sha256=hashlib.sha256(seal_raw).hexdigest(),
    verified_input_files=47, verified_output_files=28,
    numerical_checks=142, exact_rational_rows=9,
    verdict=seal['verdict'], blocking_findings=seal['blocking_findings'],
    scoped_saved_geometry_recount=seal['scoped_saved_geometry_recount'],
    scoped_family_maximum=seal['scoped_family_maximum'],
    scoped_source_path_verification=seal['scoped_source_path_verification'],
    review_independence=seal['review_independence'], acceptance_status=seal['acceptance_status'],
    requested_model=seal['requested_model'], actual_model=seal['actual_model'],
    scientific_inputs_unchanged=True, report_wording_revised=True,
    original_dataset_reloaded=False, full256_arrays_recounted=False,
    new_model_performance=False, deployable_or_learnable_gain_established=False,
    active_training_source_changed=False, training_restarts=0, remote_queries=0,
    new_observers=0, neural_forwards=0, optimizer_updates=0,
    retained_best_hits=[5677, 4920], goal_hits=[5705, 4945],
    current_normal_controller_pid=432868, current_normal_child_pid=432871,
    last_actual_remote_observation_cst='2026-10-14T03:20:49.600024+08:00',
    current_trained_formal_metrics=None,
    sole_observer_pid=26676, sole_observer_native_session=11018,
    sole_observer_functions_cell='457', next_observation_cst=owner['due_cst'],
    no_remote_progress_queries_before_next_due=True,
    independent_three_module_gains_established=False, full_goal_complete=False)
receipt_path = root / 'SELECTED_FAMILY_AUDIT_CLOSURE_20261014.json'
assert not receipt_path.exists()
receipt_path.write_bytes((json.dumps(receipt, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
states = [root / 'CURRENT_PROGRESS.json', root.parent / 'NORMAL_CONTINUATION_STATE.json',
    root.parent / 'face_native_gpu_v2_20261011' / 'CURRENT_PROGRESS.json',
    Path('\\\\?\\C:\\Users\\gb\\.codex\\tmp\\mcln_portable_best_transfer_20261012\\CURRENT_PROGRESS.json'),
    Path('\\\\?\\C:\\Users\\gb\\.codex\\tmp\\pvground_referit_mask_reference_20261006\\current_research_goals.json'),
    Path('\\\\?\\C:\\Users\\gb\\.codex\\tmp\\pvground_query_supported_geometry_20261005\\active_continuation_state.json')]
for path in states:
    state = json.loads(path.read_bytes())
    assert state['current_normal_controller_pid'] == 432868 and state['current_normal_child_pid'] == 432871
    assert state['next_observation_cst'] == owner['due_cst']
    state.update(selected_family_audit_completed_record=receipt,
        selected_family_audit_closure_path=str(receipt_path), full_goal_complete=False)
    path.write_bytes((json.dumps(state, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
print(json.dumps(dict(status=receipt['status'], verified_bindings=75,
    verdict=receipt['verdict'], continuation_files_updated=len(states),
    next_observation_cst=receipt['next_observation_cst'], new_NN_results=0)), flush=True)
