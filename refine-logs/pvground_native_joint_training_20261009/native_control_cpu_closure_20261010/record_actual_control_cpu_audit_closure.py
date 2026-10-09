import datetime
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
audit_root = root / 'native_direct_controls_20261010/actual_CPU_review'
audit = json.loads((audit_root / 'EXPERIMENT_AUDIT.json').read_bytes())
seal = json.loads((audit_root / 'SEAL.json').read_bytes())
assert audit['verdict'] == 'WARN' and audit['blocking_issue_count'] == 0
assert audit['execution_scope'] == 'CLOSED_CPU_SYNTHETIC_MODULE_ENGINEERING_NOT_PV_OR_ACCURACY'
assert audit['gpu_admission'] is False and audit['formal_accuracy_approved'] is False
assert len(audit['audited_input_hashes']) == 31
assert len(audit['hash_only_prior_source_inputs']) == 37
for inputs in (audit['audited_input_hashes'], audit['hash_only_prior_source_inputs']):
    for name, digest in inputs.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest.removeprefix('sha256:')
for name, digest in seal['artifact_hashes'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest.removeprefix('sha256:')
prior = json.loads((root / 'STATIC_COPY_AND_CPU_RECOVERY.json').read_bytes())
report = dict(status='ACTUAL_SYNTHETIC_CPU_MODULE_EXECUTION_REVIEWED_WITH_BOUNDED_WARN',
    time_cst=datetime.datetime.now().astimezone().isoformat(),
    previous_execution_record=str(root / 'STATIC_COPY_AND_CPU_RECOVERY.json'),
    execution_finished_cst=prior['CPU_execution_finished_cst'],
    actual_audit=str(audit_root / 'EXPERIMENT_AUDIT.json'),
    actual_audit_sha256=hashlib.sha256((audit_root / 'EXPERIMENT_AUDIT.json').read_bytes()).hexdigest(),
    audit_verdict=audit['verdict'], blocking_issue_count=0,
    audited_input_hashes_verified=31, continuity_only_hashes_verified=37,
    actual_CPU_outcome_review_pending=False,
    actual_reviewer_model=audit['actual_reviewer_model'],
    review_independence=audit['review_independence'], acceptance_status=audit['acceptance_status'],
    additional_bootstrap_context_read=True, bootstrap_contents_published=False,
    witness_is_reencoded_same_stdout_not_independent=True,
    fixed_half_gradient_is_independent_reference_partial_derivative_only=True,
    full_PV_native_loss_optimizer_recovery_checked=False,
    GPU_or_control_training_admission=False, normal_training_status_queries=0,
    normal_training_restarts=0, new_normal_accuracy=None, full_goal_complete=False)
target = root / 'ACTUAL_CONTROL_CPU_AUDIT_CLOSURE.json'
assert not target.exists()
target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
for path in (root.parent.parent / 'pvground_referit_mask_reference_20261006/current_research_goals.json',
             root.parent.parent / 'pvground_query_supported_geometry_20261005/active_continuation_state.json'):
    value = json.loads(path.read_bytes())
    assert value['latest_handoff_section'] == '20.376.130'
    value.update(current_turn_classification='ACTUAL_PROGRESS_CPU_OUTCOME_REVIEW_CLOSED_STATIC_PUBLICATION_PENDING',
                 full_goal_complete=False)
    value['remaining_goal_constraints'] = [
        'Ordinary native joint-training accuracy and direct mechanism effectiveness remain unverified',
        'Three effective contributions with targeted direct controls remain unestablished',
        'Same final complete model independently train/evaluate author-initialized Nr3D/Sr3D']
    value['normal_training_requirement'].update(
        current_native_control_CPU_outcome_review=report,
        current_native_control_CPU_outcome_review_pending=False,
        user_prefers_changes_in_training_framework=True, user_prefers_avoid_postprocessing=True,
        latest_user_requirement='优先在原生网络前向与正常训练损失中实现针对性改进，避免评估后再改框或另加排名',
        next_action='At the original scheduled 08:19 first observation inspect the running native experiment; keep prepared controls isolated until their full-model checks are required')
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
note = ('\nPV-Ground ' + report['time_cst'] + ': Actual separate synthetic CPU outcome audit closed WARN, '
    '0 bounded engineering blocks, 31 reviewed+37 hash-only continuity inputs rechecked unchanged. '
    'UNATTESTED/same-family/provisional; bootstrap filenames disclosed but no personal contents published. '
    'Single stdout and reencoded witness not independent; half-gradient limited to independent references. '
    'No full PV/native criterion/optimizer/recovery/accuracy or GPU admission. '
    'Latest user preference remains ordinary forward/criterion and avoidance of postprocessing. '
    'Normal configuration unchanged; original 08:19 observer remains next training check. Full goal ACTIVE_UNMET.\n')
for path in (Path('C:/Users/gb/memory/2026-10-10.md'), Path('C:/Users/gb/MEMORY.md')):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(note)
print(json.dumps({k: report[k] for k in ('status', 'audit_verdict', 'blocking_issue_count',
    'audited_input_hashes_verified', 'continuity_only_hashes_verified', 'new_normal_accuracy')}))
