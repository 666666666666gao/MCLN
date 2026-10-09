"""Bind known initialization identities; this does not load a model or launch."""
import datetime
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
span = root.parent / 'pvground_extremal_span_evidence_20261009/runner_v1'
spec = json.loads((span / 'pair_spec.json').read_bytes())
assert spec['parent_selection_status'] == 'FINALIZED_AFTER_PRIOR_PAIR_CLOSED'
assert spec['starting_hits'] == [5606, 4881]
payload = dict(
    dataset='scanrefer', seed=2027,
    official_checkpoint='/root/autodl-tmp/mcln_pvground_scan_checkpoint_inspection_20260908_v1/PV-Ground_ScanRefer.pth',
    official_checkpoint_sha256=spec['checkpoint_sha256'],
    g_checkpoint=spec['base_terminal'], g_checkpoint_sha256=spec['base_terminal_sha256'],
    support_checkpoint=spec['parent_support_terminal'],
    support_checkpoint_sha256=spec['parent_support_terminal_sha256'],
    span_checkpoint=None, span_checkpoint_sha256=None,
    use_g_supervision=True, use_selected_mask_supervision=True,
    parent_retention_decision_sha256=spec['parent_decision_sha256'],
    source_provenance_spec_sha256=hashlib.sha256((span / 'pair_spec.json').read_bytes()).hexdigest(),
    env_spec_sha256=spec['env_spec_sha256'],
    support_prior_updates=11169,
    new_span_output_zero_initialized=True,
    mode='SOURCE_ONLY_ZERO_SPAN_INITIALIZATION_PENDING_M0_NOT_A_DEPLOYMENT',
    normal_joint_training_started=False,
    source_data_overlay_admitted=False,
    actual_constructor_or_payload_loading_verified=False,
    initial_formal_accuracy_verified=False,
    normal_joint_training_recipe='NORMAL_TRAINING_PLAN.md',
    obsolete_zero_geometry_and_R_checkpoint_not_required=True)
out = root / 'init_manifests'
out.mkdir(exist_ok=False)
paths = []
for mode in ('whole_support', 'extremal_support'):
    record = dict(payload, span_source_mode=mode)
    path = out / (mode + '.json')
    path.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    paths.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
receipt = dict(status='NATIVE_INIT_MANIFESTS_SOURCE_ONLY',
    time_cst=datetime.datetime.now().astimezone().isoformat(),
    files=paths, actual_model_construction=False, source_runtime_binding=False,
    gpu_preflight=False, native_joint_training_started=False,
    current_active_pair_modified=False, parent_identity_changed=False,
    remote_queries=0, new_accuracy_result=False, full_goal_complete=False)
(root / 'INIT_MANIFEST_PREPARATION.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(status=receipt['status'], manifests=len(paths), remote_queries=0,
                     neural_calls=0, native_joint_training_started=False)))
