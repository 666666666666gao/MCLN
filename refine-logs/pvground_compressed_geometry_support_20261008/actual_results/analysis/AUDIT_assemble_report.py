"""Assemble this auditor's verdict from completed independent checks and exact inputs."""
import datetime
import hashlib
import json
from pathlib import Path

A = Path(__file__).resolve().parent
W = A.parent
P = W.parent / 'pvground_mask_support_correction_20261008_v2/complete_fit/PV-Ground'
H = W.parent / 'pvground_final_quality_20261005/runtime_bundle'
D = W.parent / 'pvground_g_p2_20261002/complete/source/joint_det_dataset.py'
TRACE = 'C:/Users/gb/.codex_mcln_g0_20260905/.aris/traces/experiment-audit/2026-10-08_run07'
AGENT = '/root/pvg_compressed_support_closed_actual_audit'


def digest(path):
    raw = path.read_bytes()
    return hashlib.sha256(raw).hexdigest(), len(raw)


def ref(path, line):
    return f'{path.as_posix()}:{line}'


def main():
    cpu = json.loads((A / 'AUDIT_independent_cpu.json').read_bytes())
    source = json.loads((A / 'AUDIT_source_checks.json').read_bytes())
    assert cpu['status'] == 'PASS_DETERMINISTIC_CHECKS_WITH_EXPLICIT_SCOPE_LIMITS'
    assert source['status'] == 'PASS_SCOPED_SOURCE_AND_EXECUTED_RECEIPT_BINDING'
    reviewed = {}
    for name in ('AUDIT_machine_reviewed_files.json', 'AUDIT_source_reviewed_files.json'):
        for row in json.loads((A / name).read_bytes()):
            path = Path(row['path']).resolve()
            sha, size = digest(path)
            assert sha == row['sha256'] and size == row['bytes'], str(path)
            key = str(path)
            scopes = set(reviewed.get(key, {}).get('review_scopes', []))
            scopes.add(row['review_scope'])
            reviewed[key] = dict(row, path=key, review_scopes=sorted(scopes))
    audit_evidence = []
    for name in ('AUDIT_independent_cpu.py', 'AUDIT_independent_cpu.json',
                 'AUDIT_machine_reviewed_files.json', 'AUDIT_source_checks.py',
                 'AUDIT_source_checks.json', 'AUDIT_source_reviewed_files.json',
                 'AUDIT_initial_execution.json', 'AUDIT_EXECUTION.md',
                 'AUDIT_assemble_report.py', 'EXPERIMENT_AUDIT.md'):
        path = (A / name).resolve()
        sha, size = digest(path)
        row = dict(path=str(path), sha256=sha, bytes=size,
                   review_scope='auditor_generated_execution_evidence_or_report',
                   review_scopes=['auditor_generated_execution_evidence_or_report'])
        reviewed[str(path)] = row
        audit_evidence.append(row)
    required = [(W / 'pair_spec.json').resolve(), (A / 'SUMMARY.json').resolve(),
                (W / 'complete_fit/INTAKE.json').resolve()]
    assert all(str(path) in reviewed for path in required)
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    checks = {
        'A_gt_provenance': {
            'status': 'WARN',
            'details': 'Dataset-derived instance GT; actual imported detection-aligned dataset source is hash-bound. Historical/initial/final identities and saved root GT align exactly. Original raw data and original imported Scan helper were not rebuilt or independently hash-bound. Native object/language-map protocol is not globally GT-free.',
            'evidence': [ref(D, 1086), ref(D, 1099), ref(D, 1105),
                         ref(W / 'run_mask_support_pair.py', 99),
                         ref(W / 'complete_fit/imports.json', 16),
                         ref(P / 'src/grounding_evaluator.py', 251),
                         ref(W / 'paired_support_loop.py', 304)],
        },
        'B_score_normalization_and_scope': {
            'status': 'PASS',
            'details': 'Fixed 9508-expression denominator, strict >0.25 and >0.50 thresholds, native last_/bbs selection, all 256 candidates, no added rank/quality mixing or prediction normalization. Selected Box and Mask use the same Query. Four training coefficients are 5/1/10/2, normalized by actual GT count.',
            'evidence': [ref(H / 'native_root_bbs.py', 4),
                         ref(W / 'support_pair_forward.py', 14),
                         ref(W / 'paired_support_loop.py', 321),
                         ref(W / 'paired_support_loop.py', 331),
                         ref(W / 'paired_support_loop.py', 351),
                         ref(W / 'matched_mask_objective.py', 35),
                         ref(W / 'matched_mask_objective.py', 51)],
        },
        'C_results_and_archive': {
            'status': 'WARN',
            'details': 'All 2423 archived files/230030167 bytes match hashes and size; 2378 NPZ files fully checked. Independent float64 recomputation covers 14604288 candidate IoUs. Every selected and oracle threshold label agrees. Native IoUs have small >1 float32 excursions, disclosed without clamping. Masks are saved-IoU recounts, not raw-mask recomputation.',
            'evidence': [ref(W / 'complete_fit/INTAKE.json', 1),
                         ref(A / 'AUDIT_independent_cpu.json', 1),
                         ref(A / 'SUMMARY.json', 1),
                         ref(W / 'complete_fit/formal/receipt.json', 1),
                         ref(A / 'AUDIT_initial_execution.json', 1)],
        },
        'D_executed_paths_and_restore': {
            'status': 'WARN',
            'details': 'Actual closed receipts, logs and all training records support 3723 new updates/29778 unique samples per arm, genuine warm start/7446 cumulative support updates and new optimizers. Losses/gradient norms finite, original matched supervision and one shared parent forward recorded. Actual M0 covers same-cache native GPU integration and CPU reconstruction, not historical formal cold GPU reconstruction. Corrected postrun terminal inspection remains pending.',
            'evidence': [ref(W / 'paired_support_loop.py', 35),
                         ref(W / 'paired_support_loop.py', 60),
                         ref(W / 'paired_support_loop.py', 101),
                         ref(W / 'paired_support_loop.py', 205),
                         ref(W / 'complete_fit/train.jsonl', 3723),
                         ref(W / 'complete_fit/preflight.json', 8),
                         ref(W / 'postrun/inspect_closed_terminals_rng_replay_authorized.py', 96)],
        },
        'E_comparison_selection_and_cleanup': {
            'status': 'WARN',
            'details': 'Both terminal arms tie at 5599/4859 with zero paired geometry repairs/damages. Content is an eligible tied metric-best under the immutable historical rule; ARMS order picks content. Independent-pass drift prevents assigning historical +1/+3 entirely to support training. Latest 5658/4850 joint target is unmet by 59 wide-threshold hits. Three effective contributions are unestablished. Pending exact cleanup must preserve six restoration dependencies and the old warm-start local archive.',
            'evidence': [ref(W / 'EXPERIMENT_PLAN.md', 17),
                         ref(W / 'pair_spec.json', 5),
                         ref(W / 'CURRENT_RESEARCH_GOALS.json', 8),
                         ref(W / 'postrun/analyze_compressed_geometry_formal.py', 200),
                         ref(P / 'models/pv_ground.py', 610),
                         ref(W / 'postrun/retire_closed_nonbest_authorized.py', 38),
                         ref(W / 'postrun/retire_closed_nonbest_authorized.py', 59)],
        },
        'F_evaluation_type': {
            'status': 'WARN',
            'classification': 'real_gt',
            'details': 'Primary Box/native Mask evaluation uses actual dataset GT under the disclosed native protocol. Model-predicted mask extents are inference geometry; candidate oracle coverage is a diagnostic. Semantic review remains same-family/provisional; deterministic saved-Box verification is accepted within its data scope.',
            'evidence': [ref(D, 1086), ref(W / 'mask_reference.py', 18),
                         ref(W / 'paired_support_loop.py', 317),
                         ref(W / 'paired_support_loop.py', 331)],
        },
    }
    report = {
        'audit_skill': 'experiment-audit',
        'verdict': 'WARN', 'overall_verdict': 'warn', 'integrity_status': 'warn',
        'reason_code': 'qualified_real_gt_results_with_restoration_and_scope_limits',
        'summary': 'Actual closed single-seed pair supports 5599/4859 Box hits for both arms; all archived candidate Box labels reconcile. Content is a tied metric-best eligible terminal. No geometry accuracy benefit, full updated joint goal, three effective contributions or cold GPU restoration is established.',
        'blocking_findings': [],
        'execution_scope': 'ACTUAL_CLOSED_TRAINED_PAIR', 'fresh_context': True,
        'review_independence': 'same-family', 'acceptance_status': 'provisional',
        'agent_id': AGENT, 'agent_id_kind': 'host_provided_canonical_task_name',
        'verdict_id': 'pvg_compressed_support_closed_actual_audit_20261008',
        'verdict_id_kind': 'auditor_local_label',
        'requested_model': 'gpt-6-astra', 'requested_reasoning_effort': 'max',
        'actual_backend': 'UNATTESTED', 'actual_model': 'UNATTESTED',
        'actual_reasoning_effort': 'UNATTESTED',
        'reviewer_model': 'UNATTESTED', 'reviewer_reasoning': 'UNATTESTED',
        'reviewer_family': 'openai', 'executor_model': 'UNATTESTED',
        'executor_family': 'openai', 'auditor': 'fresh native Codex audit agent',
        'trace_path': TRACE, 'generated_at': now, 'date': '2026-10-08',
        'trace_response_preservation': 'Executor assigned to save the actual returned reviewer response; this file does not attest that future save.',
        'checks': checks,
        'evaluation_type': 'real_gt',
        'claims': [
            {'id': 'C1_closed_hits', 'impact': 'supported_with_scope',
             'details': 'Both terminals have 5599/4859 hits out of 9508, one seed 2027, 141 scans/physical scenes.'},
            {'id': 'C2_selection', 'impact': 'needs_qualifier',
             'details': 'Content is a tied metric-best eligible terminal under the historical rule; it is not uniquely superior.'},
            {'id': 'C3_same_forward_parent', 'impact': 'supported_with_comparator',
             'details': 'Terminal correction gains +1/+11 selected hits over the same-forward frozen parent in this pass.'},
            {'id': 'C4_historical_training_gain', 'impact': 'needs_qualifier',
             'details': 'Historical/initial-to-terminal +1/+3 is an independent-pass observation with demonstrated parent drift.'},
            {'id': 'C5_signed_log_geometry_accuracy_gain', 'impact': 'unsupported'},
            {'id': 'C6_three_effective_contributions', 'impact': 'unsupported'},
            {'id': 'C7_updated_joint_goal_completed', 'impact': 'unsupported'},
            {'id': 'C8_global_gt_free_protocol', 'impact': 'unsupported'},
            {'id': 'C9_independent_raw_mask_recomputation', 'impact': 'unsupported'},
            {'id': 'C10_historical_formal_cold_gpu_restore', 'impact': 'unsupported'},
        ],
        'results': {
            'denominator': 9508, 'seed': 2027, 'scan_count': 141, 'physical_scene_count': 141,
            'threshold_comparison': 'strict_greater_than',
            'historical_or_initial_hits': {'0.25': 5598, '0.5': 4856},
            'content_terminal_hits': {'0.25': 5599, '0.5': 4859},
            'box_conditioned_terminal_hits': {'0.25': 5599, '0.5': 4859},
            'same_forward_parent_hits': {'0.25': 5598, '0.5': 4848},
            'geometry_paired_repairs': {'0.25': 0, '0.5': 0},
            'geometry_paired_damages': {'0.25': 0, '0.5': 0},
            'all_candidate_box_ious_recomputed_float64': 14604288,
            'selected_threshold_disagreements': 0, 'oracle_label_disagreements': 0,
            'native_selected_iou_max': 1.0000050067901611,
            'native_selected_vs_cpu64_max_abs_difference': 6.6200876713828904e-6,
            'raw_mask_recomputed': False,
        },
        'archive': cpu['archive'], 'training': cpu['training'], 'selection': cpu['selection'],
        'current_targets_separate_from_historical_selection': True,
        'unmet_current_wide_threshold_hits': 59,
        'restoration_scope': source['preflight'],
        'pending_operations': [
            'Corrected cold CPU terminal inspection, including actual checkpoint hash/restore checks.',
            'Verified archive of the selected new content checkpoint and all required factories/dependencies.',
            'Only the authorized exact remote nonbest/superseded checkpoints and archived NPZ cleanup, after prerequisites pass.',
            'Publish actual results and completed cleanup receipts with historical/current goals kept separate.',
        ],
        'pending_operations_are_integrity_blockers': False,
        'action_items': [
            'Carry all claim and restoration qualifiers into downstream reports.',
            'Preserve six protected weights and the superseded warm-start local archive.',
            'Do not promote SOURCE_ONLY or CPU RNG replay checks into historical GPU-forward evidence.',
            'Do not retroactively alter pair_spec, historical selection rules or closed metrics.',
        ],
        'scope_limits': [
            'No original raw dataset, raw masks, detector/superpoint files or original imported Scan helper reconstruction.',
            'No terminal checkpoint deserialization, neural forward or historical formal GPU cold reconstruction by this auditor.',
            'Native annotated object/language-map protocol is disclosed; global GT-free evaluation is not claimed.',
            'Single seed and one 9508-expression/141-scene ScanRefer formal scope; no new Nr3D/Sr3D results.',
            'Same-family semantic audit is provisional, with requested model/effort and actual identity kept distinct.',
        ],
        'execution': {
            'independent_cpu_checks_exit_code': 0, 'source_checks_exit_code': 0,
            'cpu_checks_elapsed_seconds': cpu['elapsed_seconds'],
            'python': cpu['python'], 'numpy': cpu['numpy'],
            'neural_forwards': 0, 'torch_imports': 0, 'ssh_calls': 0,
            'checkpoint_deserializations': 0, 'training_updates': 0,
            'source_mutations': 0, 'experiment_artifact_mutations': 0,
            'cleanup_performed': False, 'promotion_performed': False,
            'evidence': audit_evidence,
        },
        'reviewed_files': sorted(reviewed.values(), key=lambda row: row['path'].lower()),
        'audited_input_hashes': {str(path): 'sha256:' + reviewed[str(path)]['sha256'] for path in required},
        'reviewed_file_semantics': 'Exact content hashes for every machine/semantic read. Scope labels distinguish semantic review, complete data checks, archive-hash-only verification, and auditor-generated evidence; byte hashing alone is not a scientific-content review.',
    }
    (A / 'EXPERIMENT_AUDIT.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    trace = {key: report[key] for key in ('agent_id', 'agent_id_kind', 'verdict_id', 'verdict_id_kind',
        'trace_path', 'generated_at', 'execution_scope', 'fresh_context', 'review_independence',
        'acceptance_status', 'requested_model', 'requested_reasoning_effort', 'actual_backend',
        'actual_model', 'actual_reasoning_effort', 'trace_response_preservation')}
    trace['reports'] = [dict(path=str(A / name), sha256=digest(A / name)[0])
                        for name in ('EXPERIMENT_AUDIT.md', 'EXPERIMENT_AUDIT.json')]
    (A / 'AUDIT_trace_metadata.json').write_text(json.dumps(trace, indent=2) + '\n', encoding='utf-8')
    reread = json.loads((A / 'EXPERIMENT_AUDIT.json').read_bytes())
    assert reread['verdict'] == 'WARN' and not reread['blocking_findings']
    assert len(reread['checks']) == 6 and reread['acceptance_status'] == 'provisional'
    for row in reread['reviewed_files']:
        assert digest(Path(row['path'])) == (row['sha256'], row['bytes'])
    print(json.dumps({'status': 'PASS_REPORT_ASSEMBLY_AND_REVALIDATED_ALL_REVIEWED_HASHES',
                      'verdict': reread['verdict'], 'blocking_findings': [],
                      'reviewed_files': len(reread['reviewed_files']),
                      'report_sha256': digest(A / 'EXPERIMENT_AUDIT.json')[0],
                      'report_path': str(A / 'EXPERIMENT_AUDIT.json')}))


if __name__ == '__main__':
    main()
