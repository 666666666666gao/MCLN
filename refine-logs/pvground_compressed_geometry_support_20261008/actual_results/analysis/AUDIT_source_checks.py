"""Supplemental source/provenance checks; reads only, writes auditor evidence."""
import ast
import datetime
import hashlib
import json
from pathlib import Path

W = Path(__file__).resolve().parents[1]
P = W.parent / 'pvground_mask_support_correction_20261008_v2/complete_fit/PV-Ground'
H = W.parent / 'pvground_final_quality_20261005/runtime_bundle'
D = W.parent / 'pvground_g_p2_20261002/complete/source/joint_det_dataset.py'
reviewed = {}


def read(path, scope):
    raw = path.read_bytes()
    reviewed[str(path.resolve())] = dict(path=str(path.resolve()), sha256=hashlib.sha256(raw).hexdigest(),
                                        bytes=len(raw), review_scope=scope)
    return raw


def obj(path, scope='complete_json_content'):
    return json.loads(read(path, scope))


def main():
    sources = ['mask_support_corrector.py','mask_support_model_factory.py','mask_reference.py',
               'selected_mask_reference_factory.py','matched_mask_objective.py','support_pair_forward.py',
               'paired_support_loop.py','run_mask_support_pair.py','controller.py','collect_closed_fit_authorized.py',
               'resume_closed_fit_remaining_authorized.py','postrun/analyze_compressed_geometry_formal.py',
               'postrun/inspect_closed_terminals_rng_replay_authorized.py','postrun/archive_retained_content_authorized.py',
               'postrun/retire_closed_nonbest_authorized.py','postrun/publish_actual_results_cleanup_authorized.py']
    ast_files = []
    for rel in sources:
        path = W / rel
        ast.parse(read(path, 'full_semantic_source_review').decode('utf-8'), filename=str(path))
        ast_files.append(str(path))
    for path in [P/'models/pv_ground.py',P/'models/losses.py',P/'src/grounding_evaluator.py',
                 P/'src/joint_det_dataset.py',P/'main_utils.py',P/'prepare_data.py',
                 P/'src/visual_data_handlers.py',P/'utils/scatter_util.py',D]:
        ast.parse(read(path, 'evaluation_or_active_dataset_path_semantic_review').decode('utf-8'))
        ast_files.append(str(path))
    for rel in ['native_root_bbs.py','pvground_semantic_assignment.py','pvground_boundary_box_refiner.py',
                'pvground_boundary_evidence_readback.py','whole_mask_range.py','readback_preflight_checks.py',
                'whole_model_preflight_checks.py','install_boundary_evidence_readback.py',
                'pvground_whole_mask_box_refiner.py','pvground_tail_support_box_refiner.py','pvground_candidate_box_refiner.py']:
        ast.parse(read(H/rel, 'full_semantic_source_review').decode('utf-8'))
        ast_files.append(str(H/rel))
    read(W/'EXPERIMENT_PLAN.md', 'full_plan_semantic_review')
    goals = obj(W/'CURRENT_RESEARCH_GOALS.json')
    spec = obj(W/'pair_spec.json')
    request = read(W/'postrun/ACTUAL_AUDIT_REQUEST.txt', 'review_request')
    for name in ['experiment-audit/SKILL.md','shared-references/local-codex-policy.md',
                 'shared-references/reviewer-independence.md','shared-references/experiment-integrity.md',
                 'shared-references/review-tracing.md','shared-references/integration-contract.md']:
        read(Path('C:/Users/gb/.codex/skills')/name,'review_policy')
    previous = obj(W/'STORAGE_SOURCE_REVIEW_RNG_REPLAY.json','prior_source_only_review_provenance_not_runtime_evidence')
    assert previous['execution_scope'] == 'SOURCE_ONLY' and previous['source_only_runtime_unproved']
    preflight = obj(W/'complete_fit/preflight.json')
    assert preflight['status'] == 'pass' and preflight['optimizer_steps_per_arm'] == 2
    assert preflight['weight_files_created'] == 0 and preflight['accuracy_result'] is False
    assert preflight['spec_sha256'] == hashlib.sha256((W/'pair_spec.json').read_bytes()).hexdigest()
    assert preflight['separate_optimizers_and_gradients'] and preflight['parent_and_box_head_frozen']
    for arm in ('content','box_conditioned'):
        r = preflight['restore'][arm]
        assert r['actual_native_gpu_integration'] and r['full_cpu_state_exact'] and r['full_state_tensors'] == 1314
        assert r['integration_comparison_scope'] == 'same native forward captured uncorrected inputs'
        assert r['gpu_parent_cold_reconstruction_executed'] is False and r['weight_files_created'] == 0
        assert all(v == 0 for v in r['max_tensor_errors'].values()) and all(v == 0 for v in r['query_mask_max_errors'])
        for witness in preflight['witnesses']:
            a = witness['arms'][arm]
            assert a['original_criterion_active_terms_reconciled']
            assert a['loss'] == a['original_active_loss']
            assert a['coefficients'] == a['native_coefficients'] == [5,1,10,2]
            assert not a['division_by_seven'] and a['supervision_denominator'] == 'actual valid GT count'
    load = obj(W/'complete_fit/load.json')
    assert load['status'] == 'pass' and load['full_state_tensors'] == 1304
    assert load['initial_support_output_zero'] is False and load['fresh_optimizers_required']
    log_checks = {}
    for stage in ('initial_formal','formal'):
        log = read(W/'complete_fit'/f'{stage}.log','executed_log_markers_and_receipt_equality').decode('utf-8')
        terminal = [json.loads(line.split('PAIRED_SUPPORT_EVAL_COMPLETE ',1)[1]) for line in log.splitlines()
                    if line.startswith('PAIRED_SUPPORT_EVAL_COMPLETE ')]
        receipt = obj(W/'complete_fit'/stage/'receipt.json')
        assert terminal == [receipt]
        log_checks[stage] = dict(complete_markers=1, receipt_matches=True)
    log = read(W/'complete_fit/train.log','executed_training_and_holdout_markers').decode('utf-8')
    progress = [json.loads(line.split('PAIRED_SUPPORT_TRAIN_PROGRESS ',1)[1]) for line in log.splitlines()
                if line.startswith('PAIRED_SUPPORT_TRAIN_PROGRESS ')]
    assert progress[0]['step'] == 1 and progress[-1]['step'] == 3723
    log_checks['training'] = dict(progress_markers=len(progress), first_step=1, last_step=3723)
    old = obj(W/'CLEANUP_FOLLOWUP_20261008_1645.json')
    retained = obj(W.parent/'pvground_mask_support_correction_20261008_v2/postrun/RETAINED_BEST.json')
    old_weight = Path(retained['local_archive'])
    raw = read(old_weight,'local_previous_best_archive_bytes_and_hash_only_no_checkpoint_load')
    assert len(raw) == retained['checkpoint_bytes'] == 446789
    assert hashlib.sha256(raw).hexdigest() == spec['warm_support_terminal_sha256']
    protected = [r for r in old['remaining_required_weights'] if r['path'] != spec['warm_support_terminal']]
    assert len(protected) == 5
    expected_paths = {spec['base_terminal'],spec['selected_terminal']}
    assert expected_paths.issubset({r['path'] for r in protected})
    assert any(r['path'].endswith('PV-Ground_ScanRefer.pth') for r in protected)
    assert any(r['path'].endswith('PV-Ground_NR3D.pth') for r in protected)
    assert any(r['path'].endswith('PV-Ground_SR3D.pth') for r in protected)
    p_src=(P/'src/joint_det_dataset.py').read_text(encoding='utf-8')
    d_src=D.read_text(encoding='utf-8')
    comparison={}
    for name in ['_get_target_boxes','_get_scene_objects','_get_token_positive_map_by_parse','_get_detected_objects']:
        a=next(n for n in ast.walk(ast.parse(p_src)) if isinstance(n,ast.FunctionDef) and n.name==name)
        b=next(n for n in ast.walk(ast.parse(d_src)) if isinstance(n,ast.FunctionDef) and n.name==name)
        comparison[name]=dict(parent_source_lines=[a.lineno,a.end_lineno],imported_source_lines=[b.lineno,b.end_lineno],
                              equal_ast=ast.dump(a)==ast.dump(b))
    assert comparison['_get_target_boxes']['equal_ast']
    assert comparison['_get_scene_objects']['equal_ast']
    assert comparison['_get_token_positive_map_by_parse']['equal_ast']
    assert not comparison['_get_detected_objects']['equal_ast']
    publisher=(W/'postrun/publish_actual_results_cleanup_authorized.py').read_text(encoding='utf-8')
    assert '59.5%/51.0%' in publisher and '5658/4850' in publisher and '历史5620/4764' in publisher
    report=dict(status='PASS_SCOPED_SOURCE_AND_EXECUTED_RECEIPT_BINDING',
                generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), ast_parsed_files=len(ast_files),
                logs=log_checks, preflight=dict(status='actual_archived_pass',steps_per_arm=2,weight_files_created=0,
                    full_cpu_reconstruction_tensors=1314,same_cache_native_gpu_integration=True,
                    gpu_parent_cold_reconstruction=False,formal_terminal_cpu_reconstruction=False),
                dataset_sources=comparison,
                dataset_source_interpretation='Actual imported detection-aligned source is hash-bound; the model-tree dataset is different. Target-box/mask, scene-object and token-map method ASTs match. Detection augmentation order in the actual source applies flips before rotations, unlike the unused model-tree dataset.',
                raw_dataset_rebuilt=False, original_imported_visual_data_helper_hash_bound=False,
                previous_source_review_used_as_runtime_evidence=False,
                pending_postrun_artifacts={name:(W/'postrun'/name).exists() for name in
                    ['checkpoint_inspection.json','RETAINED_BEST.json','cleanup_receipt.json','actual_results_publication.json']},
                cleanup=dict(local_previous_best_hash_verified=True, prior_weight_required_by_final_factory=False,
                    protected_dependency_paths=[r['path'] for r in protected], expected_retained_weight_count_after_new_best=6,
                    planned_deletion='only new box_conditioned terminal, superseded warm-start remote copy, and exact 2378 already archived remote NPZ files',
                    cleanup_executed_by_auditor=False, authorization_files_read=False),
                current_target=dict(minimum_hits=[5658,4850], historical_selection_gate=[5620,4764], source_record_separates_them=True),
                no_neural_or_remote_execution=True)
    (W/'analysis/AUDIT_source_checks.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    (W/'analysis/AUDIT_source_reviewed_files.json').write_text(json.dumps(list(reviewed.values()),indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
