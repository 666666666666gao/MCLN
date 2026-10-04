"""Serialize the fresh review; does not modify any original artifact."""
from pathlib import Path
import datetime
import hashlib
import json

root=Path('C:/Users/gb/.codex/tmp/pvground_final_quality_20261005')
out=root/'analysis'
cpu=json.loads((out/'AUDIT_CPU_CHECK.json').read_bytes())
identities=json.loads((out/'AUDITED_INPUT_SHA256.json').read_bytes())
aliases={
    'ROOT':root,
    'DATA':root.parent/'pvground_g_p2_20261002/complete/source',
    'NATIVE':root.parent/'pvground_runtime_bundle_20260908_v1/PV-Ground',
    'CONTROL':root.parent/'pvground_geometry_readback_20261004/formal_draft',
    'REV':root.parent/'pvground_geometry_readback_20261004/revision2',
    'PARENT':root.parent/'pvground_boundary_distribution_20261004/complete/distribution/formal',
    'LEGACY':root.parent/'pvground_fused_support_20261002',
}


def evidence(alias,path,line):
    return str(aliases[alias]/path).replace('\\','/')+':'+str(line)


checks={
    'A':{
        'name':'ground_truth_provenance','status':'WARN',
        'details':[
            'The actual imported dataset loader hash matches the collected DATA loader, not the older NATIVE loader. Root Box/Mask come from ScanRefer object IDs and scene instance annotations.',
            'The quality term uses training GT only. Native formal selection has no added GT geometry gate or validation optimizer path. The quality pool is explicitly geometric eligibility, not semantic identity.',
            'All 3723 training records have one matched GT per expression. Nonroot exclusion is source-correct, but a nonempty multi-GT case is not demonstrated.',
            'Raw annotation/scene/point/Mask replay was unavailable. The new run pins the referenced split/source identities but not the input-manifest file itself.'
        ],
        'subchecks':{'active_dataset_loader_identity':'PASS','dataset_root_lineage':'PASS','saved_row_identity':'PASS',
                     'no_added_inference_gt_gate':'PASS','no_validation_optimizer_path':'PASS','training_geometry_proxy_labeled':'PASS',
                     'raw_dataset_replay':'UNAVAILABLE','multi_gt_case_exercised':'WARN','manifest_file_identity_pinning':'WARN'},
        'evidence':[evidence('ROOT','complete/quality/imports.json',8),evidence('DATA','joint_det_dataset.py',585),
                    evidence('DATA','joint_det_dataset.py',1086),evidence('DATA','joint_det_dataset.py',1387),
                    evidence('ROOT','runtime_bundle/run_final_quality_fit.py',48),evidence('ROOT','runtime_bundle/run_final_quality_fit.py',146),
                    evidence('ROOT','runtime_bundle/run_final_quality_fit.py',241),evidence('ROOT','runtime_bundle/run_final_quality_fit.py',273),
                    evidence('ROOT','runtime_bundle/native_final_quality.py',12),evidence('ROOT','analysis/AUDIT_CPU_CHECK.json',37)]
    },
    'B':{
        'name':'score_loss_and_metric_denominators','status':'PASS',
        'details':[
            'Signed native bbs exactly follows token softmax plus main/modify/pronoun/relation evidence minus other entity evidence at source level. Preflight and native evaluator count witnesses are consistent.',
            'Primary accuracy uses strict IoU > .25/.50 divided by evaluated rows. There is no prediction-statistic normalization of reported accuracy.',
            'Quality target is detached final Box/root IoU in original-G unmatched-qualified plus actual Hungarian-root pool. The centered error is a per-sample loss with fixed weight 1.0.',
            'Native G replacement, CE and contrastive reductions are unchanged. Real preflight receipts verify independent pairwise/gradient equivalence and an isolated quality gradient into R. Training loss is not reported as accuracy.'
        ],
        'evidence':[evidence('ROOT','runtime_bundle/native_root_bbs.py',4),evidence('NATIVE','src/grounding_evaluator.py',218),
                    evidence('NATIVE','src/grounding_evaluator.py',296),evidence('NATIVE','src/grounding_evaluator.py',535),
                    evidence('NATIVE','models/losses.py',817),evidence('NATIVE','models/losses.py',852),
                    evidence('NATIVE','models/losses.py',943),evidence('ROOT','runtime_bundle/native_final_quality.py',19),
                    evidence('ROOT','runtime_bundle/native_final_quality.py',33),evidence('ROOT','runtime_bundle/pvground_semantic_assignment.py',34),
                    evidence('ROOT','complete/preflight/preflight.json',24),evidence('ROOT','runtime_bundle/run_final_quality_fit.py',460)]
    },
    'C':{
        'name':'existence_traversal_reconstruction_restore_cleanup','status':'WARN',
        'details':[
            'All scoped intake file SHA256/size checks, executed script/spec/import identities, log/receipt equality and zero-exit controller closure checks pass.',
            '29778 distinct fit rows, 3722 batch8 updates plus one batch2 tail, 3723 total updates, and every reused-control batch order reproduce exactly.',
            'CPU selected/bypass/coarse Box-IoU reconstruction yields zero threshold disagreements; all reported primary counts, repairs/damages, GT-volume quartiles and parent quadrants reproduce.',
            'Mask counts and means are reaggregated from stored IoUs; all256 availability is reaggregated from stored flags. Raw mask and candidate tensors are absent for independent reconstruction.',
            'Strict model/AdamW restore, frozen parent states, same-frame immutability and cleanup are supported by executed code and receipts; no new tensor replay was performed.',
            'Post-cleanup receipts verify the three mandatory reconstruction parents and absence of the new nonbest delta. Historical V99 is not targeted by deletion, but its fresh hash/existence is outside the provided terminal witness.'
        ],
        'subchecks':{'artifact_identity':'PASS','controller_closed':'PASS','complete_training_order':'PASS','independent_box_reconstruction':'PASS',
                     'stored_mask_reaggregation':'PASS','raw_mask_reconstruction':'UNAVAILABLE','stored_availability_reaggregation':'PASS',
                     'full_candidate_reconstruction':'UNAVAILABLE','restore_and_freeze_receipts':'PASS','three_parent_preservation_receipts':'PASS',
                     'historical_v99_fresh_identity':'UNAVAILABLE'},
        'evidence':[evidence('ROOT','complete/INTAKE.json',1),evidence('ROOT','preflight_complete/INTAKE.json',1),
                    evidence('ROOT','complete/quality/train.jsonl',1),evidence('ROOT','complete/quality/train.jsonl',3723),
                    evidence('ROOT','runtime_bundle/readback_preflight_checks.py',19),evidence('ROOT','runtime_bundle/run_final_quality_fit.py',216),
                    evidence('ROOT','runtime_bundle/run_final_quality_fit.py',395),evidence('ROOT','complete/quality/formal_restore.json',1),
                    evidence('ROOT','complete/quality/weight_retention.json',1),evidence('ROOT','controller.py',92),
                    evidence('ROOT','CLOSED_RESOURCES.json',2),evidence('ROOT','analysis/AUDIT_CPU_CHECK.json',893)]
    },
    'D':{
        'name':'executed_paths_and_dead_code','status':'WARN',
        'details':[
            'Historical draft headers and SOURCE_ONLY review are not runtime evidence; current pinned receipt/log/exit artifacts establish actual preflight/train/formal execution.',
            'The model deploys one native head after Mask, geometry refinement and readback. Evaluation adds an explicitly logged cached-head bypass diagnostic; this is not a retrained ablation.',
            'repeated_forward_differences is imported but uncalled; verify_native_replacement, whole_range_loss_routes and distribution_loss do not provide new-run runtime evidence.',
            'Initial independent full forwards differ in one selected Query and by up to 0.0017104148864746094 in same-Query Box coordinates; no threshold outcomes differ. Formal bypass/parent Boxes differ by up to 3.457e-6 or 6.914e-6. These are measured differences, not proof of sole numerical cause.'
        ],
        'evidence':[evidence('ROOT','EXPERIMENT_PLAN.md',3),evidence('ROOT','SOURCE_REVIEW.md',3),
                    evidence('REV','source_preview/PV-Ground/models/pv_ground.py',507),evidence('REV','source_preview/PV-Ground/models/modules.py',135),
                    evidence('ROOT','runtime_bundle/run_final_quality_fit.py',107),evidence('ROOT','runtime_bundle/run_final_quality_fit.py',289),
                    evidence('ROOT','runtime_bundle/pvground_semantic_assignment.py',53),evidence('ROOT','runtime_bundle/whole_model_preflight_checks.py',5),
                    evidence('ROOT','runtime_bundle/pvground_boundary_box_refiner.py',40),evidence('ROOT','analysis/AUDIT_CPU_CHECK.json',943)]
    },
    'E':{
        'name':'scope_and_claims','status':'WARN',
        'details':[
            'One new seed2027 fit, one reused completed control, one-pass budget; no independent seed variance or retrained repetition.',
            '6887 holdout expressions in 106 scenes are pretrained-seen. 9508 expressions in 141 scenes are developer validation, not a blind test.',
            'The negative .50 result is supported: quality 4460 vs control 4477 and parent 4506. Target 5615/4754 was not reached.',
            'No broad efficacy/failure mechanism, causal generalization, semantic-identity, novelty, Nr3D or Sr3D conclusion follows.',
            'Legacy tail_fused 891 unmatched/117 root-matched count is verified on 1008 Mask-good/Box-bad validation rows with root-only GT slots. It is not current-parent training responsibility or new efficacy evidence. The supplied body is a source fragment without an execution SHA in its receipt.'
        ],
        'evidence':[evidence('ROOT','EXPERIMENT_PLAN.md',30),evidence('DATA','split_protocol.json',1),
                    evidence('ROOT','analysis/RESULTS.md',20),evidence('ROOT','complete/quality/formal/receipt.json',1),
                    evidence('LEGACY','full_candidate_audit_body.py',24),evidence('LEGACY','candidate_audit/full_gt_scope_v2/receipt.json',5),
                    evidence('ROOT','LEGACY_MASK_QUALIFIED_MATCH_ROLE.json',4),evidence('ROOT','analysis/AUDIT_CPU_CHECK.json',1034)]
    },
    'F':{
        'name':'evaluation_classification','status':'PASS',
        'details':['Actual primary Box/Mask performance is real_gt. Training qualification, oracle availability, face direction and diagnostic Hungarian-role interpretation are explicitly geometric proxies based on real GT, not synthetic-reference accuracy.'],
        'evidence':[evidence('ROOT','runtime_bundle/run_final_quality_fit.py',302),evidence('ROOT','runtime_bundle/native_final_quality.py',12),
                    evidence('ROOT','PARENT_MASK_BOUNDARY_RELATION.json',1),evidence('LEGACY','candidate_audit/full_gt_scope_v2/receipt.json',5)]
    }
}

classifications=[
    dict(evaluation='quality and control initial/terminal holdout',evaluation_type='real_gt',scope='6887 pretrained-seen module-development expressions'),
    dict(evaluation='quality, control and protected parent native formal Box/Mask metrics',evaluation_type='real_gt',scope='9508 developer-validation expressions; bbs primary'),
    dict(evaluation='cached-head bypass and same-Query coarse Box',evaluation_type='real_gt',role='forward diagnostic',independent_trained_ablation=False),
    dict(evaluation='two-update preflight',evaluation_type='real_gt',role='supervised runtime witness',accuracy_result=False),
    dict(evaluation='quality target and original-G candidate qualification',evaluation_type='explicit_geometry_proxy',underlying_reference='real_gt',accuracy_result=False),
    dict(evaluation='top-k/all256 oracle availability',evaluation_type='explicit_geometry_proxy',underlying_reference='real_gt',deployment_metric=False),
    dict(evaluation='parent Mask/Box quadrants and face direction',evaluation_type='explicit_geometry_proxy',underlying_reference='real_gt',causal_evidence=False),
    dict(evaluation='legacy tail_fused match-role recount',evaluation_type='explicit_geometry_proxy',underlying_reference='real_gt',training_responsibility=False,current_parent=False),
]

record=dict(audit_skill='experiment-audit',date='2026-10-05',generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            verdict='WARN',overall_verdict='warn',integrity_status='warn',reason_code='qualified_negative_result_with_evidence_scope_limits',
            summary='Saved artifacts support a bounded negative result: 4460/9508 at IoU > .50, versus reused control 4477 and protected parent 4506. Deterministic CPU checks pass; no concrete blocker or metric falsification found.',
            agent_id='/root/pvg_final_quality_terminal_integrity',verdict_id='/root/pvg_final_quality_terminal_integrity/2026-10-05-terminal',
            executor_family='openai',reviewer_family='openai',requested_backend='codex',requested_model='gpt-6-astra',requested_reasoning_effort='max',
            actual_backend=None,actual_model=None,actual_reasoning_effort=None,backend_attested=False,
            review_independence='same-family',acceptance_status='provisional',trace_path=out.as_posix(),
            execution_scope='read-only source/receipt audit and CPU reconstruction from saved rows; no network, GPU, model, optimizer or weight replay',
            blocking_findings=[],checks=checks,evaluation_classifications=classifications,
            deterministic_checks=dict(status='PASS',count=cpu['checks_passed'],input_files=len(identities['files']),report=(out/'AUDIT_CPU_CHECK.json').as_posix()),
            formal_metrics={**{a:s['formal']['metrics'] for a,s in cpu['stages'].items()},'protected_geometry_parent':cpu['parent_stages']['bbs']['metrics']},
            quality_vs_control=cpu['quality_vs_control']['formal'],versus_parent=cpu['versus_parent'],
            formal_gt_volume_groups=cpu['formal_gt_volume_groups'],cross_process_saved_box_differences=cpu['cross_process_saved_box_differences'],
            legacy_mask_qualified_match_roles=cpu['legacy_mask_qualified_match_roles'],
            claims=[
                dict(id='C1',claim='Bounded primary negative result under the declared seed and one-pass budget',impact='supported'),
                dict(id='C2',claim='Single deployed native head; frozen parents; exact same-frame geometry and Mask preservation; strict restore',impact='supported_at_executed_code_and_receipt_scope'),
                dict(id='C3',claim='All independent complete-forward tensors are identical',impact='unsupported'),
                dict(id='C4',claim='Raw Mask IoU or complete candidate availability independently reconstructed',impact='unsupported'),
                dict(id='C5',claim='Three required reconstruction parents preserved and owned negative delta removed',impact='supported_at_receipt_scope'),
                dict(id='C6',claim='Every historical checkpoint including V99 freshly verified intact',impact='unsupported'),
                dict(id='C7',claim='Legacy unmatched status demonstrates current-parent training responsibility',impact='unsupported'),
                dict(id='C8',claim='Improvement, target achieved, general causal efficacy, novelty, Nr3D/Sr3D completion or cross-family acceptance',impact='unsupported'),
            ],
            action_items=[
                'Close this experiment as the bounded negative result; retain the protected parent as the best primary metric system.',
                'Retain the one-seed/reused-control/developer-validation and pretrained-seen-holdout qualifiers.',
                'Describe Mask/oracle verification at stored-scalar/flag scope, and preserve exact same-frame versus nonidentical separate-forward distinctions.',
                'Do not promote root-only legacy validation assignments into training-responsibility or causal claims; do not claim a multi-GT exclusion case was exercised.',
                'Limit fresh parent-preservation verification to the three recorded reconstruction parents; V99 has no terminal rehash witness.',
            ],
            new_run_required_to_report_negative_result=False,audited_input_hashes={p:'sha256:'+v['sha256'] for p,v in identities['files'].items()},
            audited_input_identity_file=(out/'AUDITED_INPUT_SHA256.json').as_posix(),
            reviewer_artifact_sha256={name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in ['EXPERIMENT_AUDIT.md','AUDIT_CPU_CHECK.json','AUDITED_INPUT_SHA256.json','audit_cpu.py']})
(out/'EXPERIMENT_AUDIT.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(verdict=record['verdict'],checks={k:v['status'] for k,v in checks.items()},deterministic_checks=record['deterministic_checks'],blockers=[]),indent=2))
