"""Reuse authorized observer/collection/launch lifecycle; no remote calls here."""
import ast
import hashlib
import json
from pathlib import Path

local = Path(__file__).resolve().parent
prior = local.parent/'pvground_range_head_only_20261004'
assert not (local/'lifecycle_preparation.json').exists()
text = (prior/'observe_preflight_authorized.py').read_text(encoding='utf-8')
text = text.replace("local/'preflight_spec.json'", "local/'distribution_preflight_spec.json'")
text = text.replace("root=Path(sys.argv[1]);pid=int(sys.argv[2]);directory=root/'whole_range'",
    "root=Path(sys.argv[1]);pid=int(sys.argv[2])\n"
    "status=json.loads((root/'status.json').read_bytes());directory=root/status['phase']")
(local/'observe_preflight_authorized.py').write_text(text,encoding='utf-8',newline='\n')
text = (prior/'wait_preflight_authorized.py').read_text(encoding='utf-8')
(local/'wait_preflight_authorized.py').write_text(text,encoding='utf-8',newline='\n')
text = (prior/'collect_preflight_terminal.py').read_text(encoding='utf-8')
text = text.replace("local/'preflight_spec.json'", "local/'distribution_preflight_spec.json'")
text = text.replace('pvground_range_head_only_preflight_20261004', 'pvground_boundary_preflight_20261004')
text = text.replace('pvg_range_head_only_fit_20261004', 'pvg_boundary_fit_20261004')
(local/'collect_preflight_terminal.py').write_text(text,encoding='utf-8',newline='\n')

text = (prior/'launch_formal_authorized.py').read_text(encoding='utf-8')
text = text.replace("local/'complete_preflight/whole_range/preflight.json'", "local/'complete_preflight/distribution/preflight.json'")
text = text.replace("local/'whole_range_spec.json'", "local/'distribution_spec.json'")
text = text.replace('pvground_range_head_only_fit_20261004', 'pvground_boundary_fit_20261004')
text = text.replace('pvground_range_head_only_preflight_20261004', 'pvground_boundary_preflight_20261004')
text = text.replace('same_cached_inputs_zero_head_pair_exact', 'same_cached_inputs_zero_head_common_floor_exact')
text = text.replace("preflight/'whole_range/preflight.json'", "preflight/'distribution/preflight.json'")
text = text.replace("('run_range_head_only.py','controller.py','EXPERIMENT_PLAN.md','EXPERIMENT_CODE_REVIEW.json')", "('run_boundary_fit.py','controller.py','EXPERIMENT_PLAN.md','EXPERIMENT_CODE_REVIEW.json')")
text = text.replace("modules=json.loads((local/'source_preparation.json').read_bytes())['unchanged_model_modules']",
    "modules=" + repr(['whole_model_preflight_checks.py','pvground_whole_mask_box_refiner.py','whole_mask_range.py',
        'pvground_candidate_box_refiner.py','pvground_tail_support_box_refiner.py','pvground_tail_preflight.py',
        'pvground_semantic_assignment.py','pvground_source_query.py','pvground_observation_query.py',
        'pvground_task_observation_query.py','initial_range_comparison.py','pvground_boundary_box_refiner.py']))
text = text.replace("('local_range','whole_range')", "('residual','distribution')")
text = text.replace("root+'/run_range_head_only.py'", "root+'/run_boundary_fit.py'")
text = text.replace("order=['local_range','whole_range']", "order=['residual','distribution']")
text = text.replace('estimate_hours_pair=[4,7]', 'estimate_hours_pair=[4.5,5.5]')
text = text.replace("estimate_basis='prior joint pair about7.9hours, frozen mode actual full-fit throughput not yet measured'",
    "estimate_basis='completed frozen-G pair4h24m46s; distribution overhead unmeasured'")
text = text.replace("teacher=False,quality_loss=False,contrastive_expansion=False", "teacher=False,quality_loss=False,boundary_distribution_and_DFL_in_second_arm=True,contrastive_expansion=False")
# Both arms need their actual probe, not merely the distribution probe.
before = "assert proof['same_cached_inputs_zero_head_common_floor_exact'] and proof['optimizer_steps']==2\n"
assert text.count(before) == 1
text = text.replace(before, before + "for arm in ('residual','distribution'):\n"
    "    arm_proof=json.loads((local/'complete_preflight'/arm/'preflight.json').read_bytes())\n"
    "    assert arm_proof['status']=='pass' and arm_proof['boundary_mode']==arm\n"
    "    assert arm_proof['original_g_state_unchanged'] and arm_proof['optimizer_steps']==2\n"
    "    assert arm_proof['same_cached_inputs_zero_head_common_floor_exact']\n")
(local/'launch_formal_authorized.py').write_text(text,encoding='utf-8',newline='\n')
names = ('observe_preflight_authorized.py','wait_preflight_authorized.py','collect_preflight_terminal.py','launch_formal_authorized.py')
for name in names:
    ast.parse((local/name).read_text(encoding='utf-8'))
record = dict(status='PREPARED_ONLY_NO_REMOTE_CALLS',files={name:dict(bytes=(local/name).stat().st_size,
    sha256=hashlib.sha256((local/name).read_bytes()).hexdigest()) for name in names},
    model_forwards=0, optimizer_steps=0, remote_jobs=0)
(local/'lifecycle_preparation.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record))
