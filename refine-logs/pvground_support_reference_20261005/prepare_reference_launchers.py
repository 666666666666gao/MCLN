"""Prepare the reviewed, one-GPU lifecycle without launching anything."""
import ast
from pathlib import Path

root=Path(__file__).resolve().parent
old=root.parent/'pvground_auxiliary_target_20261005'
assert not (root/'deploy_preflight_authorized.py').exists()


def change(text,before,after):
    assert text.count(before)==1,before
    return text.replace(before,after)


deploy=(old/'deploy_preflight_authorized.py').read_text(encoding='utf-8')
begin=deploy.index("labels = local.parent / 'pvground_target_jitter_20261005'")
end=deploy.index("review = json.loads((local / 'SOURCE_REVIEW.json').read_bytes())")
deploy=deploy[:begin]+"previous=local.parent/'pvground_auxiliary_target_20261005'\n"+\
    "audit=json.loads((previous/'analysis/EXPERIMENT_AUDIT.json').read_bytes())\n"+\
    "assert audit['verdict'] in ('PASS','WARN') and not audit['blocking_findings']\n"+\
    "assert json.loads((previous/'terminal_publication.json').read_bytes())['retained_best']['hits']==[5616,4511]\n"+deploy[end:]
deploy=deploy.replace('pvground_auxiliary_target_20261005','pvground_support_reference_20261005')
deploy=change(deploy,"previous=local.parent/'pvground_support_reference_20261005'","previous=local.parent/'pvground_auxiliary_target_20261005'")
deploy=deploy.replace("('control', 'member_target')","('control', 'support_reference')")
deploy=deploy.replace("'member_target_spec.json'","'support_reference_spec.json'")
deploy=change(deploy,"'run_geometry_fit.py', 'query_supported_geometry.py', 'control_spec.json'",
    "'run_geometry_fit.py', 'query_supported_geometry.py', 'support_reference.py', 'control_spec.json'")
deploy=deploy.replace("closed=Path('/root/autodl-tmp/pvground_target_jitter_20261005')",
    "closed=Path('/root/autodl-tmp/pvground_auxiliary_target_20261005')")
deploy=change(deploy,"assert json.loads((closed/'receipt.json').read_bytes())['status']=='complete'\nassert (closed/'run.exit').read_text().strip()=='0'",
    "assert json.loads((closed/'fit_status.json').read_bytes())['status']=='complete'\nassert (closed/'fit_controller.exit').read_text().strip()=='0'\nassert json.loads((closed/'weight_retention.json').read_bytes())['retained_best']['hits']==[5616,4511]")
deploy=deploy.replace('pvg_auxiliary_target_preflight_20261005','pvg_support_reference_preflight_20261005')
deploy=deploy.replace('estimated_seconds=900, first_check_seconds=300','estimated_seconds=1200, first_check_seconds=900')
deploy=deploy.replace('Two sequential protected4509 reconstructions and two updates each; no formal fit or evaluation.',
    'Two sequential protected4511 reconstructions; own-support reference and extra reference gradient/replay check. No formal accuracy.')
(root/'deploy_preflight_authorized.py').write_text(deploy,encoding='utf-8')
observe=(old/'observe_preflight_authorized.py').read_text(encoding='utf-8').replace('range(1,8)','range(1,7)')
(root/'observe_preflight_authorized.py').write_text(observe,encoding='utf-8')
collect=(old/'collect_preflight_authorized.py').read_text(encoding='utf-8')
collect=collect.replace("('control', 'member_target')","('control', 'support_reference')")
collect=collect.replace("'member_target_spec.json'","'support_reference_spec.json'")
collect=change(collect,"'run_geometry_fit.py', 'query_supported_geometry.py', 'controller.py'",
    "'run_geometry_fit.py', 'query_supported_geometry.py', 'support_reference.py', 'controller.py'")
(root/'collect_preflight_authorized.py').write_text(collect,encoding='utf-8')
launch=(old/'launch_geometry_fit_authorized.py').read_text(encoding='utf-8')
begin=launch.index("launch_review=json.loads((local/'LAUNCH_REVIEW.json').read_bytes())")
end=launch.index("wait=json.loads((local/'preflight_wait.json').read_bytes())")
# One fresh SOURCE review covers this launcher's conditions as well.
launch=launch[:begin]+launch[end:]
launch=launch.replace("('control','member_target')","('control','support_reference')")
launch=launch.replace("'member_target_spec.json'","'support_reference_spec.json'")
launch=change(launch,"assert proof['head_parameters']==456102 and proof['head_state_tensors']==10",
    "assert proof['head_parameters']==(459180 if arm=='support_reference' else 456102)\n"
    "    assert proof['head_state_tensors']==(12 if arm=='support_reference' else 10)\n"
    "    assert proof['reference_enabled']==(arm=='support_reference')\n"
    "    if arm=='support_reference':\n"
    "        assert proof['witnesses'][0]['zero_reference_cached_old_head_exact']\n"
    "        assert all(item['reference_direct_geometry_gradient']>0 for item in proof['witnesses'])")
launch=change(launch,"assert witness['auxiliary_target_mode']==('native_gt' if arm=='control' else 'member_gt')",
    "assert witness['auxiliary_target_mode']=='native_gt'")
launch=change(launch,"('run_geometry_fit.py','query_supported_geometry.py','controller.py'",
    "('run_geometry_fit.py','query_supported_geometry.py','support_reference.py','controller.py'")
launch=launch.replace('pvg_auxiliary_target_fit_20261005','pvg_support_reference_fit_20261005')
launch=launch.replace('geometry_parent_fit_updates=7446','geometry_parent_fit_updates=11169')
launch=launch.replace('geometry_head_total_updates_at_terminal=11169','geometry_head_total_updates_at_terminal=14892')
launch=launch.replace('trainable_parameters=456102,trainable_state_tensors=10',
    "trainable_parameters_by_arm=dict(control=456102,support_reference=459180),trainable_state_tensors_by_arm=dict(control=10,support_reference=12)")
launch=launch.replace('protected_best_hits=[5614,4509]','protected_best_hits=[5616,4511]')
launch=launch.replace('estimated_seconds=16400','estimated_seconds=18000')
launch=launch.replace('Actual closed responsibility pair16295.32s, same frozen456102 head, input and evaluation budget. Member-target qualification may change backward throughput.',
    'Actual closed auxiliary-target pair16743.09s. Same data/evaluation; added3078 reference parameters and second support aggregation. Estimate5h, first check near control terminal.')
(root/'launch_geometry_fit_authorized.py').write_text(launch,encoding='utf-8')
for file in root.glob('*.py'):
    ast.parse(file.read_text(encoding='utf-8'))
print('Prepared preflight/deploy/observer/collector/formal launch sources; no remote calls.')
