"""Prepare existing lifecycle tools for the new controlled boundary package."""
import ast
import hashlib
import json
from pathlib import Path

local = Path(__file__).resolve().parent
prior = local.parent / 'pvground_range_head_only_20261004'
assert not (local / 'control_tools_preparation.json').exists()

text = (prior / 'controller.py').read_text(encoding='utf-8')
text = text.replace("root/'whole_range/spec.json'", "root/'distribution/spec.json'")
text = text.replace("('local_range', 'whole_range')", "('residual', 'distribution')")
text = text.replace("(root/'local_range', root/'whole_range')", "(root/'residual', root/'distribution')")
before = "    assert arm_spec['head_only'] and arm_spec['use_whole_range'] == (arm == 'whole_range')\n"
assert text.count(before) == 1
text = text.replace(before, "    assert arm_spec['head_only'] and arm_spec['use_whole_range']\n"
    "    assert proof['boundary_mode'] == arm_spec['boundary_mode'] == arm\n"
    "    assert proof['boundary_loss_weight'] == arm_spec['boundary_loss_weight']\n"
    "    assert proof['head_parameters'] == arm_spec['head_parameters']\n")
text = text.replace('same_cached_inputs_zero_head_pair_exact', 'same_cached_inputs_zero_head_common_floor_exact')
text = text.replace("root/'run_range_head_only.py'", "root/'run_boundary_fit.py'")
ast.parse(text, feature_version=(3, 7))
(local / 'controller.py').write_text(text, encoding='utf-8', newline='\n')
for arm in ('residual', 'distribution'):
    spec = json.loads((local / (arm + '_preflight_spec.json')).read_bytes())
    spec.update(root='/root/autodl-tmp/pvground_boundary_fit_20261004/' + arm,
                preflight_only_at_install=False)
    (local / (arm + '_spec.json')).write_text(json.dumps(spec, indent=2) + '\n', encoding='utf-8')

# Use the same authorized SSH lifecycle, data disk, GPU lock and byte-verified transfers.
text = (prior / 'launch_preflight_authorized.py').read_text(encoding='utf-8')
start = text.index("prior_local=local.parent/'pvground_whole_mask_fit_20261003'")
end = text.index("client=paramiko.SSHClient()", start)
text = text[:start] + (
    "prior_local=local.parent/'pvground_range_head_only_20261004'\n"
    "intake=json.loads((prior_local/'complete/INTAKE.json').read_bytes())\n"
    "audit=json.loads((prior_local/'analysis/EXPERIMENT_AUDIT.json').read_bytes())\n"
    "assert intake['status']['status']=='complete' and intake['controller_exit']==0 and not intake['controller_alive']\n"
    "assert audit['verdict'] in ('PASS','WARN') and not audit['blocking_issues']\n"
    "spec=json.loads((local/'distribution_preflight_spec.json').read_bytes())\n"
    "root='/root/autodl-tmp/pvground_boundary_preflight_20261004'\n"
    "assert spec['root']==root+'/distribution' and spec['head_only'] and spec['use_whole_range']\n"
    "prior=json.loads((prior_local/'launch.json').read_bytes())\n"
    "controller_pid=prior['process'].split()[0]\n"
) + text[end:]
start = text.index('sftp.mkdir(root);sftp.mkdir(root+\'/whole_range\')')
end = text.index("command=shlex.join(['flock'", start)
text = text[:start] + (
    "sftp.mkdir(root)\n"
    "for name in ('run_boundary_fit.py','preflight_controller.py','EXPERIMENT_PLAN.md','EXPERIMENT_CODE_REVIEW.json'):\n"
    "    content=(local/name).read_bytes()\n"
    "    with sftp.open(root+'/'+name,'wx') as stream:stream.write(content)\n"
    "    with sftp.open(root+'/'+name,'rb') as stream:assert stream.read()==content\n"
    "modules=" + repr(['whole_model_preflight_checks.py','pvground_whole_mask_box_refiner.py','whole_mask_range.py',
        'pvground_candidate_box_refiner.py','pvground_tail_support_box_refiner.py','pvground_tail_preflight.py',
        'pvground_semantic_assignment.py','pvground_source_query.py','pvground_observation_query.py',
        'pvground_task_observation_query.py','initial_range_comparison.py','pvground_boundary_box_refiner.py']) + "\n"
    "for arm in ('residual','distribution'):\n"
    "    directory=root+'/'+arm;sftp.mkdir(directory)\n"
    "    for name in modules:\n"
    "        content=(local/name).read_bytes()\n"
    "        with sftp.open(directory+'/'+name,'wx') as stream:stream.write(content)\n"
    "        with sftp.open(directory+'/'+name,'rb') as stream:assert stream.read()==content\n"
    "    content=(local/(arm+'_preflight_spec.json')).read_bytes()\n"
    "    with sftp.open(directory+'/spec.json','wx') as stream:stream.write(content)\n"
    "    with sftp.open(directory+'/spec.json','rb') as stream:assert stream.read()==content\n"
) + text[end:]
text = text.replace('pvg_range_head_only_preflight_20261004', 'pvg_boundary_preflight_20261004')
text = text.replace("order=['whole_range'],batch_size=8,optimizer_steps=2", "order=['residual','distribution'],batch_size=8,optimizer_steps_per_arm=2")
text = text.replace('estimate_seconds=480,first_check_seconds=360', 'estimate_seconds=960,first_check_seconds=780')
text = text.replace("estimate_basis='prior actual full-factory whole preflight about471seconds; new frozen-mode duration unmeasured'", "estimate_basis='two serial probes; prior actual frozen whole probe449.8seconds; distribution overhead unmeasured'")
# Verify the already existing reviewer source identities before actual uploads.
before = "assert review['execution_scope']=='SOURCE_ONLY'\n"
assert text.count(before) == 1
text = text.replace(before, before + "for item in review['reviewed_files']:\n"
    "    assert __import__('hashlib').sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256']\n")
ast.parse(text)
(local / 'launch_preflight_authorized.py').write_text(text, encoding='utf-8', newline='\n')
record = dict(status='PREPARED_NOT_DEPLOYED', prepared_files={name: hashlib.sha256((local/name).read_bytes()).hexdigest()
    for name in ('controller.py','preflight_controller.py','launch_preflight_authorized.py','residual_spec.json','distribution_spec.json')},
    model_updates=0, remote_jobs=0, weights_created=0)
(local / 'control_tools_preparation.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
print(json.dumps(record))
