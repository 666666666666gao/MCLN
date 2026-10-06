"""Start the reviewed pair only after BOTH actual two-update checks close."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import paramiko

local=Path(__file__).resolve().parent
assert not (local/'fit_launch.json').exists()
review=json.loads((local/'FIT_SOURCE_REVIEW.json').read_bytes())
assert review['verdict'] in ('PASS','WARN') and not review['blocking_findings']
assert review['execution_scope']=='SOURCE_ONLY'
for entry in review['reviewed_files']:
    assert hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest()==entry['sha256'],entry['path']
wait=json.loads((local/'preflight_wait.json').read_bytes())
assert wait['observer_closed'] and not wait['controller_alive']
assert wait['exitcode']==0 and wait['status']['status']=='complete'
proofs={arm:json.loads((local/'preflight_complete'/arm/'preflight.json').read_bytes()) for arm in ('control','keep')}
for arm,proof in proofs.items():
    assert proof['status']=='pass' and proof['optimizer_steps']==2 and proof['weight_files_created']==0
    assert proof['head_parameters']==456102 and proof['head_state_tensors']==10
    assert proof['reference_mode']=='fused_mask'
    assert proof['common_output_reset']==['output.weight','output.bias']
    assert proof['witnesses'][0]['neutral_initial_decode_equals_reference']
    assert proof['invalid_reference_fixture']['actual_empty_support_rows_verified']==39
    assert proof['all_parent_and_R_states_exact'] and proof['qualified_extra_output_gradient_scope_verified']
    assert proof['optimizer_exact_check']['all_keys_moments_steps_and_groups_exact']
    assert proof['extra_geometry_weight']==1.0
    assert proof['reference_keep_weight']==(0.0 if arm=='control' else 1.0)
    assert proof['witnesses'][0]['neutral_initial_reference_keep_exact_zero']
    assert proof['runner_sha256']==hashlib.sha256((local/'run_reference_keep_fit.py').read_bytes()).hexdigest()
    assert proof['spec_sha256']==hashlib.sha256((local/(arm+'_spec.json')).read_bytes()).hexdigest()
    assert proof['native_data_and_member_target_exact']
    for witness in proof['witnesses']:
        assert witness['cached_upstream_after_head_update_native_bbs_exact']
        assert witness['cached_upstream_after_head_update_masks_exact']
        assert witness['extra_direct_output_gradients_only_qualified']
        assert witness['actual_all256_raw_member_extent_verified']
        assert witness['auxiliary_target_mode']=='native_gt'
        assert witness['reference_keep_gradient_only_qualified']
        assert witness['reference_keep_weight']==proof['reference_keep_weight']
assert proofs['keep']['witnesses'][1]['reference_keep_output_gradient_norm']>0
spec=json.loads((local/'control_spec.json').read_bytes())
root=str(Path(spec['root']).parent).replace('\\','/')
reserve=3*max(proof['serialization_bytes'] for proof in proofs.values())+900*1024**2
client=paramiko.SSHClient();client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
sftp=client.open_sftp()
for arm in ('control','keep'):
    with sftp.open(root+'/'+arm+'/preflight.json','rb') as stream:
        assert stream.read()==(local/'preflight_complete'/arm/'preflight.json').read_bytes()
for name in ('run_reference_keep_fit.py','reference_keep.py','selected_mask_reference_factory.py','query_supported_geometry.py','mask_reference.py','check_invalid_reference.py','invalid_reference_fixture.npz','invalid_reference_fixture.json','controller.py','control_spec.json','keep_spec.json'):
    with sftp.open(root+'/'+name,'rb') as stream:
        assert stream.read()==(local/name).read_bytes()
with sftp.open(spec['runtime']+'/env_spec.json','rb') as stream:
    environment=json.loads(stream.read())
python=spec['runtime']+'/venv/bin/python'
with sftp.open(spec['selected_terminal'],'rb') as stream:
    assert hashlib.sha256(stream.read()).hexdigest()==spec['selected_terminal_sha256']
probe='''
import json,shutil,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]);reserve=int(sys.argv[2])
assert json.loads((root/'preflight_status.json').read_bytes())['status']=='complete'
assert (root/'preflight_controller.exit').read_text().strip()=='0'
assert not (root/'fit_status.json').exists()
assert all(json.loads((root/arm/'preflight.json').read_bytes())['status']=='pass' for arm in ('control','keep'))
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
assert shutil.disk_usage(root).free>=reserve
print(json.dumps(dict(GPU_idle=True,preflight_controller_closed=True,required_reserve_bytes=reserve,
    data_free_bytes=shutil.disk_usage(root).free,system_free_bytes=shutil.disk_usage('/').free)))
'''
_,stdout,stderr=client.exec_command(shlex.join([python,'-c',probe,root,str(reserve)]),timeout=60)
raw=stdout.read();assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
resources=json.loads(raw)
(local/'fit_resource_check.json').write_bytes(raw)
command=shlex.join(['flock','-n',environment['resource_limits']['gpu_lock'],python,'-B','-u',root+'/controller.py','--phase','fit'])
inner=command+' > '+shlex.quote(root+'/fit_controller.log')+' 2>&1; code=$?; printf "%s\\n" "$code" > '+shlex.quote(root+'/fit_controller.exit')+'; exit "$code"'
screen='pvg_reference_keep_fit_20261007'
_,stdout,stderr=client.exec_command(shlex.join(['screen','-dmS',screen,'bash','-c',inner]),timeout=30)
assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
pattern='^'+python+' -B -u '+root+'/controller.py --phase fit$'
_,stdout,stderr=client.exec_command('pgrep -af '+shlex.quote(pattern),timeout=30)
process=stdout.read().decode().strip()
assert stdout.channel.recv_exit_status()==0 and len(process.splitlines())==1,stderr.read().decode()
record=dict(time_cst=datetime.datetime.now().astimezone().isoformat(),root=root,process=process,controller_pid=int(process.split()[0]),screen=screen,
    resources=resources,status='TWO_ARM_FIT_LAUNCHED_NOT_COMPLETED',accuracy_result=False,
    fit_rows_per_arm=29778,fit_passes=1,physical_batch=8,effective_batch=8,accumulation=1,
    updates_per_arm=3723,tail_batch_rows=2,retained_hidden_prior_updates=11169,
    retained_hidden_total_updates_at_terminal=14892,reset_output_total_updates_at_terminal=3723,
    common_output_reset=['output.weight','output.bias'],initial_holdout_rows=6887,terminal_holdout_rows=6887,
    initial_formal_rows_per_arm=9508,terminal_formal_rows_per_arm=9508,
    trainable_parameters_by_arm=dict(control=456102,keep=456102),
    trainable_state_tensors_by_arm=dict(control=10,keep=10),parent_and_R_frozen=True,
    primary_mode='bbs',primary_threshold=.5,protected_best_hits=[5598,4848],candidate_gate_hits=[5620,4764],no_multiseed=True,
    first_check_seconds=24300,later_poll_seconds=240,estimated_seconds=25200,
    estimate_basis='Closed previous pair24792s (6h53min), same frozen-parent budget plus train-only keep loss. Estimate7h; first sole observation6h45min after launch, then240s near closure.')
(local/'fit_launch.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
with sftp.open(root+'/fit_launch.json','wx') as stream:
    stream.write((local/'fit_launch.json').read_bytes())
sftp.close();client.close()
print(json.dumps(record),flush=True)
