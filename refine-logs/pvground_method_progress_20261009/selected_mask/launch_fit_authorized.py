"""Launch the unchanged full pair only after actual M0 closure and fit review."""
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shlex

import paramiko


local = Path(__file__).resolve().parent
assert not (local / 'fit_launch.json').exists()
review = json.loads((local / 'LAUNCH_SOURCE_REVIEW.json').read_bytes())
actual = json.loads((local / 'PREFLIGHT_ACTUAL_REVIEW.json').read_bytes())
assert actual['execution_scope'] == 'ACTUAL_PREFLIGHT'
assert actual['verdict'] in ('PASS', 'WARN') and not actual['blocking_findings']
assert actual['fresh_context'] is True
required_actual_paths = {path.resolve() for path in (local / 'preflight_wait.json',
    local / 'preflight_complete/preflight.json', local / 'pair_spec.json')}
assert required_actual_paths.issubset({Path(item['path']).resolve() for item in actual['reviewed_files']})
for item in actual['reviewed_files']:
    assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() == item['sha256']

assert review['execution_scope'] == 'SOURCE_ONLY'
assert review['verdict'] in ('PASS', 'WARN') and not review['blocking_findings']
for item in review['reviewed_files']:
    assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() == item['sha256']
wait = json.loads((local / 'preflight_wait.json').read_bytes())
assert wait['observer_closed'] and not wait['controller_alive'] and wait['exitcode'] == 0
assert wait['status']['status'] == 'complete' and wait['status']['protected_parents_exact']
proof = json.loads((local / 'preflight_complete/preflight.json').read_bytes())
assert proof['status'] == 'pass' and proof['optimizer_steps_per_arm'] == 2
assert proof['weight_files_created'] == 0 and proof['separate_optimizers_and_gradients']
assert proof['parent_and_box_head_frozen'] and proof['trainable_parameters_per_arm'] == 27841
for arm in ('content', 'selected_query'):
    assert proof['restore'][arm]['full_cpu_state_exact']
    assert proof['restore'][arm]['full_state_tensors'] == 1314
    assert proof['restore'][arm]['actual_native_gpu_integration']
    assert proof['restore'][arm]['same_cache_masks_and_geometry_exact']
    assert proof['restore'][arm]['gpu_head_and_optimizer_restore']['all_keys_moments_steps_and_groups_exact']
    assert proof['witnesses'][0]['arms'][arm]['zero_update_warm_mask_exact']
    assert proof['witnesses'][0]['arms'][arm]['warm_arm_mask_and_box_exact']
spec = json.loads((local / 'pair_spec.json').read_bytes())
root = spec['root']
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root',
               password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
with sftp.open(spec['runtime'] + '/env_spec.json', 'rb') as stream:
    env = json.loads(stream.read())
assert hashlib.sha256(json.dumps(env, sort_keys=True, separators=(',', ':')).encode()).hexdigest() == spec['env_spec_sha256']
arrays_per_full_pair = math.ceil(9508 / 8)
# Exactly three B8 all256 6D Box arrays, all256 scores, GT and row IDs.
batch_array_bytes = 8 * (3 * 256 * 6 + 256 + 6) * 4 + 8 * 8
reserve = arrays_per_full_pair * (batch_array_bytes + 16384) + 128 * 1024**2 + 2 * 1024**2
probe = '''import hashlib,json,shutil,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]);required=int(sys.argv[2]);spec=json.loads((root/'pair_spec.json').read_bytes())
assert (root/'preflight_controller.exit').read_text().strip()=='0'
assert json.loads((root/'preflight_status.json').read_bytes())['status']=='complete'
assert not (root/'fit_status.json').exists()
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).decode().strip()
capacity=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.total','--format=csv,noheader,nounits']).decode().strip().splitlines()
assert len(capacity)==1
index,name,used,total=[value.strip() for value in capacity[0].split(',')]
assert int(index)==0 and 'A100' in name and 40000<=int(total)<=45000 and int(used)<500
assert shutil.disk_usage(root).free>required
for key in ('base_terminal','selected_terminal','warm_support_terminal'):
    assert hashlib.sha256(Path(spec[key]).read_bytes()).hexdigest()==spec[key+'_sha256']
print(json.dumps(dict(GPU_idle=True,preflight_closed=True,required_reserve_bytes=required,
    data_free_bytes=shutil.disk_usage(root).free,system_free_bytes=shutil.disk_usage('/').free,
    files_deleted=0,new_optimizer_or_inference_steps=0)))
'''
python = spec['runtime'] + '/venv/bin/python'
_, stdout, stderr = client.exec_command(shlex.join([python, '-B', '-c', probe, root, str(reserve)]), timeout=60)
raw = stdout.read()
assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
resources = json.loads(raw)
(local / 'fit_capacity.json').write_bytes(raw)
command = shlex.join(['flock','-n',env['resource_limits']['gpu_lock'],python,'-B','-u',
                      root + '/controller.py','--phase','fit'])
shell = command + ' > ' + shlex.quote(root + '/fit_controller.log') + ' 2>&1; code=$?; printf "%s\\n" "$code" > ' + shlex.quote(root + '/fit_controller.exit') + '; exit "$code"'
screen = 'pvg_selected_mask_fit_20261009'
_, stdout, stderr = client.exec_command(shlex.join(['screen','-dmS',screen,'bash','-c',shell]), timeout=30)
assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
pattern = '^' + python + ' -B -u ' + root + '/controller.py --phase fit$'
_, stdout, stderr = client.exec_command('pgrep -af ' + shlex.quote(pattern), timeout=30)
process = stdout.read().decode().strip()
assert stdout.channel.recv_exit_status() == 0 and len(process.splitlines()) == 1, stderr.read().decode()
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(),root=root,
    controller_pid=int(process.split()[0]),process=process,screen=screen,resources=resources,
    status='FIT_LAUNCHED_NOT_COMPLETED',accuracy_result=False,
    seed=2027,batch_size=8,updates_per_arm=3723,fit_rows_per_arm=29778,
    fresh_formal_initial_states_and_optimizers=True,
    support_prior_updates=7446,support_total_updates_at_terminal=11169,
    estimated_seconds=17038,first_check_seconds=16738,later_poll_seconds=240,
    estimate_basis='Actual prior head-pair fit receipt18990.032s minus initialholdout2927.655 and terminalholdout2893.475 plus formal9508 3868.926 =17037.828s. Fit starts directly; first check5min before estimated finish, then240s. No intermediate go/no-go query; never infer completion or restart from elapsed time.')
(local / 'fit_launch.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
with sftp.open(root + '/fit_launch.json', 'wx') as stream:
    stream.write((local / 'fit_launch.json').read_bytes())
sftp.close()
client.close()
print(json.dumps(record), flush=True)
