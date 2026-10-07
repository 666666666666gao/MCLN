"""Launch the reviewed paired fit only after actual M0 and disk reserve pass."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex

import paramiko

local = Path(__file__).resolve().parent
assert not (local / 'fit_launch.json').exists()
review = json.loads((local / 'FIT_SOURCE_REVIEW.json').read_bytes())
assert review['execution_scope'] == 'SOURCE_ONLY'
assert review['verdict'] in ('PASS', 'WARN') and not review['blocking_findings']
for entry in review['reviewed_files']:
    assert hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest() == entry['sha256'], entry['path']
wait = json.loads((local / 'preflight_wait.json').read_bytes())
assert wait['observer_closed'] and not wait['controller_alive']
assert wait['exitcode'] == 0 and wait['status']['status'] == 'complete'
proof = json.loads((local / 'preflight_complete/preflight.json').read_bytes())
assert proof['status'] == 'pass' and proof['optimizer_steps_per_arm'] == 2
assert proof['weight_files_created'] == 0 and proof['accuracy_result'] is False
assert proof['same_forward_native_score_mask_reference_exact'] and proof['separate_optimizers_and_gradients']
assert proof['frozen_parent_forward_calls'] == 2
spec = json.loads((local / 'pair_spec.json').read_bytes())
assert proof['spec_sha256'] == hashlib.sha256((local / 'pair_spec.json').read_bytes()).hexdigest()
for arm in ('face_center', 'face_region'):
    restore = proof['cpu_restore'][arm]
    assert restore['full_cpu_model_exact'] and restore['declared_sampler_restored']
    assert restore['optimizer']['all_keys_moments_steps_and_groups_exact']
    assert restore['deployed_geometry_heads'] == 1 and restore['full_state_tensors'] == 1304
    assert proof['witnesses'][0]['arms'][arm]['neutral_decode_equals_reference']
    assert all(value > 0 for value in proof['witnesses'][1]['arms'][arm]['raw_parameter_gradient_norms'].values())
for witness in proof['witnesses']:
    assert witness['center_members_exact'] and witness['independent_real_rectangle_checks'] > 0
    assert witness['actual_all256_raw_member_extent_verified'] and witness['shared_native_score_and_masks_exact']
    assert witness['zero_R_semantic_exact'] and witness['frozen_parent_forwards_per_batch'] == 1
    assert witness['native_final_semantic_head_calls'] == 1
    assert all(witness['arms'][arm]['cross_head_gradients_absent'] for arm in ('face_center', 'face_region'))
root = spec['root']
# Keep the previously proven whole-pair reserve. Shared arrays reduce expected
# output bytes, but that does not justify weakening a successful capacity gate.
reserve = 3 * max(value['serialization_bytes'] for value in proof['cpu_restore'].values()) + 900 * 1024**2
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root',
               password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
names = ('run_face_support_pair.py', 'paired_geometry_loop.py', 'face_region_box_refiner.py',
         'face_support_model_factory.py', 'selected_mask_reference_factory.py',
         'query_supported_geometry.py', 'mask_reference.py', 'pair_spec.json', 'controller.py')
for name in names:
    with sftp.open(root + '/' + name, 'rb') as stream:
        assert stream.read() == (local / name).read_bytes()
with sftp.open(root + '/preflight.json', 'rb') as stream:
    assert stream.read() == (local / 'preflight_complete/preflight.json').read_bytes()
with sftp.open(spec['runtime'] + '/env_spec.json', 'rb') as stream:
    environment = json.loads(stream.read())
python = spec['runtime'] + '/venv/bin/python'
probe = '''import json,shutil,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]);reserve=int(sys.argv[2])
assert json.loads((root/'preflight_status.json').read_bytes())['status']=='complete'
assert (root/'preflight_controller.exit').read_text().strip()=='0'
assert not (root/'fit_status.json').exists()
assert json.loads((root/'preflight.json').read_bytes())['status']=='pass'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).decode().strip()
assert shutil.disk_usage(root).free>=reserve
print(json.dumps(dict(GPU_idle=True,preflight_controller_closed=True,required_reserve_bytes=reserve,
    data_free_bytes=shutil.disk_usage(root).free,system_free_bytes=shutil.disk_usage('/').free)))
'''
_, stdout, stderr = client.exec_command(shlex.join([python, '-B', '-c', probe, root, str(reserve)]), timeout=60)
raw = stdout.read()
assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
resource = json.loads(raw)
(local / 'fit_resource_check.json').write_bytes(raw)
command = shlex.join(['flock', '-n', environment['resource_limits']['gpu_lock'], python,
                      '-B', '-u', root + '/controller.py', '--phase', 'fit'])
inner = command + ' > ' + shlex.quote(root + '/fit_controller.log') + ' 2>&1; code=$?; printf "%s\\n" "$code" > ' + shlex.quote(root + '/fit_controller.exit') + '; exit "$code"'
screen = 'pvg_face_support_fit_20261007'
_, stdout, stderr = client.exec_command(shlex.join(['screen', '-dmS', screen, 'bash', '-c', inner]), timeout=30)
assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
pattern = '^' + python + ' -B -u ' + root + '/controller.py --phase fit$'
_, stdout, stderr = client.exec_command('pgrep -af ' + shlex.quote(pattern), timeout=30)
process = stdout.read().decode().strip()
assert stdout.channel.recv_exit_status() == 0 and len(process.splitlines()) == 1, stderr.read().decode()
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(), root=root,
    process=process, controller_pid=int(process.split()[0]), screen=screen, resources=resource,
    status='PAIRED_FACE_FIT_LAUNCHED_NOT_COMPLETED', accuracy_result=False,
    fit_rows_per_arm=29778, updates_per_arm=3723, fit_passes=1, physical_batch=8,
    effective_batch=8, accumulation=1, tail_batch_rows=2, trainable_parameters_per_arm=456102,
    shared_frozen_parent_forwards_per_batch=1, independent_optimizers=2, deployed_heads_per_model=1,
    retained_hidden_prior_updates=11169, retained_hidden_total_updates_at_terminal=14892,
    reset_output_total_updates_at_terminal=3723, primary_mode='bbs',
    protected_best_hits=[5598, 4848], candidate_gate_hits=[5620, 4764], no_multiseed=True,
    first_check_seconds=12600, later_poll_seconds=240, estimated_seconds=13500,
    estimate_basis='Closed separate-parent pair25450s; shared frozen forward halves parent work but both geometry/losses remain, estimate3h45min; first check3h30min. Actual M0 timing and perstep logs disclosed separately.')
(local / 'fit_launch.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
with sftp.open(root + '/fit_launch.json', 'wx') as stream:
    stream.write((local / 'fit_launch.json').read_bytes())
sftp.close()
client.close()
print(json.dumps(record), flush=True)
