import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import xml.etree.ElementTree as ET

import paramiko


repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
old_run = repo / 'refine-logs/scanrefer_local_visual_official_20260906_v1'
data_audit = json.loads((old_run / 'data_root_audit/receipt.json').read_bytes())
old_audit_raw = (old_run / 'result/independent_audit.json').read_bytes()
old_audit = json.loads(old_audit_raw)
assert old_audit['integrity_pass'] and not old_audit['promotion']['advance_to_nr3d_sr3d_rec']
assert data_audit['different_count'] == 206 and data_audit['same_count'] == 106
manifest = json.loads((old_run / 'input_manifest.json').read_bytes())
local = repo / 'refine-logs/scanrefer_local_visual_official_20260906_v2'
remote = '/root/autodl-tmp/mcln_scanrefer_local_visual_official_20260906_v2'
source_names = ['scripts/evaluate_scanrefer_local_visual_official.py',
                'scripts/scanrefer_joint_readout.py', 'scripts/scanrefer_rec_evaluation.py',
                'scripts/scanrefer_data_contract.py', 'scripts/audit_scanrefer_local_visual_official.py',
                'scripts/audit_scanrefer_joint_readout_pair.py']
test_names = ['tests/test_scanrefer_data_contract.py', 'tests/test_evaluate_scanrefer_local_visual_official.py',
              'tests/test_audit_scanrefer_local_visual_official.py']
files = {name: (repo / name).read_bytes() for name in source_names + test_names}
files['scripts/__init__.py'] = b''
local.mkdir()
(local / 'scripts').mkdir()
(local / 'tests').mkdir()
client = paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
with sftp.open('/root/autodl-tmp/mcln_scanrefer_local_visual_official_20260906_v1/result/independent_audit.json', 'rb') as stream:
    assert stream.read() == old_audit_raw
sftp.mkdir(remote)
sftp.mkdir(remote + '/scripts')
sftp.mkdir(remote + '/tests')
for name, raw in files.items():
    (local / name).write_bytes(raw)
    with sftp.open(remote + '/' + name, 'wx') as stream:
        stream.write(raw)
program = r'''
import hashlib,json,os
from pathlib import Path
root=Path('/root/autodl-tmp/DATA_ROOT_mcln_meshsp')
old=Path('/root/autodl-tmp/DATA_ROOT')
shared={}
for child in root.iterdir():
    if child.name!='superpoints':
        assert os.path.samefile(str(child),str(old/child.name)),str(child)
        shared[child.name]=str(child.resolve())
superpoints={}
for split,count in [('train',1201),('val',312)]:
    folder=root/'superpoints'/split
    paths=sorted(path for path in folder.iterdir() if path.is_file())
    assert len(paths)==count,(split,len(paths))
    superpoints[split]={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
print(json.dumps({'data_root':str(root),'superpoint_files':superpoints,'non_superpoint_entries_share_original_inode':shared}))
'''
_, output, error = client.exec_command('/root/miniconda3/envs/bdetr/bin/python -c ' + shlex.quote(program), timeout=300)
data_raw = output.read()
error_text = error.read().decode()
assert output.channel.recv_exit_status() == 0, error_text
data_inputs = json.loads(data_raw)
expected = {item['name']: item['mesh_sha256'] for item in data_audit['same_files'] + data_audit['different_files']}
assert data_inputs['superpoint_files']['val'] == expected
(local / 'data_inputs.json').write_bytes(data_raw)
with sftp.open(remote + '/data_inputs.json', 'wx') as stream:
    stream.write(data_raw)
manifest.update(schema='mcln-scanrefer-local-visual-official-input-v2',
                data_root=data_inputs['data_root'], val_superpoint_files=expected,
                files={name: hashlib.sha256(files[name]).hexdigest() for name in source_names},
                correction_of_old_root_run='/root/autodl-tmp/mcln_scanrefer_local_visual_official_20260906_v1',
                old_root_audit_sha256=hashlib.sha256(old_audit_raw).hexdigest(),
                decision='Correct the established superpoint data-root mismatch;reevaluate the same fixed weights once;no retraining,checkpoint selection,or threshold changes.')
raw = (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode()
(local / 'input_manifest.json').write_bytes(raw)
with sftp.open(remote + '/input_manifest.json', 'wx') as stream:
    stream.write(raw)
test_command = ('cd ' + shlex.quote(remote)
                + " && CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1"
                + ' /root/miniconda3/envs/bdetr/bin/python -m pytest ' + ' '.join(test_names)
                + ' -q --junitxml=validation.xml > validation.log 2>&1')
_, output, error = client.exec_command(test_command, timeout=300)
output.read()
error_text = error.read().decode()
status = output.channel.recv_exit_status()
for source_name, target_name in [('validation.log', 'validation.txt'), ('validation.xml', 'validation.xml')]:
    with sftp.open(remote + '/' + source_name, 'rb') as stream:
        (local / target_name).write_bytes(stream.read())
assert status == 0, (local / 'validation.txt').read_text() + error_text
suites = list(ET.parse(str(local / 'validation.xml')).getroot().iter('testsuite'))
counts = {key: sum(int(suite.attrib[key]) for suite in suites) for key in ['tests', 'failures', 'errors', 'skipped']}
assert counts == {'tests': 29, 'failures': 0, 'errors': 0, 'skipped': 0}
_, output, error = client.exec_command('nvidia-smi --query-compute-apps=pid --format=csv,noheader', timeout=30)
assert not output.read().strip() and output.channel.recv_exit_status() == 0, error.read().decode()
controller = '''#!/usr/bin/env bash
set -u
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
cd ''' + remote + '''
flock -x /root/autodl-tmp/mcln_v99_backbone_gpu0.lock /root/miniconda3/envs/bdetr/bin/python -u scripts/evaluate_scanrefer_local_visual_official.py --manifest ''' + remote + '''/input_manifest.json
status=$?
printf '%s\\n' "$status" > controller.exit
exit "$status"
'''
(local / 'controller.sh').write_bytes(controller.encode())
with sftp.open(remote + '/controller.sh', 'wx') as stream:
    stream.write(controller.encode())
command = 'screen -dmS mcln_scanrefer_local_visual_official_v2 bash -lc ' + shlex.quote('cd ' + shlex.quote(remote) + ' && bash controller.sh > run.log 2>&1')
_, output, error = client.exec_command(command, timeout=30)
output.read()
assert output.channel.recv_exit_status() == 0, error.read().decode()
_, output, error = client.exec_command('screen -ls', timeout=30)
sessions = output.read().decode()
assert output.channel.recv_exit_status() == 0 and 'mcln_scanrefer_local_visual_official_v2' in sessions
launch = {'schema': 'mcln-scanrefer-local-visual-official-launch-v2',
          'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
          'screen_session': [line.strip() for line in sessions.splitlines() if 'mcln_scanrefer_local_visual_official_v2' in line],
          'manifest_sha256': hashlib.sha256(raw).hexdigest(), 'controller_sha256': hashlib.sha256(controller.encode()).hexdigest(),
          'tests': counts, 'data_root': data_inputs['data_root'], 'data_inputs_sha256': hashlib.sha256(data_raw).hexdigest(),
          'formal_rows_planned': 9508, 'optimizer_steps': 0, 'checkpoint_writes': 0,
          'same_trained_checkpoint': manifest['trained_checkpoint'], 'launch_is_not_completed_evaluation': True}
launch_raw = (json.dumps(launch, indent=2, sort_keys=True) + '\n').encode()
(local / 'launch.json').write_bytes(launch_raw)
(local / 'launch.py').write_bytes(Path(__file__).read_bytes())
with sftp.open(remote + '/launch.json', 'wx') as stream:
    stream.write(launch_raw)
sftp.close()
client.close()
print(json.dumps(launch))
