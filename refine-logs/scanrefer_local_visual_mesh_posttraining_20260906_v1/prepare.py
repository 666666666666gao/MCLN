import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex

import paramiko

repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
local = repo / 'refine-logs/scanrefer_local_visual_mesh_posttraining_20260906_v1'
remote = '/root/autodl-tmp/mcln_scanrefer_local_visual_mesh_posttraining_20260906_v1'
training = '/root/autodl-tmp/mcln_scanrefer_local_visual_mesh_pair_20260906_v1'
formal = '/root/autodl-tmp/mcln_scanrefer_local_visual_mesh_official_20260906_v1'
old_formal = repo / 'refine-logs/scanrefer_local_visual_official_20260906_v3'
old_manifest = json.loads((old_formal / 'input_manifest.json').read_bytes())
files = {name: (repo / name).read_bytes() for name in old_manifest['files']}
assert all(hashlib.sha256(raw).hexdigest() == old_manifest['files'][name] for name, raw in files.items())
audit_prep = repo / 'refine-logs/scanrefer_local_visual_mesh_audit_preparation_20260906_v1'
audit_tests = json.loads((audit_prep / 'receipt.json').read_bytes())
for name in ['scripts/audit_scanrefer_local_visual_pair.py']:
    files[name] = (repo / name).read_bytes()
    assert hashlib.sha256(files[name]).hexdigest() == audit_tests['files'][name]
files['scripts/__init__.py'] = b''
code = r"""import datetime,hashlib,json,os,shlex,subprocess,sys
from pathlib import Path
root=Path(__file__).parent
expected=json.loads((root/'preparation.json').read_text())
def sha(path):
 d=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):d.update(block)
 return d.hexdigest()
for name,digest in expected['files'].items():assert sha(root/name)==digest,name
training=Path(expected['training_directory'])
assert (training/'controller.exit').read_text().strip()=='0'
assert sha(training/'input_manifest.json')==expected['training_manifest_sha256']
train=json.loads((training/'input_manifest.json').read_text())
assert train['data_root']=='/root/autodl-tmp/DATA_ROOT_mcln_meshsp/'
assert not (training/'independent_audit.json').exists()
environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
command=[sys.executable,'-m','scripts.audit_scanrefer_local_visual_pair',str(training),str(training/'independent_audit.json')]
with (training/'independent_audit.log').open('xb') as log:
 subprocess.check_call(command,cwd=str(root),env=environment,stdout=log,stderr=subprocess.STDOUT)
audit=json.loads((training/'independent_audit.json').read_text())
receipt=json.loads((training/'receipt.json').read_text())
assert audit['integrity_pass'] and receipt['status']=='complete'
assert audit['receipt_sha256']==sha(training/'receipt.json')
assert receipt['steps_per_arm']==2482 and receipt['holdout_rows']==6887
for arm in ['control','local']:
 assert audit['checkpoints'][arm]['optimizer_steps']==2482
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).strip()
formal=Path(expected['formal_directory'])
formal.mkdir()
(formal/'scripts').mkdir()
for name in expected['formal_files']:
 (formal/name).write_bytes((root/name).read_bytes())
(formal/'data_inputs.json').write_bytes((training/'data_inputs.json').read_bytes())
manifest={'schema':'mcln-scanrefer-local-visual-official-input-v2',
 'training_directory':str(training),'training_receipt_sha256':sha(training/'receipt.json'),
 'training_audit_sha256':sha(training/'independent_audit.json'),'trained_checkpoint':receipt['checkpoints']['local'],
 'data_root':train['data_root'],'val_superpoint_files':train['superpoint_files']['val'],
 'files':{name:sha(formal/name) for name in expected['formal_files'] if name!='scripts/__init__.py'},
 'arms':['protected_v99','local_v99'],'formal_rows':9508,'optimizer_steps':0,
 'scan_rec_historical_floor_hits':[5572,4797],'scan_mask_paper_floor_percent':[58.70,50.70,44.72],
 'nr3d_sr3d_mask_gate':False,
 'decision':'One fixed endpoint after the registered mesh-data repeat;development scores do not select epochs or thresholds.'}
(formal/'input_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
controller='''#!/usr/bin/env bash
set -u
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
cd FORMAL_ROOT
flock -x /root/autodl-tmp/mcln_v99_backbone_gpu0.lock PYTHON_EXEC -u scripts/evaluate_scanrefer_local_visual_official.py --manifest FORMAL_MANIFEST
status=$?
printf '%s\\n' "$status" > controller.exit
exit "$status"
'''.replace('FORMAL_ROOT',shlex.quote(str(formal))).replace('PYTHON_EXEC',shlex.quote(sys.executable)).replace('FORMAL_MANIFEST',shlex.quote(str(formal/'input_manifest.json')))
(formal/'controller.sh').write_text(controller)
subprocess.check_call(['bash','-n',str(formal/'controller.sh')])
screen='mcln_scanrefer_mesh_official_v1'
launch_command=['screen','-dmS',screen,'bash','-lc','cd '+shlex.quote(str(formal))+' && bash controller.sh > run.log 2>&1']
subprocess.check_call(launch_command)
sessions=subprocess.check_output(['screen','-ls']).decode()
matches=[line.strip() for line in sessions.splitlines() if screen in line]
assert len(matches)==1
launch={'time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
 'screen_session':matches,'training_audit_sha256':sha(training/'independent_audit.json'),
 'manifest_sha256':sha(formal/'input_manifest.json'),'controller_sha256':sha(formal/'controller.sh'),
 'command':launch_command,'formal_rows_planned':9508,'optimizer_steps':0,'checkpoint_writes':0,
 'status':'formal evaluation launched;not a completed metric or promotion'}
(formal/'launch.json').write_text(json.dumps(launch,indent=2,sort_keys=True)+'\n')
(root/'executed.json').write_text(json.dumps(launch,indent=2,sort_keys=True)+'\n')
print(json.dumps(launch))
"""
files['post_training.py'] = code.encode()
files['verify_baseline.py'] = Path('C:/Users/gb/.codex/tmp/verify_mcln_mesh_local_visual_baseline_20260906.py').read_bytes()
formal_files = list(old_manifest['files']) + ['scripts/__init__.py']
preparation = {'schema': 'mcln-scanrefer-mesh-posttraining-preparation-v1',
    'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    'training_directory': training, 'formal_directory': formal,
    'training_manifest_sha256': '3b87b7d1dff9641d9bbf6419119396abf5de5616342d741c45e622fd60c12413',
    'formal_files': formal_files, 'files': {name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()},
    'formal_sources_identical_to_completed_v3': True,
    'reused_actual_formal_test_report_sha256': hashlib.sha256((old_formal / 'validation.xml').read_bytes()).hexdigest(),
    'reused_actual_training_audit_test_receipt_sha256': hashlib.sha256((audit_prep / 'receipt.json').read_bytes()).hexdigest(),
    'scope': 'Prepared existing audited entrypoints;requires actual2482-step completion before any audit or formal GPU launch. No new architecture or thresholds.'}
files['preparation.json'] = (json.dumps(preparation, indent=2, sort_keys=True) + '\n').encode()
files['prepare.py'] = Path(__file__).read_bytes()
client = paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
local.mkdir()
sftp.mkdir(remote)
sftp.mkdir(remote + '/scripts')
for name, raw in files.items():
    with sftp.open(remote + '/' + name, 'wx') as stream:
        stream.write(raw)
    target = local / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
check = "import json;from pathlib import Path;r=Path(" + repr(remote) + ");names=[n for n in json.loads((r/'preparation.json').read_text())['files'] if n.endswith('.py')];[compile((r/n).read_bytes(),str(r/n),'exec') for n in names];print('Original Python compiled %d staged entries; no entry executed'%len(names))"
_, out, err = client.exec_command('/root/miniconda3/envs/bdetr/bin/python -c ' + shlex.quote(check), timeout=30)
stdout = out.read().decode()
assert out.channel.recv_exit_status() == 0, err.read().decode()
proof = {'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    'status': 'prepared_original_python_compile_pass', 'stdout': stdout,
    'preparation_sha256': hashlib.sha256(files['preparation.json']).hexdigest(),
    'gpu_forwards': 0, 'optimizer_steps': 0, 'checkpoint_writes': 0,
    'post_training_entry_executed': False, 'baseline_verifier_executed': False}
raw = (json.dumps(proof, indent=2, sort_keys=True) + '\n').encode()
with sftp.open(remote + '/receipt.json', 'wx') as stream:
    stream.write(raw)
(local / 'receipt.json').write_bytes(raw)
sftp.close()
client.close()
print(raw.decode())
