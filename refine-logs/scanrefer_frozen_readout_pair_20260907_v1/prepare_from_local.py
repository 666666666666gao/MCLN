import ast
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex

import paramiko

repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
local = repo / 'refine-logs/scanrefer_frozen_readout_pair_20260907_v1'
remote = '/root/autodl-tmp/mcln_scanrefer_frozen_readout_pair_20260907_v1'
queue_local = repo / 'refine-logs/scanrefer_frozen_readout_posttraining_20260907_v1'
queue_remote = '/root/autodl-tmp/mcln_scanrefer_frozen_readout_posttraining_20260907_v1'
formal_remote = '/root/autodl-tmp/mcln_scanrefer_frozen_readout_official_20260907_v1'
probe_local = repo / 'refine-logs/scanrefer_frozen_readout_probe_20260907_v1'
probe = json.loads((probe_local / 'input_manifest.json').read_bytes())
probe_receipt = json.loads((probe_local / 'receipt.json').read_bytes())
assert probe_receipt['status'] == 'pass' and probe_receipt['readout_frozen_and_unchanged']
previous = json.loads((repo / 'refine-logs/scanrefer_range_preflight_20260907_v1/input_manifest.json').read_bytes())
sha = lambda raw: hashlib.sha256(raw).hexdigest()
encode = lambda value: (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
names = ['run_scanrefer_frozen_readout_pair.py', 'audit_scanrefer_frozen_readout_pair.py',
         'evaluate_scanrefer_frozen_readout_official.py', 'audit_scanrefer_frozen_readout_official.py',
         'queue_scanrefer_frozen_readout_posttraining.py', 'scanrefer_joint_readout.py',
         'scanrefer_data_contract.py', 'audit_scanrefer_joint_readout_pair.py', 'scanrefer_rec_evaluation.py']
files = {'scripts/' + name: (repo / 'scripts' / name).read_bytes() for name in names}
for name, raw in files.items():
    ast.parse(raw, filename=name)
files['scripts/__init__.py'] = b''
manifest = {key: probe[key] for key in ['model_source', 'source_manifest_sha256', 'artifacts',
            'split_protocol', 'split_protocol_sha256', 'split_salt', 'data_root', 'train_superpoint_files']}
manifest.update(schema='mcln-scanrefer-frozen-readout-training-input-v1',
    native_probe_receipt='/root/autodl-tmp/mcln_scanrefer_frozen_readout_probe_20260907_v1/receipt.json',
    native_probe_receipt_sha256=sha((probe_local / 'receipt.json').read_bytes()),
    files={name: sha(raw) for name, raw in files.items()},
    val_superpoint_files=previous['superpoint_files']['val'],
    epochs=1, batch_size=12, steps_per_arm=2482, fit_rows=29778, holdout_rows=6887,
    core_learning_rate=1e-6, weight_decay=.0005, clip_norm=.1, readout_loss_weight=1. / 3.,
    readouts_frozen=True, new_network_modules=0, checkpoint_writes_planned=2,
    core_trainable_tensors=probe_receipt['candidate_trainable_tensors'],
    arms=['native_only', 'frozen_gt'], candidate_predeclared='frozen_gt',
    sole_difference='Whether GT supervision through the frozen old readout backpropagates into the same68 core tensors; control detaches readout inputs.',
    frozen_core_uses_eval_mode=True, full_native_gt_loss_retained=True,
    training_data_augmentation=False, formal_rows_during_training=0,
    formal_evaluation_policy='Only if both REC thresholds are nondegrading versus baseline and native_only on the fixed6887-row module holdout, run one fixed9508-row endpoint; never substitute the control.',
    protected_historical_scanrefer_rec_hits=[5572,4797], scanrefer_mask_floor_percent=[58.70,50.70,44.72],
    nr3d_sr3d_mask_gate=False, backbone_has_seen_module_holdout_scenes=True,
    retained_inference_dependencies=['Parent','Geometry','V99','fixed Pareto and geometry construction'],
    not_postprocessing_removal=True, no_validation_hyperparameter_search=True,
    inherited_budget='Same one-pass2482-step protocol and coreLR as previous fixedScanRefer pair; no epoch or weight sweep.',
    input_binding='Same correctmesh root, full1201train and312val file identities; no retiredrangecheckpoint or module.',
    environment_reuse=probe['environment_reuse'])
files['input_manifest.json'] = encode(manifest)

def controller(directory, command, gpu):
    return ('''#!/usr/bin/env bash
set -u
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=''' + ('0' if gpu else "''") + '\ncd ' + directory + '\n' + command + '''
status=$?
printf '%s\\n' "$status" > controller.exit
exit "$status"
''').encode()

files['controller.sh'] = controller(remote, 'flock -x /root/autodl-tmp/mcln_v99_backbone_gpu0.lock /root/miniconda3/envs/bdetr/bin/python -u scripts/run_scanrefer_frozen_readout_pair.py --manifest '+remote+'/input_manifest.json', True)
files['prepare_from_local.py'] = Path(__file__).read_bytes()
queue_files = {'queue.py': files['scripts/queue_scanrefer_frozen_readout_posttraining.py']}
queue_manifest = {'schema': 'mcln-frozen-readout-posttraining-queue-v1', 'training_directory': remote,
    'formal_directory': formal_remote, 'training_manifest_sha256': sha(files['input_manifest.json']),
    'queue_script_sha256': sha(queue_files['queue.py']), 'interval_seconds': 240,
    'candidate_predeclared': 'frozen_gt', 'formal_only_after_module_rec_screen': True,
    'no_nr3d_sr3d_training_before_scanrefer_promotion': True}
queue_files['input_manifest.json'] = encode(queue_manifest)
queue_files['controller.sh'] = controller(queue_remote, '/root/miniconda3/envs/bdetr/bin/python -u queue.py --manifest '+queue_remote+'/input_manifest.json', False)
client = paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
with sftp.open(manifest['native_probe_receipt'], 'rb') as stream:
    assert sha(stream.read()) == manifest['native_probe_receipt_sha256']
machine_code = '''import json,os,socket,subprocess,shutil
p=subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.used,memory.total','--format=csv,noheader,nounits'],stdout=subprocess.PIPE)
assert p.returncode==0
result={'uid':os.getuid(),'hostname':socket.gethostname(),'cwd':os.getcwd(),'disk_free':shutil.disk_usage('/root/autodl-tmp').free,'gpu':p.stdout.decode().strip()}
assert result['disk_free']>4*1024**3 and int(result['gpu'].split(',')[2])<500
print(json.dumps(result))
'''
_, output, error = client.exec_command('/root/miniconda3/envs/bdetr/bin/python -c '+shlex.quote(machine_code), timeout=30)
machine = json.loads(output.read())
assert output.channel.recv_exit_status() == 0, error.read().decode()
assert machine['uid'] == 0 and machine['hostname'] == 'autodl-container-c7cb4299a4-24929f53'
for local_dir, remote_dir, mapping in [(local,remote,files),(queue_local,queue_remote,queue_files)]:
    local_dir.mkdir()
    sftp.mkdir(remote_dir)
    if local_dir == local:
        sftp.mkdir(remote_dir+'/scripts')
    for name, raw in mapping.items():
        path = local_dir/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        with sftp.open(remote_dir+'/'+name, 'wx') as stream:
            stream.write(raw)
        with sftp.open(remote_dir+'/'+name, 'rb') as stream:
            assert stream.read() == raw

check_code = '''import ast,hashlib,json,subprocess,sys
from pathlib import Path
from scripts.evaluate_scanrefer_frozen_readout_official import promotion_check
from scripts.audit_scanrefer_frozen_readout_official import native_metrics,rec_compare
from scripts.audit_scanrefer_frozen_readout_pair import audit_rows,audit_checkpoints
m=json.loads(Path('input_manifest.json').read_text())
for name,digest in m['files'].items():
 raw=Path(name).read_bytes()
 assert hashlib.sha256(raw).hexdigest()==digest
 ast.parse(raw,filename=name)
base={'rows':9508,'rec_hits025':5572,'rec_hits050':4797,'mask_hits025':5689,'mask_hits050':4974,'mask_miou':45.9226}
assert promotion_check(base,base)['advance_to_nr3d_sr3d_rec']
for field,value in [('rec_hits025',5571),('rec_hits050',4796),('mask_hits025',5581),('mask_hits050',4820),('mask_miou',44.71)]:
 bad=dict(base);bad[field]=value
 assert not promotion_check(base,bad)['advance_to_nr3d_sr3d_rec'],field
better=dict(base,rec_hits025=5573)
assert not promotion_check(better,base)['advance_to_nr3d_sr3d_rec']
historical=Path('/root/autodl-tmp/mcln_scanrefer_range_official_20260907_v1/result/native_rows.json')
rows=json.loads(historical.read_text())
assert native_metrics(rows['protected_v99'])=={'rows':9508,'rec_hits025':5515,'rec_hits050':4411}
effect=rec_compare(rows['protected_v99'],rows['local_v99'])
assert effect['effects']['025']['net']==-13 and effect['effects']['050']['net']==8
for module in ['run_scanrefer_frozen_readout_pair','audit_scanrefer_frozen_readout_pair','evaluate_scanrefer_frozen_readout_official','audit_scanrefer_frozen_readout_official','queue_scanrefer_frozen_readout_posttraining']:
 p=subprocess.run([sys.executable,'-m','scripts.'+module,'--help'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
 assert p.returncode==0,p.stdout.decode()
print(json.dumps({'python':sys.version.split()[0],'original_environment_cli_and_imports':True,'all_source_ast':True,'promotion_nonregression_and_mask_checks':True,'historical_real_row_recomputation':True,'new_gpu_forwards':0,'no_trial_result_claim':True}))
'''
command = 'cd '+shlex.quote(remote)+' && CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='+shlex.quote(remote)+' /root/miniconda3/envs/bdetr/bin/python -c '+shlex.quote(check_code)
_, output, error = client.exec_command(command, timeout=60)
checks = json.loads(output.read())
assert output.channel.recv_exit_status() == 0, error.read().decode()
preparation = {'time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    'manifest_sha256':sha(files['input_manifest.json']), 'queue_manifest_sha256':sha(queue_files['input_manifest.json']),
    'machine':machine,'checks':checks,'uploaded_bytes_verified':True,'training_started':False,'formal_training_updates':0}
raw=encode(preparation)
(local/'preparation.json').write_bytes(raw)
with sftp.open(remote+'/preparation.json','wx') as stream:
    stream.write(raw)
with sftp.open(remote+'/preflight_checks.py','wx') as stream:
    stream.write(check_code.encode())
(local/'preflight_checks.py').write_text(check_code,encoding='utf-8',newline='\n')
sftp.close()
client.close()
print(json.dumps(preparation),flush=True)
