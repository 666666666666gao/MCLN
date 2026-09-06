"""Launch the disposable native GPU probe only after actual ScanRefer promotion."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys

import paramiko

repo=Path('C:/Users/gb/.codex_mcln_g0_20260905')
sys.path.insert(0,str(repo))
from scripts.evaluate_scanrefer_range_official import promotion_check, row_metrics
prep=repo/'refine-logs/native_range_preparation_20260907_v2'
prepared=json.loads((prep/'receipt.json').read_bytes())
expected=json.loads((prep/'expected.json').read_bytes())
assert prepared['status']=='pass' and prepared['reader_variant']=='extent'
assert prepared['schema']=='mcln-candidate-range-native-preparation-v1'
assert prepared['gpu_forwards']==prepared['native_model_optimizer_updates']==prepared['checkpoint_writes']==0
assert prepared['superpoint_files_verified']=={'train':1201,'val':312}
assert hashlib.sha256((prep/'native_source_manifest.json').read_bytes()).hexdigest()==prepared['source_manifest_sha256']
entry=(repo/'scripts/run_native_candidate_range_preflight.py').read_bytes()
assert hashlib.sha256(entry).hexdigest()==expected['prepared_files']['run_native_candidate_range_preflight.py']
files={name:(prep/name).read_bytes() for name in ['annotation_receipt.json','preflight_rows.json','nr_contract.json','data_inputs.json']}
for name,raw in files.items():
    assert hashlib.sha256(raw).hexdigest()==expected['prepared_files'][name]
files['run_native_candidate_range_preflight.py']=entry
files['cpu_preparation_receipt.json']=(prep/'receipt.json').read_bytes()
formal_root='/root/autodl-tmp/mcln_scanrefer_range_official_20260907_v1'
client=paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
sftp=client.open_sftp()
with sftp.open(formal_root+'/controller.exit','rb') as stream:
    assert stream.read().strip()==b'0'
with sftp.open(formal_root+'/result/receipt.json','rb') as stream:
    formal_raw=stream.read()
formal=json.loads(formal_raw)
assert formal['schema']=='mcln-scanrefer-range-official-v1'
assert formal['data_root']==prepared['data_root']
assert formal['status']=='complete' and formal['formal_rows']==9508
assert formal['optimizer_steps']==formal['checkpoint_writes']==0
assert formal['all_model_states_unchanged'] and formal['native_evaluators_match_row_metrics']
with sftp.open(formal_root+'/input_manifest.json','rb') as stream:
    raw=stream.read()
assert hashlib.sha256(raw).hexdigest()==formal['manifest_sha256']
with sftp.open(formal_root+'/result/rows.json','rb') as stream:
    stream.prefetch(file_size=stream.stat().st_size)
    row_raw=stream.read()
assert hashlib.sha256(row_raw).hexdigest()==formal['rows_sha256']
rows=json.loads(row_raw)
recomputed={}
for arm in ['protected_v99','center_v99','local_v99']:
    recomputed[arm]=row_metrics(rows[arm])
    assert recomputed[arm]['rows']==9508
    assert len({row['row_id'] for row in rows[arm]})==9508
    for key,value in recomputed[arm].items():
        if key=='mask_miou':
            assert abs(value-formal['metrics'][arm][key])<1e-8
        else:
            assert value==formal['metrics'][arm][key],(arm,key)
for before,after in zip(rows['protected_v99'],rows['local_v99']):
    assert all(before[key]==after[key] for key in ['row_id','scan_id','physical_space','point_sha256'])
promotion=promotion_check(recomputed['protected_v99'],recomputed['local_v99'])
assert promotion==formal['promotion'] and promotion['advance_to_nr3d_sr3d_rec']
with sftp.open(formal_root+'/result/independent_audit.json','rb') as stream:
    formal_audit_raw=stream.read()
formal_audit=json.loads(formal_audit_raw)
assert formal_audit['schema']=='mcln-scanrefer-range-official-audit-v1'
assert formal_audit['integrity_pass']
assert formal_audit['receipt_sha256']==hashlib.sha256(formal_raw).hexdigest()
assert formal_audit['promotion']==promotion
_,out,err=client.exec_command('nvidia-smi --query-compute-apps=pid --format=csv,noheader',timeout=30)
assert not out.read().strip() and out.channel.recv_exit_status()==0,err.read().decode()
local=repo/'refine-logs/native_range_preflight_20260907_v1'
remote='/root/autodl-tmp/mcln_native_range_preflight_20260907_v1'
local.mkdir()
sftp.mkdir(remote)
manifest={'schema':'mcln-native-range-gpu-preflight-input-v1',
          'model_source':'/root/autodl-tmp/mcln_native_range_preparation_20260907_v2/model_source',
          'source_manifest_sha256':prepared['source_manifest_sha256'],
          'annotation_source_manifest_sha256':expected['parent_manifest_sha256'],
          'candidate_local_visual_variant':'extent',
          'checkpoint':'/root/autodl-tmp/DATA_ROOT/output/network_v99_baseline_gt/nr3d/control/official_rec_monitor/official_best_rec025_epoch_57_0p56652741.pth',
          'checkpoint_sha256':'76aa6cd49ca20a34e78509465f1185b1b9040e60807ad327d6b0876aeb6edba1',
          'scan_formal_receipt':formal_root+'/result/receipt.json',
          'scan_formal_receipt_sha256':hashlib.sha256(formal_raw).hexdigest(),
          'scan_formal_audit_sha256':hashlib.sha256(formal_audit_raw).hexdigest(),
          'scan_formal_audit':formal_root+'/result/independent_audit.json',
          'files':{name:hashlib.sha256(raw).hexdigest() for name,raw in files.items()},
          'data_root':prepared['data_root'], 'rows_per_dataset':16,'optimizer_steps_per_dataset':2,
          'core_learning_rate':1e-6,'local_learning_rate':1e-4,'checkpoint_writes':0,
          'pretraining_scope':'Nr protected weights for both shape-compatible protocols;Sr historical best not restored.',
          'scope':'Disposable actual native train-mode/input/gradient check;not a cross-dataset training endpoint.'}
files['input_manifest.json']=(json.dumps(manifest,indent=2,sort_keys=True)+'\n').encode()
for name,raw in files.items():
    (local/name).write_bytes(raw)
    with sftp.open(remote+'/'+name,'wx') as stream:
        stream.write(raw)
(local/'scan_formal_receipt.json').write_bytes(formal_raw)
(local/'scan_formal_rows.json').write_bytes(row_raw)
controller='''#!/usr/bin/env bash
set -u
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
cd '''+remote+'''
flock -x /root/autodl-tmp/mcln_v99_backbone_gpu0.lock /root/miniconda3/envs/bdetr/bin/python -u run_native_candidate_range_preflight.py --manifest '''+remote+'''/input_manifest.json
status=$?
printf '%s\\n' "$status" > controller.exit
exit "$status"
'''
(local/'controller.sh').write_text(controller,encoding='utf-8')
with sftp.open(remote+'/controller.sh','wx') as stream:
    stream.write(controller.encode())
command='screen -dmS mcln_native_range_preflight_v1 bash -lc '+shlex.quote('cd '+shlex.quote(remote)+' && bash controller.sh > run.log 2>&1')
_,out,err=client.exec_command(command,timeout=30)
out.read()
assert out.channel.recv_exit_status()==0,err.read().decode()
_,out,err=client.exec_command('screen -ls',timeout=30)
sessions=out.read().decode()
assert out.channel.recv_exit_status()==0 and 'mcln_native_range_preflight_v1' in sessions
launch={'time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        'screen_session':[line.strip() for line in sessions.splitlines() if 'mcln_native_range_preflight_v1' in line],
        'manifest_sha256':hashlib.sha256(files['input_manifest.json']).hexdigest(),
        'controller_sha256':hashlib.sha256(controller.encode()).hexdigest(),
        'scan_formal_promotion_recomputed_from_rows':promotion,
        'initial_observation_delay_seconds':240,
        'launch_is_not_completed_preflight':True}
raw=(json.dumps(launch,indent=2,sort_keys=True)+'\n').encode()
(local/'launch.json').write_bytes(raw)
with sftp.open(remote+'/launch.json','wx') as stream:
    stream.write(raw)
sftp.close()
client.close()
print(json.dumps(launch))
