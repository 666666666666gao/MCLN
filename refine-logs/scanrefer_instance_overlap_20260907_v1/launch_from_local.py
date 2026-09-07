import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import paramiko

repo=Path('C:/Users/gb/.codex_mcln_g0_20260905')
archive=repo/'refine-logs/scanrefer_instance_overlap_20260907_v1'
remote='/root/autodl-tmp/mcln_scanrefer_instance_overlap_20260907_v1'
source='/root/autodl-tmp/mcln_scanrefer_local_visual_preflight_20260906_v2/model_source'
stage='/root/autodl-tmp/mcln_scanrefer_stage_diagnostic_20260907_v1/diagnostic_result'
local_stage=repo/'refine-logs/scanrefer_stage_diagnostic_20260907_v1/diagnostic_result'
sha=lambda raw:hashlib.sha256(raw).hexdigest()
files={name:(repo/name).read_bytes() for name in ['scripts/analyze_scanrefer_instance_overlap.py','scripts/audit_scanrefer_stage_diagnostic.py','tests/test_instance_overlap_diagnostic.py']}
files['scripts/__init__.py']=b''
manifest={'schema':'mcln-instance-overlap-input-v1','stage_rows':stage+'/stage_rows.json',
 'scan_source':source,'scene_pickle':'/root/autodl-tmp/DATA_ROOT/val_v3scans.pkl',
 'fixed_categories':['root_unique_max','other_same_label_unique_max','other_different_label_unique_max','tied_max','no_overlap'],
 'tie_atol':1e-6,'counts_are_geometric_proxies_not_semantic_identity':True,
 'input_files':{stage+'/'+name:sha((local_stage/name).read_bytes()) for name in ['stage_rows.json','receipt.json','independent_audit.json']}}
manifest['input_files'].update({
 manifest['scene_pickle']:'ceaf79c713b6d270cc8c988169c4bbd3e885e7bf7c98ed82b5b0a08711ffaad9',
 source+'/src/visual_data_handlers.py':sha((repo/'src/visual_data_handlers.py').read_bytes()),
 '/root/autodl-tmp/mcln_scanrefer_native_box_transfer_pair_20260907_v1/input_manifest.json':'2e7b7fa65afd056c0ad98a9a92e974e563df6e104624abdb30cfaf4716e1e72a',
 '/root/autodl-tmp/mcln_scanrefer_native_box_transfer_posttraining_20260907_v1/input_manifest.json':'c1760a8a6107ee1b8dbca5ba2fde536dfe60b501ec3e38f1358daa29c609e942'})
manifest['input_files'].update({remote+'/'+name:sha(raw) for name,raw in files.items()})
files['input_manifest.json']=(json.dumps(manifest,indent=2,sort_keys=True)+'\n').encode()
files['launch_from_local.py']=Path(__file__).read_bytes()
controller='''#!/usr/bin/env bash
set -euo pipefail
cd REPLACE_ROOT
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
trap 'printf "%s\\n" "$?" > controller.exit' EXIT
/root/miniconda3/envs/bdetr/bin/python -m pytest -q tests/test_instance_overlap_diagnostic.py > cpu_tests.txt 2>&1
/root/miniconda3/envs/bdetr/bin/python -u -m scripts.analyze_scanrefer_instance_overlap --directory .
'''.replace('REPLACE_ROOT',shlex.quote(remote))
files['controller.sh']=controller.encode()
c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
check="import json,os,socket; m={line.split(':')[0]:int(line.split()[1])*1024 for line in open('/proc/meminfo') if line.startswith('MemAvailable:')}; assert m['MemAvailable']>12*1024**3; print(json.dumps(dict(m,host=socket.gethostname(),uid=os.getuid())))"
_,o,e=c.exec_command('/root/miniconda3/envs/bdetr/bin/python -c '+shlex.quote(check),timeout=30)
raw=o.read();err=e.read();assert o.channel.recv_exit_status()==0,err.decode()
resources=json.loads(raw)
s=c.open_sftp();archive.mkdir();s.mkdir(remote);s.mkdir(remote+'/scripts');s.mkdir(remote+'/tests')
for name,raw in files.items():
 p=archive/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
 with s.open(remote+'/'+name,'wx') as f:f.write(raw)
 with s.open(remote+'/'+name,'rb') as f:assert f.read()==raw
inner='exec bash '+shlex.quote(remote+'/controller.sh')+' > '+shlex.quote(remote+'/run.log')+' 2>&1'
_,o,e=c.exec_command('screen -dmS mcln_instance_overlap_v1 bash -c '+shlex.quote(inner),timeout=30)
assert o.channel.recv_exit_status()==0,e.read().decode()
_,o,e=c.exec_command('screen -ls',timeout=30)
screens=o.read().decode();assert o.channel.recv_exit_status()==0,e.read().decode()
matches=[line.strip() for line in screens.splitlines() if '.mcln_instance_overlap_v1' in line]
assert len(matches)==1,matches
result={'time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
 'screen':matches[0],'resources_before':resources,'manifest_sha256':sha(files['input_manifest.json']),
 'gpu_forwards':0,'actual_completion_verified':False,'expected_duration_seconds':120}
raw=(json.dumps(result,indent=2)+'\n').encode();(archive/'launch.json').write_bytes(raw)
with s.open(remote+'/launch.json','wx') as f:f.write(raw)
s.close();c.close();print(json.dumps(result),flush=True)
