import datetime
import hashlib
import json
import os
from pathlib import Path

import paramiko

repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
local = repo / 'refine-logs/scanrefer_range_preflight_20260907_v1'
remote = '/root/autodl-tmp/mcln_scanrefer_range_preflight_20260907_v1'
assert not local.exists()
local.mkdir()
old = json.loads((repo / 'refine-logs/scanrefer_local_visual_mesh_pair_20260906_v1/input_manifest.json').read_bytes())
names = ['models/candidate_range_visual.py', 'models/candidate_local_visual.py',
         'scripts/run_scanrefer_range_preflight.py', 'scripts/scanrefer_data_contract.py',
         'scripts/scanrefer_joint_readout.py', 'tests/test_candidate_range_visual.py']
files = {}
for name in names:
    raw = (repo / name).read_bytes()
    target = local / name
    target.parent.mkdir(exist_ok=True)
    target.write_bytes(raw)
    files[name] = hashlib.sha256(raw).hexdigest()
manifest = {key: old[key] for key in ['model_source','source_manifest_sha256','artifacts',
    'split_protocol','split_protocol_sha256','split_salt','data_root','superpoint_files']}
manifest.update({'schema':'mcln-scanrefer-range-preflight-v1','formal_rows':0,'checkpoint_writes':0,
    'files':files,'purpose':'Train-only16 rows;two matched145008-parameter hierarchical readers;64 slots with explicit valid observations;no quality result claimed.',
    'sampling':{'center':'nearest64 in common expanded RoI','extent':'up to8 unique points in each octant nearest +/-0.5 supports'},
    'shared_reader':'region point attention then Query-conditioned region attention;same trainable parameters',
    'window_half_extent_multiplier':1.5,'minimum_half_extent_m':.05,'slots':64,'regions':8,
    'disposable_optimizer_steps_per_arm':2,'preflight_weights_discarded':True,
    'next_training_not_started':True})
raw = (json.dumps(manifest,indent=2,sort_keys=True)+'\n').encode()
(local/'input_manifest.json').write_bytes(raw)
controller = ("#!/usr/bin/env bash\nset -u\nexport CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1\ncd '"+remote+"'\nflock -x /root/autodl-tmp/mcln_v99_backbone_gpu0.lock /root/miniconda3/envs/bdetr/bin/python -u scripts/run_scanrefer_range_preflight.py --manifest '"+remote+"/input_manifest.json'\nstatus=$?\nprintf '%s\\n' \"$status\" > controller.exit\nexit \"$status\"\n").encode()
(local/'controller.sh').write_bytes(controller)
client=paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
sftp=client.open_sftp()
sftp.mkdir(remote)
for subdir in ['models','scripts','tests']:
    sftp.mkdir(remote+'/'+subdir)
for name in names+['input_manifest.json','controller.sh']:
    raw=(local/name).read_bytes()
    with sftp.open(remote+'/'+name,'wx') as stream:
        stream.write(raw)
    with sftp.open(remote+'/'+name,'rb') as stream:
        assert stream.read()==raw
command = "cd '"+remote+"' && CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/bdetr/bin/python -m pytest -q tests/test_candidate_range_visual.py"
_,stdout,stderr=client.exec_command(command,timeout=60)
out=stdout.read()+stderr.read()
status=stdout.channel.recv_exit_status()
(local/'unit_tests.txt').write_bytes(out)
test={'time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
      'exit_code':status,'manifest_sha256':hashlib.sha256((local/'input_manifest.json').read_bytes()).hexdigest(),
      'test_output_sha256':hashlib.sha256(out).hexdigest(),'cuda_visible_devices':'','gpu_forwards':0,'training_started':False}
raw=(json.dumps(test,indent=2,sort_keys=True)+'\n').encode()
(local/'unit_test_receipt.json').write_bytes(raw)
for name in ['unit_tests.txt','unit_test_receipt.json']:
    with sftp.open(remote+'/'+name,'wx') as stream:
        stream.write((local/name).read_bytes())
sftp.close()
client.close()
print(out.decode())
print(json.dumps(test))
assert status==0
