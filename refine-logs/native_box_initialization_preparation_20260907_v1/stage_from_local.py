import hashlib
import json
import os
from pathlib import Path
import shlex

import paramiko

repo=Path('C:/Users/gb/.codex_mcln_g0_20260905')
local=repo/'refine-logs/native_box_initialization_preparation_20260907_v1'
remote='/root/autodl-tmp/mcln_native_box_initialization_preparation_20260907_v1'
local.mkdir()
m=json.loads((repo/'refine-logs/native_transfer_input_audit_20260907_v1/input_manifest.json').read_bytes())
m.update(schema='mcln-native-box-initialization-preparation-v1',synthetic_fixture_only=True,
         training_directory='/root/autodl-tmp/mcln_scanrefer_native_box_transfer_pair_20260907_v1',
         training_manifest_sha256='2e7b7fa65afd056c0ad98a9a92e974e563df6e104624abdb30cfaf4716e1e72a')
files={'input_manifest.json':(json.dumps(m,indent=2)+'\n').encode(),
       'scripts/__init__.py':b'',
       'scripts/export_native_box_transfer_initialization.py':(repo/'scripts/export_native_box_transfer_initialization.py').read_bytes(),
       'tests/test_native_box_transfer_initialization.py':(repo/'tests/test_native_box_transfer_initialization.py').read_bytes(),
       'check_native_box_initialization_cpu.py':Path('C:/Users/gb/.codex/tmp/check_native_box_initialization_cpu_20260907.py').read_bytes(),
       'stage_from_local.py':Path(__file__).read_bytes()}
controller='''set -e
export CUDA_VISIBLE_DEVICES=
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH='''+shlex.quote(remote)+'''
cd '''+shlex.quote(remote)+'''
/root/miniconda3/envs/bdetr/bin/python -m pytest -q tests/test_native_box_transfer_initialization.py > cpu_tests.txt 2>&1
/root/miniconda3/envs/bdetr/bin/python -u check_native_box_initialization_cpu.py > cpu_load_stdout.txt 2> cpu_load_stderr.txt
'''
files['controller.sh']=controller.encode()
client=paramiko.SSHClient();client.load_system_host_keys();client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
sftp=client.open_sftp()
assert remote.rsplit('/',1)[1] not in sftp.listdir('/root/autodl-tmp')
sftp.mkdir(remote);sftp.mkdir(remote+'/scripts');sftp.mkdir(remote+'/tests')
for name,data in files.items():
    path=local/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    with sftp.open(remote+'/'+name,'wx') as stream:stream.write(data)
    with sftp.open(remote+'/'+name,'rb') as stream:assert stream.read()==data,name
_,output,error=client.exec_command('bash '+shlex.quote(remote+'/controller.sh'),timeout=60)
output.read();errors=error.read().decode();code=output.channel.recv_exit_status()
for name in ['cpu_tests.txt','cpu_load_stdout.txt','cpu_load_stderr.txt','receipt.json']:
    if name in sftp.listdir(remote):
        size=sftp.stat(remote+'/'+name).st_size
        with sftp.open(remote+'/'+name,'rb') as stream:stream.prefetch(file_size=size);data=stream.read()
        (local/name).write_bytes(data)
assert code==0,{'exit':code,'stderr':errors,'captured_files':sftp.listdir(remote)}
receipt=json.loads((local/'receipt.json').read_bytes())
assert receipt['status']=='pass' and receipt['synthetic_fixture_only'] and receipt['remaining_new_weight_files']==0
print((local/'cpu_tests.txt').read_text(),flush=True)
print(json.dumps(receipt),flush=True)
sftp.close();client.close()
