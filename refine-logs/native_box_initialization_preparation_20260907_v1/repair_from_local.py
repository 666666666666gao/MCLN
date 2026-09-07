import json
import os
from pathlib import Path
import shlex

import paramiko

local=Path('C:/Users/gb/.codex_mcln_g0_20260905/refine-logs/native_box_initialization_preparation_20260907_v1')
remote='/root/autodl-tmp/mcln_native_box_initialization_preparation_20260907_v1'
assert 'FileNotFoundError' in (local/'cpu_load_stderr.txt').read_text()
client=paramiko.SSHClient();client.load_system_host_keys();client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
sftp=client.open_sftp()
assert 'receipt.json' not in sftp.listdir(remote)
for name in ['check_native_box_initialization_cpu.py','cpu_load_stdout.txt','cpu_load_stderr.txt']:
    old=local/name;archived=old.with_name(old.stem+'.relativepath'+old.suffix)
    assert not archived.exists()
    old.rename(archived);sftp.rename(remote+'/'+name,remote+'/'+archived.name)
script=Path('C:/Users/gb/.codex/tmp/check_native_box_initialization_cpu_20260907.py').read_bytes()
(local/'check_native_box_initialization_cpu.py').write_bytes(script)
with sftp.open(remote+'/check_native_box_initialization_cpu.py','wx') as stream:stream.write(script)
command='cd '+shlex.quote(remote)+'\nCUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH='+shlex.quote(remote)+' /root/miniconda3/envs/bdetr/bin/python -u check_native_box_initialization_cpu.py > cpu_load_stdout.txt 2> cpu_load_stderr.txt'
_,output,error=client.exec_command(command,timeout=60)
output.read();errors=error.read().decode();code=output.channel.recv_exit_status()
for name in ['cpu_load_stdout.txt','cpu_load_stderr.txt','receipt.json']:
    if name in sftp.listdir(remote):
        size=sftp.stat(remote+'/'+name).st_size
        with sftp.open(remote+'/'+name,'rb') as stream:stream.prefetch(file_size=size);data=stream.read()
        (local/name).write_bytes(data)
assert code==0,{'exit':code,'stderr':errors}
receipt=json.loads((local/'receipt.json').read_bytes());assert receipt['status']=='pass'
record={'initial_errors':['Staged scripts package shadowed native scripts.train_rec_reranker before model loading',
                          'Relative __file__ directory changed meaning after chdir to model source'],
        'correction':'Bind scripts.__path__ to fixed native scripts and resolve the audit directory before changing cwd',
        'training_source_or_environment_changed':False,'unit_tests_repeated':False}
for name,data in [('repair_from_local.py',Path(__file__).read_bytes()),('preparation_correction.json',(json.dumps(record,indent=2)+'\n').encode())]:
    (local/name).write_bytes(data)
    with sftp.open(remote+'/'+name,'wx') as stream:stream.write(data)
sftp.close();client.close();print(json.dumps(receipt),flush=True)
