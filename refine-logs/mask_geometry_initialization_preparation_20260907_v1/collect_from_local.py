import hashlib,json,os,shlex
from pathlib import Path
import paramiko
repo=Path('C:/Users/gb/.codex_mcln_g0_20260905')
local=repo/'refine-logs/mask_geometry_initialization_preparation_20260907_v1'
remote='/root/autodl-tmp/mcln_mask_geometry_initialization_preparation_20260907_v1'
c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
s=c.open_sftp();files=s.listdir(remote)
for name in ['controller.exit','cpu_tests.txt','cpu_load_stdout.txt','cpu_load_stderr.txt','receipt.json']:
    if name in files:
        with s.open(remote+'/'+name,'rb') as f:f.prefetch(file_size=s.stat(remote+'/'+name).st_size);raw=f.read()
        (local/name).write_bytes(raw)
assert 'controller.exit' in files,files
code=(local/'controller.exit').read_text().strip()
assert code=='0',{'exit':code,'stderr':(local/'cpu_load_stderr.txt').read_text(),'tests':(local/'cpu_tests.txt').read_text()}
record=json.loads((local/'receipt.json').read_bytes())
assert record['status']=='pass' and record['synthetic_fixture_only'] and not record['actual_trained_endpoint_loaded']
assert record['temporary_fixture_deleted'] and record['remaining_new_weight_files']==0
assert record['gpu_forwards']==record['actual_training_steps']==record['formal_rows']==0
assert all(v['changed_core_parameters_loaded']==84 and v['frozen_other_tensors_preserved']==1060 for v in record['protocols'].values())
manifest=json.loads((local/'input_manifest.json').read_bytes())
for name,digest in manifest['files'].items():
    with s.open(remote+'/'+name,'rb') as f:assert hashlib.sha256(f.read()).hexdigest()==digest,name
assert not any(name.endswith(('.pth','.pt')) for name in s.listdir(remote))
(local/'collect_from_local.py').write_bytes(Path(__file__).read_bytes())
print((local/'cpu_tests.txt').read_text(),flush=True);print(json.dumps(record),flush=True)
s.close();c.close()
