import ast,hashlib,json,os,shlex
from pathlib import Path
import paramiko
repo=Path('C:/Users/gb/.codex_mcln_g0_20260905')
local=repo/'refine-logs/scanrefer_native_box_transfer_posttraining_20260907_v1'
remote='/root/autodl-tmp/mcln_scanrefer_native_box_transfer_posttraining_20260907_v1'
stage=Path('C:/Users/gb/.codex/tmp/stage_native_box_transfer_queue_20260907.py')
assert local.resolve().is_relative_to(repo.resolve())
expected=json.loads((local/'input_manifest.json').read_bytes())['queue_script_sha256']
assert hashlib.sha256((local/'queue.py').read_bytes()).hexdigest()==expected
c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
s=c.open_sftp();assert 'launch.json' not in s.listdir(remote) and 'controller.exit' not in s.listdir(remote)
with s.open(remote+'/queue.py','rb') as f:assert hashlib.sha256(f.read()).hexdigest()==expected
s.rename(remote+'/queue.py',remote+'/posttraining_queue.py')
(local/'queue.py').rename(local/'posttraining_queue.py')
controller=(local/'controller.sh').read_bytes().replace(b' -u queue.py --manifest ',b' -u posttraining_queue.py --manifest ')
(local/'controller.sh').write_bytes(controller);s.put(str(local/'controller.sh'),remote+'/controller.sh')
command='cd '+shlex.quote(remote)+' && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/bdetr/bin/python -m pytest -q tests'
_,o,e=c.exec_command(command,timeout=45);tests=o.read().decode()+e.read().decode();assert o.channel.recv_exit_status()==0,tests
(local/'cpu_tests.txt').write_bytes(tests.encode());print(tests,flush=True)
tree=ast.parse(stage.read_text())
assignment=next(node for node in tree.body if isinstance(node,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='check' for x in node.targets))
check=ast.literal_eval(assignment.value.func.value).replace('BASE',repr(remote))
_,o,e=c.exec_command('/root/miniconda3/envs/bdetr/bin/python -c '+shlex.quote(check)+' && bash -n '+shlex.quote(remote+'/controller.sh'),timeout=45)
body=o.read().decode();err=e.read().decode();assert o.channel.recv_exit_status()==0,err
proof=json.loads(body)
proof['prelaunch_correction']={'old_name':'queue.py','new_name':'posttraining_queue.py','actual_error':"pytest plugin imported local queue instead of standard library: module 'queue' has no attribute 'Queue'",'training_was_not_modified':True}
(local/'preparation.json').write_bytes((json.dumps(proof,indent=2)+'\n').encode())
(local/'source_input_check.py').write_bytes(check.encode());(local/'stage_from_local.py').write_bytes(stage.read_bytes())
(local/'repair_prelaunch_from_local.py').write_bytes(Path(__file__).read_bytes())
for name in ['preparation.json','cpu_tests.txt','source_input_check.py']:
 s.put(str(local/name),remote+'/'+name)
_,o,e=c.exec_command('tail -n 7 /root/autodl-tmp/mcln_scanrefer_native_box_transfer_pair_20260907_v1/run.log',timeout=30)
print(o.read().decode(),flush=True)
s.close();c.close();print(json.dumps(proof),flush=True)
