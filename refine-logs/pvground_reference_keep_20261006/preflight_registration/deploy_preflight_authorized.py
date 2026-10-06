"""Deploy only the reviewed two-arm sanity from the protected4848 state."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex

import paramiko

local=Path(__file__).resolve().parent
assert not (local/'preflight_launch.json').exists()
previous=local.parent/'pvground_mask_reference_20261006'
accepted=json.loads((previous/'analysis/TERMINAL_ACCEPTANCE.json').read_bytes())
assert accepted['audit_result_received'] and not accepted['blocking_findings'] and accepted['selected_restore_complete']
assert accepted['metric_best_hits']==[5598,4848]
review=json.loads((local/'SOURCE_REVIEW.json').read_bytes())
assert review['execution_scope']=='SOURCE_ONLY' and review['verdict'] in ('PASS','WARN') and not review['blocking_findings']
for entry in review['reviewed_files']:
    assert hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest()==entry['sha256'],entry['path']
spec=json.loads((local/'control_spec.json').read_bytes())
root='/root/autodl-tmp/pvground_reference_keep_20261006'
assert spec['root']==root+'/control'
client=paramiko.SSHClient();client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
sftp=client.open_sftp()
with sftp.open(spec['runtime']+'/env_spec.json','rb') as stream:env=json.loads(stream.read())
assert hashlib.sha256(json.dumps(env,sort_keys=True,separators=(',',':')).encode()).hexdigest()==spec['env_spec_sha256']
python=spec['runtime']+'/venv/bin/python'
probe='''import hashlib,json,shutil,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]);best=Path(sys.argv[2]);assert not root.exists()
previous=Path('/root/autodl-tmp/pvground_mask_reference_20261006')
assert (previous/'fit_controller.exit').read_text().strip()=='0'
assert json.loads((previous/'fit_status.json').read_bytes())['status']=='complete'
assert hashlib.sha256(best.read_bytes()).hexdigest()==sys.argv[3]
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).decode().strip()
assert shutil.disk_usage(root.parent).free>64*1024**2
root.mkdir();(root/'control').mkdir();(root/'keep').mkdir()
print(json.dumps(dict(GPU_idle=True,protected4848_exact=True,data_free_bytes=shutil.disk_usage(root).free,system_free_bytes=shutil.disk_usage('/').free)))
'''
_,stdout,stderr=client.exec_command(shlex.join([python,'-B','-c',probe,root,spec['selected_terminal'],spec['selected_terminal_sha256']]),timeout=60)
raw=stdout.read();assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
resource=json.loads(raw);(local/'resource_check.json').write_bytes(raw)
names=('run_reference_keep_fit.py','reference_keep.py','selected_mask_reference_factory.py',
    'query_supported_geometry.py','mask_reference.py','check_invalid_reference.py',
    'invalid_reference_fixture.npz','invalid_reference_fixture.json','control_spec.json','keep_spec.json',
    'controller.py','EXPERIMENT_PLAN.md','research_contract.md','SOURCE_REVIEW.json','SOURCE_REVIEW.md')
for name in names:
    raw=(local/name).read_bytes()
    with sftp.open(root+'/'+name,'wx') as stream:stream.write(raw)
    with sftp.open(root+'/'+name,'rb') as stream:assert stream.read()==raw
command=shlex.join(['flock','-n',env['resource_limits']['gpu_lock'],python,'-B','-u',root+'/controller.py','--phase','preflight'])
shell=command+' > '+shlex.quote(root+'/preflight_controller.log')+' 2>&1; code=$?; printf "%s\\n" "$code" > '+shlex.quote(root+'/preflight_controller.exit')+'; exit "$code"'
screen='pvg_reference_keep_preflight_20261006'
_,stdout,stderr=client.exec_command(shlex.join(['screen','-dmS',screen,'bash','-c',shell]),timeout=30)
assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
pattern='^'+python+' -B -u '+root+'/controller.py --phase preflight$'
_,stdout,stderr=client.exec_command('pgrep -af '+shlex.quote(pattern),timeout=30)
process=stdout.read().decode().strip();assert stdout.channel.recv_exit_status()==0 and len(process.splitlines())==1,stderr.read().decode()
record=dict(time_cst=datetime.datetime.now().astimezone().isoformat(),root=root,controller_pid=int(process.split()[0]),
    process=process,screen=screen,resource=resource,status='PREFLIGHT_LAUNCHED_NOT_PASSED',
    optimizer_steps_planned_per_arm=2,new_weights_saved=0,accuracy_result=False,
    estimated_seconds=1000,first_check_seconds=720,later_poll_seconds=240,
    estimate_basis='Same prior M0 measured470s per arm; two arms estimated16-20min, first12min near projected closure.')
(local/'preflight_launch.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
with sftp.open(root+'/preflight_launch.json','wx') as stream:stream.write((local/'preflight_launch.json').read_bytes())
sftp.close();client.close();print(json.dumps(record),flush=True)
