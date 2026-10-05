"""Deploy only reviewed two-step engineering checks after branch qualification."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import paramiko

local=Path(__file__).resolve().parent
branch=local.parent/'pvground_mask_branch_responsibility_20261005'
summary=json.loads((branch/'analysis/SUMMARY.json').read_bytes())
audit=json.loads((branch/'analysis/EXPERIMENT_AUDIT.json').read_bytes())
assert summary['status']=='CPU_RECOUNT_PASS' and summary['counts']['own_query_confirmed']>0
assert audit['verdict'] in ('PASS','WARN') and not audit['blocking_findings']
review=json.loads((local/'SOURCE_REVIEW.json').read_bytes())
assert review['verdict'] in ('PASS','WARN') and not review['blocking_findings']
assert review['execution_scope']=='SOURCE_ONLY'
for entry in review['reviewed_files']:
    assert hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest()==entry['sha256'],entry['path']
assert not (local/'preflight_launch.json').exists()
root='/root/autodl-tmp/pvground_query_supported_geometry_20261005'
spec=json.loads((local/'control_spec.json').read_bytes())
client=paramiko.SSHClient();client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
sftp=client.open_sftp()
with sftp.open(spec['runtime']+'/env_spec.json','rb') as stream:
    env=json.loads(stream.read())
python=spec['runtime']+'/venv/bin/python'
probe='''
import json,subprocess,shutil,sys
from pathlib import Path
root=Path(sys.argv[1]);closed=Path('/root/autodl-tmp/pvground_mask_branch_responsibility_20261005')
assert not root.exists()
assert json.loads((closed/'status.json').read_bytes())['status']=='complete'
assert (closed/'controller.exit').read_text().strip()=='0'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
assert shutil.disk_usage(root.parent).free>64*1024**2
print(json.dumps(dict(GPU_idle=True,closed_branch_probe=True,data_free_bytes=shutil.disk_usage(root.parent).free,system_free_bytes=shutil.disk_usage('/').free)))
'''
_,stdout,stderr=client.exec_command(shlex.join([python,'-c',probe,root]),timeout=60)
raw=stdout.read();assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
(local/'resource_check.json').write_bytes(raw)
sftp.mkdir(root)
for arm in ('control','query_supported'):
    sftp.mkdir(root+'/'+arm)
for name in ('run_geometry_fit.py','query_supported_geometry.py','control_spec.json','query_supported_spec.json',
             'controller.py','EXPERIMENT_PLAN.md','SOURCE_REVIEW.md','SOURCE_REVIEW.json'):
    with sftp.open(root+'/'+name,'wx') as stream:
        stream.write((local/name).read_bytes())
command=shlex.join(['flock','-n',env['resource_limits']['gpu_lock'],python,'-B','-u',root+'/controller.py','--phase','preflight'])
inner=command+' > '+shlex.quote(root+'/preflight_controller.log')+' 2>&1; code=$?; printf "%s\\n" "$code" > '+shlex.quote(root+'/preflight_controller.exit')+'; exit "$code"'
screen='pvg_query_geometry_preflight_20261005'
_,stdout,stderr=client.exec_command(shlex.join(['screen','-dmS',screen,'bash','-c',inner]),timeout=30)
assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
pattern='^'+python+' -B -u '+root+'/controller.py --phase preflight$'
_,stdout,stderr=client.exec_command('pgrep -af '+shlex.quote(pattern),timeout=30)
process=stdout.read().decode().strip();assert stdout.channel.recv_exit_status()==0 and len(process.splitlines())==1,stderr.read().decode()
record=dict(time_cst=datetime.datetime.now().astimezone().isoformat(),root=root,process=process,screen=screen,
    status='PREFLIGHT_LAUNCHED_NOT_COMPLETED',optimizer_steps_planned_per_arm=2,accuracy_result=False,
    weight_files_planned=0,estimated_seconds=900,first_check_seconds=300,later_poll_seconds=240,
    estimate_basis='Two sequential fresh protected model reconstructions, two actual fit updates each; no full evaluation or training fit.')
(local/'preflight_launch.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
sftp.close();client.close()
print(json.dumps(record),flush=True)
