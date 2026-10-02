"""Launch one reviewed full support arm after its real preflight has passed."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import paramiko

parser=argparse.ArgumentParser()
parser.add_argument('--arm',choices=('tail_raw','tail_fused'),required=True)
args=parser.parse_args()
local=Path(r'C:\Users\gb\.codex\tmp\pvground_fused_support_20261002')
marker=local/(args.arm+'_training_launch.json')
assert not marker.exists()
review=json.loads((local/'CODE_REVIEW_ROUND2.json').read_bytes())
assert review['verdict']=='PASS' and not review['blocking_findings']
for item in review['reviewed_files']:
    assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256']
launch_review=json.loads((local/'FULL_LAUNCH_REVIEW.json').read_bytes())
assert launch_review['verdict']=='PASS' and not launch_review['blocking_findings']
for item in launch_review['reviewed_files']:
    assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256']
root='/root/autodl-tmp/pvground_tail_support_20261002'
python='/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python'
client=paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',
               password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)

def run(command):
    _,stdout,stderr=client.exec_command(command,timeout=60)
    raw=stdout.read()
    error=stderr.read()
    assert stdout.channel.recv_exit_status()==0,error.decode()
    return raw.decode()

code="""
from pathlib import Path
import json,shutil,subprocess,sys
root=Path('/root/autodl-tmp/pvground_tail_support_20261002')
arm=sys.argv[1]
assert json.loads((Path('/root/autodl-tmp/pvground_p3_20261002')/'status.json').read_bytes())['status']=='complete'
assert json.loads((root/(arm+'_status.json')).read_bytes())['status']=='preflight_complete'
proof=json.loads((root/('preflight_'+arm)/'preflight.json').read_bytes())
assert proof['status']=='pass' and proof['optimizer_steps']==2
assert proof['support_arm']==arm and proof['member_mask_mapping_exact']
assert proof['direct_semantic_to_p3_gradients_zero']
assert proof['direct_native_mask_loss_to_refiner_gradients_zero']
assert proof['native_call_order_verified'] and len(proof['native_call_witnesses'])==4
if arm=='tail_fused':
    control=json.loads((root/'tail_raw/formal/receipt.json').read_bytes())
    assert control['status']=='pass' and control['rows']==9508
required=2*proof['serialization_bytes']+128*1024**2
free=shutil.disk_usage(root).free
assert free>=required,(free,required)
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],universal_newlines=True).strip()
assert not (root/arm/'train.log').exists()
print(json.dumps(dict(free_bytes=free,required_bytes=required,preflight=proof)))
"""
boundary=json.loads(run(shlex.quote(python)+' -c '+shlex.quote(code)+' '+shlex.quote(args.arm)))
sftp=client.open_sftp()
for name,path in (
    ('run.py',local/'run_pvground_tail_support.py'),
    ('controller.py',local/'run_pvground_tail_support_control.py'),
    ('pvground_tail_support_box_refiner.py',local/'pvground_tail_support_box_refiner.py'),
    ('pvground_tail_preflight.py',local/'pvground_tail_preflight.py')):
    with sftp.open(root+'/'+name,'rb') as stream:assert stream.read()==path.read_bytes()
with sftp.open(root+'/'+args.arm+'/spec.json','rb') as stream:
    raw=stream.read()
    assert raw==(local/(args.arm+'_spec.json')).read_bytes()
    spec=json.loads(raw)
assert spec['support_arm']==args.arm and spec['fused_support']==(args.arm=='tail_fused')
with sftp.open(spec['source_port'],'rb') as stream:
    assert hashlib.sha256(stream.read()).hexdigest()==spec['source_port_sha256']
with sftp.open(spec['runtime']+'/env_spec.json','rb') as stream:
    environment=json.loads(stream.read())
variables=dict(environment['env'])
variables['PYTHONPATH']=root+':'+variables['PYTHONPATH']
assign=' '.join(name+'='+shlex.quote(value) for name,value in variables.items())
inner=assign+' exec '+shlex.join([python,'-u',root+'/controller.py','--root',root,
    '--arm',args.arm,'--train-after-preflight'])+' > '+shlex.quote(root+'/'+args.arm+'_controller_training.log')+' 2>&1'
screen='pvg_'+args.arm+'_training_20261002'
run('screen -dmS '+screen+' bash -c '+shlex.quote(inner))
process=run('pgrep -af '+shlex.quote('^'+python+' -u '+root+'/controller.py --root '+root+' --arm '+args.arm+' --train-after-preflight$')).strip()
assert process
record=dict(time_cst=datetime.datetime.now().astimezone().isoformat(),root=root,
    support_arm=args.arm,screen=screen,process=process,
    capacity={key:boundary[key] for key in ('free_bytes','required_bytes')},
    preflight_pass=True,base='original G5615/4495',fresh_optimizer=True,
    seed=2027,batch_size=8,fit_rows=29778,training_steps=3723,
    formal_rows=9508,initial_rows=6887,primary_mode='bbs',
    full_training_started=True,formal_result_available=False,
    teacher=False,quality_loss=False,distribution=False,
    estimated_full_hours=3.5,poll_seconds=240)
marker.write_text(json.dumps(record,indent=2)+'\n')
with sftp.open(root+'/'+args.arm+'_training_launch.json','wx') as stream:
    stream.write(marker.read_bytes())
sftp.close()
client.close()
print(json.dumps(record))
