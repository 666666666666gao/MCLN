"""Launch the two full native parent evaluations only after the actual preflight."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import paramiko

root=Path(__file__).resolve().parent
assert not (root/'FORMAL_LAUNCH.json').exists()
intake=json.loads((root/'preflight_actual/INTAKE.json').read_bytes())
assert intake['exitcode']==0
preflight=json.loads((root/'preflight_actual/preflight_receipt.json').read_bytes())
assert preflight['status']=='COMPLETED' and not preflight['formal_accuracy_result']
receipt=preflight['receipts'][0]
assert receipt['rows']==24 and receipt['batch']==24 and receipt['weight_state_unchanged']
assert receipt['model_state_tensors']==1234 and not receipt['observation_reader_installed']
assert receipt['state_free_observation_groupers']==10
spec=json.loads((root/'spec.json').read_bytes())
assert hashlib.sha256((root/'spec.json').read_bytes()).hexdigest()==receipt['spec_sha256']
remote='/root/autodl-tmp/pvground_pretrained_protocol_20261009/preflight_r3'
client=paramiko.SSHClient();client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
code="""import json,shutil,subprocess
print(json.dumps(dict(gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total','--format=csv,noheader,nounits']).decode(),compute_processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader']).decode(),data_free=shutil.disk_usage('/root/autodl-tmp').free)))
"""
_,stdout,stderr=client.exec_command(shlex.join([spec['runtime']+'/venv/bin/python','-B','-c',code]),timeout=30)
resources=json.loads(stdout.read());assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
assert int(resources['gpu'].strip().split(',')[1])<500 and not resources['compute_processes'].strip()
assert resources['data_free']>=240000000
sftp=client.open_sftp()
for name in ('evaluate_parent_batches.py','spec.json'):
    with sftp.open(remote+'/'+name,'rb') as stream:assert stream.read()==(root/name).read_bytes()
env=json.loads((root/'actual_env_spec.json').read_bytes())['env']
command=shlex.join(['env']+[key+'='+value for key,value in env.items()]+[spec['runtime']+'/venv/bin/python','-B','-u',remote+'/evaluate_parent_batches.py','--spec',remote+'/spec.json','--phase','formal'])
body=command+' >'+shlex.quote(remote+'/formal.log')+' 2>&1\nresult=$?\nprintf "%s\\n" "$result" >'+shlex.quote(remote+'/formal.exit')+'\nexit "$result"\n'
with sftp.open(remote+'/run_formal.sh','wx') as stream:stream.write(body.encode())
sftp.close()
screen='pvg_official_parent_batch_protocol_20261009'
_,stdout,stderr=client.exec_command(shlex.join(['screen','-dmS',screen,'bash',remote+'/run_formal.sh']),timeout=30)
assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
_,stdout,stderr=client.exec_command('pgrep -af '+shlex.quote('[e]valuate_parent_batches.py --spec '+remote+'/spec.json --phase formal'),timeout=30)
process=stdout.read().decode();assert stdout.channel.recv_exit_status()==0 and process.strip(),stderr.read().decode()
client.close()
stamp=datetime.datetime.now().astimezone()
old=json.loads(Path('C:/Users/gb/.codex_mcln_g0_20260905/refine-logs/pvground_scanrefer_formal_20260918_semantic_assignment_v1/published_parent/receipt.json').read_bytes())
old_seconds=old['elapsed_seconds']
estimate=2*old_seconds+180
first=estimate-180
record=dict(status='TWO_FULL_READ_ONLY_PARENT_BATCH_EVALUATIONS_LAUNCHED_NOT_COMPLETED',time_cst=stamp.isoformat(),remote_root=remote,process=process,screen=screen,batches=[8,24],rows_each=9508,resources=resources,seed=2027,multiseed=False,first_outcome_check_cst=(stamp+datetime.timedelta(seconds=first)).isoformat(),estimated_finish_cst=(stamp+datetime.timedelta(seconds=estimate)).isoformat(),estimated_seconds=estimate,later_poll_seconds=240,estimate_basis=dict(historical_batch8_full9508_seconds=old_seconds,startup_and_parser_allowance_seconds=180),new_weights=0,optimizer_updates=0,preflight_spec_sha256=receipt['spec_sha256'],preflight_intake_sha256=hashlib.sha256((root/'preflight_actual/INTAKE.json').read_bytes()).hexdigest(),current_best_unchanged=True,scope='Separate full bbs/bbf reports for author reproduction; batch protocol includes Gumbel draws and padding. Not a score switch or a new-method accuracy claim.')
(root/'FORMAL_LAUNCH.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record),flush=True)
