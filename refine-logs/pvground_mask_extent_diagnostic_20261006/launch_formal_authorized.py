"""Run the same reviewed diagnostic on9508 only after actual M0 acceptance."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import paramiko

local = Path(__file__).resolve().parent
assert not (local/'formal_launch.json').exists()
review = json.loads((local/'FORMAL_LAUNCH_REVIEW.json').read_bytes())
assert review['execution_scope']=='SOURCE_ONLY' and review['verdict'] in ('PASS','WARN') and not review['blocking_findings']
for item in review['reviewed_files']:
    assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256'],item['path']
wait = json.loads((local/'preflight_wait.json').read_bytes())
local_cpu = json.loads((local/'local_cpu_m0/CPU_SUMMARY.json').read_bytes())
assert wait['observer_closed'] and wait['controller']['completed'] and wait['controller']['exit_code']==0
assert local_cpu == wait['cpu'] and local_cpu['rows']==8 and local_cpu['preflight_actual_raw_member_rows_replayed']==8
assert not any(value for item in local_cpu['CPU_stored_threshold_flips'].values() for value in item.values())
spec = json.loads((local/'spec.json').read_bytes())
root = spec['root']
assert root=='/root/autodl-tmp/pvground_mask_extent_diagnostic_20261006' and spec['read_only_diagnostic']
pref = json.loads((local/'complete/preflight/receipt.json').read_bytes())
assert pref['model_states_unchanged'] and not pref['optimizer_created'] and pref['optimizer_updates']==0
forecast = ((9508+7)//8)*pref['preflight_formal_batch_bytes'] + 128*1024*1024
client = paramiko.SSHClient(); client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
probe = '''import hashlib,json,shutil,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]);expected=json.loads(sys.argv[2])
assert not (root/'formal_status.json').exists() and not (root/'formal').exists()
for name,digest in expected.items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
check=json.loads((root/'preflight/CPU_SUMMARY.json').read_text())
assert check['rows']==check['preflight_actual_raw_member_rows_replayed']==8
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).decode().strip()
assert not gpu,gpu
memory=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total','--format=csv,noheader,nounits']).decode().strip()
print(json.dumps(dict(gpu_compute_processes=gpu,gpu_memory=memory,data_free_bytes=shutil.disk_usage(str(root)).free)))
'''
expected = json.loads((local/'preflight_launch.json').read_bytes())['files']
_,stdout,stderr = client.exec_command(shlex.join([spec['runtime']+'/venv/bin/python','-B','-c',probe,root,json.dumps(expected)]),timeout=120)
resources = json.loads(stdout.read()); assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
assert resources['data_free_bytes'] > forecast
command = shlex.join(['screen','-dmS','pvg_mask_extent_formal_20261006',spec['runtime']+'/venv/bin/python',
                      '-B','-u',root+'/controller.py','--mode','formal'])
_,stdout,stderr = client.exec_command(command,timeout=120)
stdout.read(); assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
record = dict(status='ACTUAL_READ_ONLY_FORMAL_COMMAND_SUBMITTED',time_cst=datetime.datetime.now().astimezone().isoformat(),
              root=root,command=command,resources=resources,forecast_bytes_from_first_batch=forecast,
              estimate_total_seconds=1800,first_poll_before_estimated_end_seconds=180,remote_poll_seconds=240,
              forecast_is_sample_based_not_whole_split_bound=True,optimizer_updates=0,new_weights=0)
(local/'formal_launch.json').write_text(json.dumps(record,indent=2)+'\n')
client.close(); print(json.dumps(record))
