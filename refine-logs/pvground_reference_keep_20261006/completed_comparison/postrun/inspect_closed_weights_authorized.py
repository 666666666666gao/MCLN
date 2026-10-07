"""Run one reviewed CPU witness after complete local result intake."""
import hashlib
import json
import os
from pathlib import Path
import shlex

import paramiko


local = Path(__file__).resolve().parents[1]
destination = local/'closed_weight_inspection.json'
assert not destination.exists()
review = json.loads((local/'postrun/INSPECTION_SOURCE_REVIEW.json').read_bytes())
assert review['verdict'] in ('PASS','WARN') and not review['blocking_findings']
assert review['execution_scope'] == 'SOURCE_ONLY'
for item in review['reviewed_files']:
    assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() == item['sha256']
summary_path = local/'analysis/SUMMARY.json'
summary = json.loads(summary_path.read_bytes())
assert summary['status'] == 'ACTUAL_CLOSED_ALL256_REFERENCE_KEEP_PAIR_RECOUNTED'
assert summary['formal_rows_per_result'] == 9508 and summary['fit_order_exact']
for row in summary['table']:
    if row['arm'] != 'protected_geometry_parent':
        assert all(not any(item.values()) for item in row['cpu_selected_threshold_flips_by_box_type'].values())
        assert not any(row['cpu_full256_oracle_label_mismatches'].values())
best = summary['metric_best_candidate']
decision = dict(winner=best['arm'] if best['arm']=='protected_geometry_parent' else best['arm']+'/'+best['stage'],
    hits={row['arm'] if row['arm']=='protected_geometry_parent' else row['arm']+'/'+row['stage']:
          [row['rec_hits25'],row['rec_hits50']] for row in summary['table']},
    summary_sha256=hashlib.sha256(summary_path.read_bytes()).hexdigest())
wait = json.loads((local/'fit_wait.json').read_bytes())
assert wait['observer_closed'] and not wait['terminal']['controller_alive'] and wait['terminal']['exitcode']==0
spec = json.loads((local/'control_spec.json').read_bytes())
root = '/root/autodl-tmp/pvground_reference_keep_20261006'
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
sftp = client.open_sftp()
with sftp.open(spec['runtime']+'/env_spec.json','rb') as stream:
    environment = json.loads(stream.read())
assert hashlib.sha256(json.dumps(environment,sort_keys=True,separators=(',',':')).encode()).hexdigest()==spec['env_spec_sha256']
source = (local/'postrun/inspect_closed_weights.py').read_bytes()
with sftp.open(root+'/inspect_closed_weights.py','wx') as stream:
    stream.write(source)
with sftp.open(root+'/inspect_closed_weights.py','rb') as stream:
    assert stream.read()==source
command = shlex.join(['env']+[key+'='+value for key,value in environment['env'].items()]
    +[spec['runtime']+'/venv/bin/python','-B',root+'/inspect_closed_weights.py',json.dumps(decision)])
_,stdout,stderr = client.exec_command(command,timeout=300)
raw = stdout.read()
assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
record = json.loads(raw)
assert record['status']=='CLOSED_FIXED_CHECKPOINTS_CPU_INSPECTED' and record['winner']==decision['winner']
assert record['script_sha256']==hashlib.sha256(source).hexdigest()
assert record['weights_created']==record['weights_deleted']==record['optimizer_updates']==0
with sftp.open(root+'/closed_weight_inspection.json','rb') as stream:
    receipt = stream.read()
assert json.loads(receipt)==record
destination.write_bytes(receipt)
sftp.close()
client.close()
print(json.dumps({'status':record['status'],'winner':record['winner'],'checkpoint_count':len(record['identities']),
    'weights_deleted':0,'GPU_forward_replayed':False}),flush=True)
