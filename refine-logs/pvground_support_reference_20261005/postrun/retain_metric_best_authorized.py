"""Execute reviewed cleanup only after actual closed rows and fresh terminal audit."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import paramiko

local=Path(__file__).resolve().parents[1]
assert not (local/'weight_retention.json').exists()
wait=json.loads((local/'fit_wait.json').read_bytes())
assert wait['observer_closed'] and not wait['terminal']['controller_alive'] and wait['terminal']['exitcode']==0
summary=json.loads((local/'analysis/SUMMARY.json').read_bytes())
assert summary['status']=='ACTUAL_CLOSED_REFERENCE_ROWS_ANALYZED' and summary['fit_order_exact']
assert not (summary['scanrefer_target_pass'] and not summary['metric_best_target_pass']), 'Keep candidates when strict-best selection conflicts with the stated development line.'
assert all(not any(system['cpu_selected_coarse_reference_box_threshold_flip_counts'].values()) for system in summary['systems'].values())
audit=json.loads((local/'analysis/EXPERIMENT_AUDIT.json').read_bytes())
call=json.loads((local/'analysis/TERMINAL_REVIEW_CALL.json').read_bytes())
assert audit['verdict'] in ('PASS','WARN') and not audit['blocking_findings'] and call['result_received']
review=json.loads((local/'RETENTION_REVIEW.json').read_bytes())
assert review['verdict'] in ('PASS','WARN') and not review['blocking_findings'] and review['execution_scope']=='SOURCE_ONLY'
for entry in review['reviewed_files']:
    assert hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest()==entry['sha256'],entry['path']
spec=json.loads((local/'control_spec.json').read_bytes())
root=str(Path(spec['root']).parent).replace('\\','/')
decision=dict(winner=summary['metric_best_candidate']['system'],
    hits={row['system']:[row['rec_hits25'],row['rec_hits50']] for row in summary['table']})
client=paramiko.SSHClient();client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
sftp=client.open_sftp()
with sftp.open(root+'/retain_metric_best.py','wx') as stream:
    stream.write((local/'postrun/retain_metric_best.py').read_bytes())
with sftp.open(spec['runtime']+'/env_spec.json','rb') as stream:
    environment=json.loads(stream.read())
command=shlex.join(['env']+[key+'='+value for key,value in environment['env'].items()]
    +[spec['runtime']+'/venv/bin/python','-B',root+'/retain_metric_best.py',root,json.dumps(decision)])
_,stdout,stderr=client.exec_command(command,timeout=120)
raw=stdout.read();assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
record=json.loads(raw)
with sftp.open(root+'/weight_retention.json','rb') as stream:
    receipt=stream.read()
assert json.loads(receipt)==record
(local/'weight_retention.json').write_bytes(receipt)
sftp.close();client.close()
print(json.dumps(record),flush=True)
