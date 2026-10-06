"""Run fixed-path retention only after actual closure, fresh audit and restore."""
import hashlib
import json
import os
from pathlib import Path
import shlex

import paramiko


local = Path(__file__).resolve().parents[1]
assert not (local/'weight_retention.json').exists()
review = json.loads((local/'postrun/RETENTION_SOURCE_REVIEW.json').read_bytes())
assert review['execution_scope']=='SOURCE_ONLY' and review['verdict'] in ('PASS','WARN') and not review['blocking_findings']
for entry in review['reviewed_files']:
    assert hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest()==entry['sha256'],entry['path']
wait = json.loads((local/'fit_wait.json').read_bytes())
assert wait['observer_closed'] and not wait['terminal']['controller_alive'] and wait['terminal']['exitcode']==0
summary_path = local/'analysis/SUMMARY.json'
summary = json.loads(summary_path.read_bytes())
assert summary['status']=='ACTUAL_CLOSED_ALL256_REFERENCE_PAIR_RECOUNTED' and summary['fit_order_exact']
assert summary['formal_rows_per_result']==9508
for row in summary['table']:
    if row['arm']!='protected_geometry_parent':
        assert all(not any(value.values()) for value in row['cpu_selected_threshold_flips_by_box_type'].values())
        assert not any(row['cpu_full256_oracle_label_mismatches'].values())
audit_path = local/'analysis/EXPERIMENT_AUDIT.json'
audit = json.loads(audit_path.read_bytes())
call = json.loads((local/'analysis/TERMINAL_REVIEW_CALL.json').read_bytes())
assert audit['execution_scope']=='TERMINAL_SOURCE_AND_ACTUAL_ARTIFACTS'
assert audit['verdict'] in ('PASS','WARN') and not audit['blocking_findings']
assert audit['fresh_context'] and call['result_received']
assert audit['actual_file_digests']['analysis/SUMMARY.json']['sha256']==hashlib.sha256(summary_path.read_bytes()).hexdigest()
best = summary['metric_best_candidate']
decision = dict(winner=best['arm'] if best['arm']=='protected_geometry_parent' else best['arm']+'/'+best['stage'],
    hits={row['arm'] if row['arm']=='protected_geometry_parent' else row['arm']+'/'+row['stage']:
          [row['rec_hits25'],row['rec_hits50']] for row in summary['table']},
    summary_sha256=hashlib.sha256(summary_path.read_bytes()).hexdigest(),
    audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest())
if best['arm']!='protected_geometry_parent':
    restored_path = local/'selected_candidate_CPU_restore.json'
    restored = json.loads(restored_path.read_bytes())
    assert restored['status']=='SELECTED_CANDIDATE_CPU_STATE_REBUILT'
    assert audit['actual_file_digests']['selected_candidate_CPU_restore.json']['sha256']==hashlib.sha256(restored_path.read_bytes()).hexdigest()
    assert restored['analysis_sha256']==decision['summary_sha256']
    assert restored['receipt']['arm']==best['arm'] and restored['receipt']['stage']==best['stage']
    assert restored['receipt']['actual_step']==best['optimizer_updates']
    decision['selected_restore']=restored['receipt']
spec = json.loads((local/'native_reference_spec.json').read_bytes())
root = '/root/autodl-tmp/pvground_mask_reference_20261006'
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
sftp = client.open_sftp()
with sftp.open(spec['runtime']+'/env_spec.json','rb') as stream:
    environment = json.loads(stream.read())
assert hashlib.sha256(json.dumps(environment,sort_keys=True,separators=(',', ':')).encode()).hexdigest()==spec['env_spec_sha256']
script = (local/'postrun/retain_metric_best.py').read_bytes()
with sftp.open(root+'/retain_metric_best.py','wx') as stream:
    stream.write(script)
with sftp.open(root+'/retain_metric_best.py','rb') as stream:
    assert stream.read()==script
command = shlex.join(['env']+[key+'='+value for key,value in environment['env'].items()]
    +[spec['runtime']+'/venv/bin/python','-B',root+'/retain_metric_best.py',root,json.dumps(decision)])
_,stdout,stderr = client.exec_command(command,timeout=180)
raw = stdout.read()
assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
record = json.loads(raw)
with sftp.open(root+'/weight_retention.json','rb') as stream:
    receipt = stream.read()
assert json.loads(receipt)==record and record['retained_best']['candidate']==decision['winner']
(local/'weight_retention.json').write_bytes(receipt)
sftp.close()
client.close()
print(json.dumps(record),flush=True)
