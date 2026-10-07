"""Apply user-authorized best-only retention to this completed comparison."""
import hashlib
import json
import os
from pathlib import Path
import shlex

import paramiko


local=Path(__file__).resolve().parents[1]
assert not (local/'weight_retention.json').exists()
review=json.loads((local/'postrun/RETENTION_SOURCE_REVIEW.json').read_bytes())
assert review['execution_scope']=='SOURCE_ONLY' and review['verdict'] in ('PASS','WARN') and not review['blocking_findings']
for item in review['reviewed_files']:
    assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256']
summary_path=local/'analysis/SUMMARY.json'
inspection_path=local/'closed_weight_inspection.json'
audit_path=local/'analysis/EXPERIMENT_AUDIT.json'
summary=json.loads(summary_path.read_bytes())
inspection=json.loads(inspection_path.read_bytes())
audit=json.loads(audit_path.read_bytes())
call=json.loads((local/'analysis/TERMINAL_REVIEW_CALL.json').read_bytes())
assert call['result_received'] and audit['fresh_context']
assert audit['execution_scope']=='TERMINAL_SOURCE_AND_ACTUAL_ARTIFACTS'
assert audit['verdict'] in ('PASS','WARN') and not audit['blocking_findings']
for name,path in (('analysis/SUMMARY.json',summary_path),('closed_weight_inspection.json',inspection_path)):
    assert audit['actual_file_digests'][name]['sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
best=summary['metric_best_candidate']
winner=best['arm'] if best['arm']=='protected_geometry_parent' else best['arm']+'/'+best['stage']
assert winner==inspection['winner']
decision=dict(winner=winner,summary_sha256=hashlib.sha256(summary_path.read_bytes()).hexdigest(),
    inspection_sha256=hashlib.sha256(inspection_path.read_bytes()).hexdigest(),audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest())
spec=json.loads((local/'control_spec.json').read_bytes())
root='/root/autodl-tmp/pvground_reference_keep_20261006'
client=paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
sftp=client.open_sftp()
with sftp.open(spec['runtime']+'/env_spec.json','rb') as stream:
    environment=json.loads(stream.read())
assert hashlib.sha256(json.dumps(environment,sort_keys=True,separators=(',',':')).encode()).hexdigest()==spec['env_spec_sha256']
source=(local/'postrun/retain_closed_best.py').read_bytes()
with sftp.open(root+'/retain_closed_best.py','wx') as stream:
    stream.write(source)
with sftp.open(root+'/retain_closed_best.py','rb') as stream:
    assert stream.read()==source
command=shlex.join(['env']+[key+'='+value for key,value in environment['env'].items()]
    +[spec['runtime']+'/venv/bin/python','-B',root+'/retain_closed_best.py',json.dumps(decision)])
_,stdout,stderr=client.exec_command(command,timeout=180)
raw=stdout.read()
assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
record=json.loads(raw)
assert record['status']=='CLOSED_NONBEST_WEIGHTS_REMOVED' and record['retained_best']['candidate']==winner
with sftp.open(root+'/weight_retention.json','rb') as stream:
    receipt=stream.read()
assert json.loads(receipt)==record
(local/'weight_retention.json').write_bytes(receipt)
sftp.close()
client.close()
print(json.dumps(dict(status=record['status'],retained_best=record['retained_best'],
    deleted_count=len(record['deleted']),released_bytes=record['released_bytes'],raw_evidence_deleted=False)),flush=True)
