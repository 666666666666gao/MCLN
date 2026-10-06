"""Deploy only the reviewed closed-job reconstruction check; do not change weights."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex

import paramiko


root = Path(__file__).resolve().parents[1]
source_root = root/'postrun'
report_path = root/'selected_candidate_CPU_restore.json'
assert not report_path.exists()
review = json.loads((source_root/'RESTORE_SOURCE_REVIEW.json').read_bytes())
assert review['execution_scope']=='SOURCE_ONLY' and review['verdict'] in ('PASS','WARN') and not review['blocking_findings']
for item in review['reviewed_files']:
    assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256'], item['path']
analysis_review = json.loads((source_root/'ANALYSIS_SOURCE_REVIEW.json').read_bytes())
assert analysis_review['execution_scope']=='SOURCE_ONLY' and not analysis_review['blocking_findings']
assert hashlib.sha256((source_root/'analyze_mask_reference_formal.py').read_bytes()).hexdigest()==next(
    item['sha256'] for item in analysis_review['reviewed_files'] if Path(item['path']).name=='analyze_mask_reference_formal.py')
summary = json.loads((root/'analysis/SUMMARY.json').read_bytes())
assert summary['status']=='ACTUAL_CLOSED_ALL256_REFERENCE_PAIR_RECOUNTED' and summary['formal_rows_per_result']==9508
best = summary['metric_best_candidate']
assert best['arm'] in ('native_reference','fused_mask_reference') and best['stage'] in ('initial_formal','formal')
assert best['rec_hits50']>4511
spec_path = root/(best['arm']+'_spec.json')
spec = json.loads(spec_path.read_bytes())
remote_root = '/root/autodl-tmp/pvground_mask_reference_20261006'
remote_folder = remote_root+'/postrun_restoration'
files = {name:(source_root/name).read_bytes() for name in
    ('selected_mask_reference_factory.py','restore_candidate_state.py','RESTORE_SOURCE_REVIEW.json','RESTORE_SOURCE_REVIEW.md')}
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
sftp = client.open_sftp()
with sftp.file(spec['runtime']+'/env_spec.json','rb') as stream:
    env = json.loads(stream.read())
assert hashlib.sha256(json.dumps(env,sort_keys=True,separators=(',', ':')).encode()).hexdigest()==spec['env_spec_sha256']
probe = '''import hashlib,json,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]);assert root.resolve()==root
assert (root/'fit_controller.exit').read_text().strip()=='0'
status=json.loads((root/'fit_status.json').read_bytes())
assert status['status']=='complete' and status['protected_parents_exact']
spec=root/sys.argv[2]
assert hashlib.sha256(spec.read_bytes()).hexdigest()==sys.argv[3]
assert not (root/'postrun_restoration').exists()
pattern='^'+sys.argv[4]+'/venv/bin/python -B -u '+str(root)+'/controller.py --phase fit$'
assert subprocess.run(['pgrep','-af',pattern],stdout=subprocess.PIPE).returncode==1
(root/'postrun_restoration').mkdir()
print(json.dumps(dict(actual_controller_closed=True,exitcode=0,source_destination_created=True)))
'''
stdin, stdout, stderr = client.exec_command(shlex.join([spec['runtime']+'/venv/bin/python','-B','-c',probe,
    remote_root,spec_path.name,hashlib.sha256(spec_path.read_bytes()).hexdigest(),spec['runtime']]),timeout=60)
raw = stdout.read()
assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
closed_probe = json.loads(raw)
for name, raw in files.items():
    with sftp.file(remote_folder+'/'+name,'wb') as stream:
        stream.write(raw)
    with sftp.file(remote_folder+'/'+name,'rb') as stream:
        assert stream.read()==raw
variables = dict(env['env'])
command = ['env']+[key+'='+value for key,value in variables.items()]+[
    spec['runtime']+'/venv/bin/python','-B','-u',remote_folder+'/restore_candidate_state.py',
    '--root',remote_root,'--arm',best['arm'],'--stage',best['stage']]
_, stdout, stderr = client.exec_command(shlex.join(command),timeout=1800)
log = stdout.read()
error = stderr.read()
exitcode = stdout.channel.recv_exit_status()
(root/'selected_candidate_CPU_restore.log').write_bytes(log+error)
assert exitcode==0,error.decode()
with sftp.file(remote_folder+'/'+best['arm']+'_'+best['stage']+'.json','rb') as stream:
    receipt_raw = stream.read()
receipt = json.loads(receipt_raw)
assert receipt['status']=='pass' and receipt['arm']==best['arm'] and receipt['stage']==best['stage']
assert receipt['actual_step']==best['optimizer_updates']
assert [receipt['formal_metrics']['bbs'][key] for key in ('rec_hits25','rec_hits50')]==[best['rec_hits25'],best['rec_hits50']]
assert receipt['strict_CPU_geometry_restore'] and receipt['all_full_model_states_equal_to_original_construction']
assert not receipt['GPU_forward_replayed'] and receipt['weights_created']==receipt['weights_deleted']==0
sftp.close()
client.close()
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(),status='SELECTED_CANDIDATE_CPU_STATE_REBUILT',
    metric_best=best,actual_closed_probe=closed_probe,receipt=receipt,
    source_verdict=review['verdict'],analysis_sha256=hashlib.sha256((root/'analysis/SUMMARY.json').read_bytes()).hexdigest(),
    source_files={name:hashlib.sha256(raw).hexdigest() for name,raw in files.items()},
    new_full_validation=False,weights_changed=False)
report_path.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status=record['status'],arm=best['arm'],stage=best['stage'],step=receipt['actual_step'],
    strict_CPU_restore=True,optimizer_exact=True,weights_changed=False)),flush=True)
