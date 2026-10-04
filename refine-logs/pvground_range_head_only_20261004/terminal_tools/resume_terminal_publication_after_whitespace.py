"""Finish the observed partial publication; preserve the exact raw audit report."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import paramiko

local=Path(__file__).resolve().parent
assert not (local/'terminal_publication.json').exists()
previous=json.loads((local/'local_phase_publication.json').read_bytes())
intake=json.loads((local/'complete/INTAKE.json').read_bytes())
summary=json.loads((local/'analysis/SUMMARY.json').read_bytes())
audit=json.loads((local/'analysis/EXPERIMENT_AUDIT.json').read_bytes())
trace=local/'.aris/traces/experiment-audit/2026-10-04_head_only_terminal_run01'
response=json.loads((trace/'002-terminal-integrity.response.json').read_bytes())
assert audit['verdict']=='WARN' and not audit['blocking_issues']
assert response['actual_task']=='/root/pvg_head_only_terminal_integrity'
assert response['review_independence']=='same-family' and response['acceptance_status']=='provisional'
for name,item in response['reviewer_reports'].items():
    raw=(local/'analysis'/name).read_bytes()
    assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256']
workspace=Path('C:/Users/gb')
repos=[workspace/'.codex_mcln_g0_20260905',workspace/'.codex_pvground_cs_20261002',workspace/'.codex_mcln_v99_internal_20260928']
doc='docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
copies=[repo/doc for repo in repos]+[workspace/'Desktop/document'/Path(doc).name]
new=copies[0].read_bytes()
assert all(path.read_bytes()==new for path in copies)
assert new.count(b'## 20.376.42 ')==1
old_working_prefix=new[:previous['handoff_bytes']]
assert hashlib.sha256(old_working_prefix).hexdigest()==previous['handoff_sha256']
for repo,head in zip(repos,previous['heads']):
    assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip()==head
    old=subprocess.check_output(['git','-C',str(repo),'show','HEAD:'+doc])
    # Observed existing document: working copies CRLF, Git stores LF (19995 CRLFs).
    assert old==old_working_prefix.replace(b'\r\n',b'\n')
prefix='refine-logs/pvground_range_head_only_20261004/'
payloads={}
for directory in ('complete','analysis'):
    for path in sorted((local/directory).rglob('*')):
        if path.is_file():payloads[prefix+path.relative_to(local).as_posix()]=path.read_bytes()
for path in sorted(trace.iterdir()):
    if path.is_file():payloads[prefix+'terminal_audit_trace/'+path.name]=path.read_bytes()
for name in ('collect_terminal.py','analyze_terminal.py','prepare_terminal_audit_request.py','TERMINAL_AUDIT_REQUEST.txt',
    'record_terminal_audit_request.py','record_terminal_audit_result.py','prepare_terminal_publisher.py','publish_terminal.py',
    'NEXT_EXPERIMENT_DECISION.md','terminal_collector_metadata_update.json','terminal_analyzer_text_update.json','terminal_audit_preparation.json'):
    payloads[prefix+'terminal_tools/'+name]=(local/name).read_bytes()
wait=json.loads((local/'wait.json').read_bytes())
latest=wait['observations'][-1]['file']
payloads[prefix+'terminal_tools/'+latest]=(local/latest).read_bytes()
# Prior publisher reached cached --check only AFTER its SFTP byte checks completed.
for repo in repos[:2]:
    for relative,raw in payloads.items():assert (repo/relative).read_bytes()==raw
marker='head-only pair complete9508 local[5588, 4447] whole[5589, 4456]; fresh audit WARN, owned nonbest retired.'
main_manifest=(repos[0]/'MANIFEST.md').read_text(encoding='utf-8')
lines=[line for line in main_manifest.splitlines() if marker in line]
assert len(lines)==1
assert marker not in (repos[1]/'MANIFEST.md').read_text(encoding='utf-8')
with (repos[1]/'MANIFEST.md').open('a',encoding='utf-8') as stream:stream.write('\n'+lines[0]+'\n')
relative=prefix+'terminal_tools/'+Path(__file__).name
raw=Path(__file__).read_bytes()
for repo in repos[:2]:
    assert not (repo/relative).exists()
    (repo/relative).write_bytes(raw)
payloads[relative]=raw
client=paramiko.SSHClient();client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
sftp=client.open_sftp();remote='/home/gb/new butd/butd_detr-main/MCLN-main'
with sftp.open(remote+'/'+doc,'rb') as stream:assert stream.read()==new
with sftp.open(remote+'/'+relative,'wx') as stream:stream.write(raw)
with sftp.open(remote+'/'+relative,'rb') as stream:assert stream.read()==raw
sftp.close();client.close()
heads=[]
audit_relative=prefix+'analysis/EXPERIMENT_AUDIT.md'
for index,repo in enumerate(repos):
    stage=[doc]+(['MANIFEST.md',*payloads] if index<2 else [])
    changed=subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--untracked-files=all']).decode()
    assert {line[3:] for line in changed.splitlines()}.issubset(set(stage))
    subprocess.check_call(['git','-C',str(repo),'add','-f','--',*stage])
    strict=[path for path in stage if path!=audit_relative]
    subprocess.check_call(['git','-C',str(repo),'-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol',
        'diff','--cached','--check','--',*strict])
    if index<2:
        # Only the immutable raw audit report may retain its observed final blank line.
        subprocess.check_call(['git','-C',str(repo),'-c','core.whitespace=blank-at-eol,space-before-tab,cr-at-eol',
            'diff','--cached','--check','--',audit_relative])
        for relative,raw in payloads.items():assert subprocess.check_output(['git','-C',str(repo),'show',':'+relative])==raw
    subprocess.check_call(['git','-C',str(repo),'commit','-m','Record complete stable G range comparison and raw fresh audit'])
    heads.append(subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip())
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'])
subprocess.check_call(['git','-C',str(repos[0]),'push','origin','HEAD:main'])
assert subprocess.check_output(['git','-C',str(repos[0]),'ls-remote','origin','refs/heads/main']).decode().split()[0]==heads[0]
for relative in (doc,*payloads):
    assert subprocess.check_output(['git','-C',str(repos[0]),'rev-parse','HEAD:'+relative])==subprocess.check_output(['git','-C',str(repos[1]),'rev-parse','HEAD:'+relative])
digest=hashlib.sha256(new).hexdigest()
guard=workspace/'.codex/tmp/sync_cs_handoff_remote_20260923.py'
guard_raw=guard.read_bytes()
assert guard_raw.count(previous['handoff_sha256'].encode())==1
guard.write_bytes(guard_raw.replace(previous['handoff_sha256'].encode(),digest.encode()))
retentions=[json.loads((local/'complete'/arm/'weight_retention.json').read_bytes()) for arm in ('local_range','whole_range')]
deleted=sum(item['bytes'] for record in retentions for item in record['deleted'])
metrics=summary['phases']['formal']['metrics']
hits=lambda arm:[metrics[arm]['bbs'][key] for key in ('rec_hits25','rec_hits50')]
record=dict(time_cst=datetime.datetime.now().astimezone().isoformat(),section='20.376.42',heads=heads,github_main=heads[0],
    handoff_sha256=digest,handoff_bytes=len(new),four_local_and_remote_equal=True,exact_committed_payloads=True,payload_count=len(payloads),
    local_bbs_hits=hits('local_range'),whole_bbs_hits=hits('whole_range'),whole_formal_complete=True,
    retained_metric_best=intake['status']['retained_best']['name'],owned_deleted_bytes=deleted,weights_downloaded=0,
    fresh_pair_audit_complete=True,audit_verdict=audit['verdict'],review_independence='same-family',acceptance_status='provisional',
    observed_publication_failure='original publisher2815 exit1: raw reviewer Markdown trailing blank line',
    recovery='preserved all exact raw reports; scoped EOF check exception to that one audit Markdown only',goal_achieved=False)
(local/'terminal_publication.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
state_path=local/'active_continuation_state.json';state=json.loads(state_path.read_bytes())
state.update(status='COMPLETE_AUDITED_PUBLISHED',latest_actual_publication=heads[0],handoff_sha256=digest,
    terminal_publication=record,terminal_publisher_execution=dict(session_id=2815,status='CLOSED_EXIT1_WHITESPACE_ONLY_RECOVERED'),
    goal_status='ACTIVE_UNMET',new_goal_achieved=False)
state_path.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
with (workspace/'memory/2026-10-04.md').open('a',encoding='utf-8') as stream:
    stream.write('\nPVGround '+record['time_cst']+' actual §20.376.42 published '+heads[0]+' fourlocal+remote SHA'+digest+
        '; raw audit preserved after known cached whitespace stop (only one raw MD EOF exception), verifiednonbest retired'+str(deleted)+'B; originalG best, goalACTIVE_UNMET.\n')
print(json.dumps(record),flush=True)
