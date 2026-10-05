"""Complete the existing control publication with its immutable full JSONL."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess

import paramiko


root=Path(__file__).resolve().parent
assert not (root/'control_rows_publication.json').exists()
previous=json.loads((root/'control_closed_publication.json').read_bytes())
assert previous['section']=='20.376.74'
intake=json.loads((root/'CONTROL_CLOSED_INTAKE.json').read_bytes())
identity=intake['files']['control/formal/rows.jsonl']
row_path='closed_control_receipts/control/formal/rows.jsonl'
raw=(root/row_path).read_bytes()
assert len(raw)==identity['bytes'] and hashlib.sha256(raw).hexdigest()==identity['sha256']
repos=[Path(r'C:\Users\gb\.codex_mcln_g0_20260905'),Path(r'C:\Users\gb\.codex_pvground_cs_20261002'),
       Path(r'C:\Users\gb\.codex_mcln_v99_internal_20260928')]
doc='docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
doc_raw=(repos[0]/doc).read_bytes()
assert hashlib.sha256(doc_raw).hexdigest()==previous['handoff_sha256']
assert all((repo/doc).read_bytes()==doc_raw for repo in repos)
assert (Path(r'C:\Users\gb\Desktop\document')/Path(doc).name).read_bytes()==doc_raw
def git(repo,*args):
    return subprocess.check_output(['git','-C',str(repo),*args]).decode().strip()
for repo,head in zip(repos,previous['heads']):
    assert git(repo,'rev-parse','HEAD')==head and not git(repo,'status','--porcelain')
prefix='refine-logs/pvground_auxiliary_target_20261005/'
names=[row_path,'prepare_control_publication.py','publish_closed_control.py','publish_control_rows.py']
payloads={prefix+name:(root/name).read_bytes() for name in names}
client=paramiko.SSHClient();client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
sftp=client.open_sftp()
project='/home/gb/new butd/butd_detr-main/MCLN-main'
with sftp.open(project+'/'+doc,'rb') as stream:
    assert stream.read()==doc_raw
for name,contents in payloads.items():
    for repo in repos[:2]:
        path=repo/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(contents)
    with sftp.open(project+'/'+name,'wb') as stream:
        stream.write(contents)
    with sftp.open(project+'/'+name,'rb') as stream:
        assert stream.read()==contents
sftp.close();client.close()
stamp=datetime.datetime.now().astimezone().isoformat()
heads=[]
for repo in repos[:2]:
    subprocess.check_call(['git','-C',str(repo),'-c','core.autocrlf=false','add','-f','--',*payloads],stderr=subprocess.DEVNULL)
    subprocess.check_call(['git','-C',str(repo),'-c','core.whitespace=cr-at-eol','diff','--cached','--check','--',*payloads])
    for name,contents in payloads.items():
        assert subprocess.check_output(['git','-C',str(repo),'show',':'+name])==contents
    subprocess.check_call(['git','-C',str(repo),'commit','--quiet','-m','Complete closed control publication with all9508 raw rows'])
    heads.append(git(repo,'rev-parse','HEAD'))
    assert not git(repo,'status','--porcelain')
heads.append(previous['heads'][2])
subprocess.check_call(['git','-C',str(repos[0]),'-c','http.version=HTTP/1.1','push','origin','HEAD:main'])
assert git(repos[0],'ls-remote','origin','refs/heads/main').split()[0]==heads[0]
record=dict(previous)
record.update(time_cst=stamp,heads=heads,github_main=heads[0],status='CLOSED_CONTROL_FULL_JSONL_PUBLICATION_COMPLETED',
    full_control_rows_published=True,published_control_rows=9508,published_control_rows_sha256=identity['sha256'],
    published_control_rows_bytes=identity['bytes'],handoff_unchanged=True,weights_touched=0,
    inference_or_optimizer_replayed=False)
(root/'control_rows_publication.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
state_path=root.parent/'pvground_query_supported_geometry_20261005/active_continuation_state.json'
state=json.loads(state_path.read_bytes())
state.update(latest_publication=str(root/'control_rows_publication.json'),published_heads=heads,
    handoff_section='20.376.74',handoff_sha256=previous['handoff_sha256'],
    auxiliary_target_control_full_rows_published=True,current_goal_turn_classification='PROGRESS_COMPLETE_CONTROL_RAW_ROWS_PUBLICATION')
state_path.write_text(json.dumps(state,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
with Path(r'C:\Users\gb\memory\2026-10-05.md').open('a',encoding='utf-8') as stream:
    stream.write('\nPVGround '+stamp+': Completed Doc74 control publication with immutable all9508 JSONL7623290bytes/ec73b6..., remote and MAIN/PV indexbytes exact. Fixed actual publisher suffix omission to include.jsonl; no doc/result/training/weight changes. Main '+heads[0]+', same member fit/observer38531 continues.\n')
print(json.dumps(dict(section=record['section'],heads=heads,rows=9508,bytes=identity['bytes'],handoff_unchanged=True)),flush=True)
