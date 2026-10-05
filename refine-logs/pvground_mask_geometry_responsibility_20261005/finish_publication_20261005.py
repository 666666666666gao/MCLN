"""Finish the partial publication; respect existing Git text normalization."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import paramiko

local=Path(__file__).resolve().parent
workspace=Path('C:/Users/gb')
previous=json.loads((local.parent/'pvground_final_quality_20261005/terminal_remote_sync_publication.json').read_bytes())
failure=json.loads((local/'publication_failure.json').read_bytes())
assert not (local/'publication.json').exists()
assert all(item['LF_normalization_only'] for item in failure['mismatches'])
repos=[workspace/'.codex_mcln_g0_20260905',workspace/'.codex_pvground_cs_20261002',workspace/'.codex_mcln_v99_internal_20260928']
doc='docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
copies=[repo/doc for repo in repos]+[workspace/'Desktop/document'/Path(doc).name]
new=copies[0].read_bytes()
assert all(path.read_bytes()==new for path in copies)
assert hashlib.sha256(new).hexdigest()==failure['handoff_sha256']
names=['EXPERIMENT_PLAN.md','GEOMETRY_TARGET_NEXT_PLAN.md','SOURCE_REVIEW.md','SOURCE_REVIEW.json','SOURCE_REVIEW_CALL.json',
    'prepare_probe.py','probe_body.py','run_mask_geometry_probe.py','spec.json','GENERATION.json','controller.py',
    'launch_probe_authorized.py','observe_probe_authorized.py','collect_probe_authorized.py','analyze_probe.py',
    'resource_check.json','launch.json','wait.json','publish_probe_20261005.py','publication_failure.json',Path(__file__).name]
for folder in ('complete','analysis','observations'):
    names.extend(str(path.relative_to(local)).replace('\\','/') for path in sorted((local/folder).rglob('*')) if path.is_file())
prefix='refine-logs/pvground_mask_geometry_responsibility_20261005/'
payloads={prefix+name:(local/name).read_bytes() for name in names}
client=paramiko.SSHClient();client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
sftp=client.open_sftp()
project='/home/gb/new butd/butd_detr-main/MCLN-main'
with sftp.open(project+'/'+doc,'rb') as stream:
    assert stream.read()==new
for name in ('publication_failure.json',Path(__file__).name):
    with sftp.open(project+'/'+prefix+name,'wb') as stream:
        stream.write((local/name).read_bytes())
for name,raw in payloads.items():
    with sftp.open(project+'/'+name,'rb') as stream:
        assert stream.read()==raw
sftp.close();client.close()
stamp=datetime.datetime.now().astimezone().isoformat()
heads=[]
for index,repo in enumerate(repos):
    assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip()==previous['heads'][index]
    stage=[doc]
    if index<2:
        for name,raw in payloads.items():
            path=repo/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
        manifest=repo/'MANIFEST.md'
        if index==1:
            with manifest.open('a',encoding='utf-8') as stream:
                stream.write('\n- '+stamp+' Actual fixed64 augmentedfit Mask/Box geometry-role probe closed and audited; no optimizer/weights. Best5616/4506 unchanged, fixed-panel scope.\n')
        stage+=['MANIFEST.md',*payloads]
    changed=subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--untracked-files=all']).decode()
    assert {line[3:] for line in changed.splitlines()}.issubset(set(stage))
    subprocess.check_call(['git','-C',str(repo),'add','-f','--',*stage])
    subprocess.check_call(['git','-C',str(repo),'diff','--cached','--check','--',*stage])
    if index<2:
        for name,raw in payloads.items():
            indexed=subprocess.check_output(['git','-C',str(repo),'show',':'+name])
            expected=raw if name.endswith('.npz') else raw.replace(b'\r\n',b'\n')
            assert indexed==expected,name
    indexed=subprocess.check_output(['git','-C',str(repo),'show',':'+doc])
    old_index=subprocess.check_output(['git','-C',str(repo),'show',previous['heads'][index]+':'+doc])
    assert indexed.startswith(old_index) and indexed.count(b'## 20.376.62 ')==1
    subprocess.check_call(['git','-C',str(repo),'commit','--quiet','-m','Record actual fit Mask Box geometry supervision responsibility'])
    heads.append(subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip())
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'])
subprocess.check_call(['git','-C',str(repos[0]),'-c','http.version=HTTP/1.1','push','origin','HEAD:main'])
assert subprocess.check_output(['git','-C',str(repos[0]),'ls-remote','origin','refs/heads/main']).decode().split()[0]==heads[0]
guard=workspace/'.codex/tmp/sync_cs_handoff_remote_20260923.py'
raw=guard.read_bytes();assert raw.count(previous['handoff_sha256'].encode())==1
guard.write_bytes(raw.replace(previous['handoff_sha256'].encode(),failure['handoff_sha256'].encode()))
record=dict(time_cst=datetime.datetime.now().astimezone().isoformat(),section='20.376.62',heads=heads,
    github_main=heads[0],handoff_bytes=len(new),handoff_sha256=failure['handoff_sha256'],
    four_local_and_remote_equal=True,remote_sync_pending=False,payload_count=len(payloads),
    optimizer_steps=0,accuracy_result=False,new_weights_created_or_archived=False,
    protected_best_hits=[5616,4506],full_goal_status='ACTIVE_UNMET',
    recovery='Existing Git autocrlf normalized text only; exact raw local/remote evidence preserved')
(local/'publication.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
state=json.loads((local/'active_continuation_state.json').read_bytes())
state.update(time_cst=record['time_cst'],status='FIT_ROLE_PROBE_FULLY_PUBLISHED',active_reviewer=None,
    owned_gpu_job_active=False,observer_closed=True,latest_publication=str(local/'publication.json'),
    published_heads=heads,handoff_sha256=record['handoff_sha256'],handoff_bytes=len(new),handoff_section=record['section'],
    next_work_owner=str(local.parent/'pvground_mask_branch_responsibility_20261005'),
    next_action='Check candidate Query/Text/fused support qualification before defining additional geometry targets; no new loss/training executed.')
(local/'active_continuation_state.json').write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
with (workspace/'memory/2026-10-05.md').open('a',encoding='utf-8') as stream:
    stream.write('\nPVGround '+record['time_cst']+': actual augmentedfit64 geometry-role probe closed, audited WARN same-family/provisional, published doc62/main '+heads[0]+'. No optimizer/newweights/accuracy. Best5616/4506, gap248. 1112 fusedMask-only unmatched directBox/edge gradientszero. Shared Text requires candidate Query qualification before geometry-positive pool. Next work owner pvground_mask_branch_responsibility_20261005 source review pending.\n')
print(json.dumps(record),flush=True)
