"""Publish the existing scheduled observation without reading the active job again."""
import base64
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

root = Path(__file__).resolve().parent
prior = json.loads((root/'referit_metadata_publication.json').read_bytes())
assert prior['section']=='20.376.136' and prior['remote_handoff_sync_complete']
assert prior['github_main_verified']
prior_remote = json.loads(Path(prior['remote_sync_receipt']).read_bytes())
assert prior_remote['remote_sync_complete'] and prior_remote['doc_sha256']==prior['doc_sha256']
assert not (root/'normal_epoch2_publication.json').exists()
summary = json.loads((root/'NORMAL_EPOCH2_FULL_OBSERVATION_SUMMARY.json').read_bytes())
plan = json.loads((root/'NORMAL_EPOCH3_BOUNDARY_PLAN.json').read_bytes())
owner = json.loads((root/'NORMAL_EPOCH3_BOUNDARY_OBSERVER_OWNER.json').read_bytes())
assert owner['first_due_cst']==plan['due_cst'] and owner['local_pid']==45492
assert summary['epoch2_delta_from_initial']==[-125,-368] and summary['formal_results_unaudited']
assert json.loads((root/'NORMAL_EPOCH2_FINAL_OBSERVATION_EXIT.json').read_bytes())['exit_code']==0
raw_observation = json.loads((root/'NORMAL_EPOCH2_FINAL_OBSERVATION_STDOUT.json').read_bytes())
for row in raw_observation['files']:
    raw=base64.b64decode(row['base64'])
    assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
    assert (root/'normal_epoch2_final_observation'/row['name']).read_bytes()==raw
assert base64.b64decode(raw_observation['train_tail_base64'])==(root/'normal_epoch2_final_observation/train_tail.txt').read_bytes()
repos=[Path('C:/Users/gb')/name for name in (
    '.codex_mcln_g0_20260905','.codex_pvground_cs_20261002','.codex_mcln_v99_internal_20260928')]
doc='docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
desktop=Path('C:/Users/gb/Desktop/document')/Path(doc).name
old=(repos[0]/doc).read_bytes()
assert hashlib.sha256(old).hexdigest()==prior['doc_sha256'] and b'20.376.137' not in old
assert all((repo/doc).read_bytes()==old for repo in repos) and desktop.read_bytes()==old
for index,repo in enumerate(repos):
    assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip()==prior['heads'][index]
    assert not subprocess.check_output(['git','-C',str(repo),'-c','core.longpaths=true','status','--porcelain']).strip()
notes=(root/'HANDOFF_NORMAL_EPOCH2_20261010.md').read_text(encoding='utf-8')
document=old+('\n\n'+notes.replace('\r\n','\n')).replace('\n','\r\n').encode('utf-8')
digest=hashlib.sha256(document).hexdigest()
prefix='refine-logs/pvground_native_joint_training_20261009/normal_epoch2_observation_20261010/'
names=['HANDOFF_NORMAL_EPOCH2_20261010.md', 'NORMAL_EPOCH2_FULL_OBSERVATION_SUMMARY.json', 'NORMAL_EPOCH2_FINAL_OBSERVATION.json', 'NORMAL_EPOCH2_FINAL_OBSERVATION_STDOUT.json', 'NORMAL_EPOCH2_FINAL_OBSERVATION_EXIT.json', 'NORMAL_EPOCH2_OBSERVATION_SUMMARY.json', 'NORMAL_EPOCH2_BOUNDARY_TRANSPORT_FAILURE.json', 'NORMAL_EPOCH2_RECOVERY_SUMMARY.json', 'NORMAL_EPOCH3_BOUNDARY_PLAN.json', 'NORMAL_EPOCH3_BOUNDARY_OBSERVER_OWNER.json', 'NORMAL_EPOCH3_LOCAL_OWNER_VERIFIED_20261010.json', 'NORMAL_EPOCH3_MAINTENANCE_PUBLIC_SUMMARY.json', 'NORMAL_VALIDATION_POINT_INPUT_SOURCE_CHECK_20261010.md', 'NORMAL_VALIDATION_POINT_INPUT_SOURCE_CHECK_20261010.json', 'observe_normal_epoch2_final_authorized.py', 'observe_normal_epoch3_boundary_authorized.py', 'prepare_normal_epoch3_boundary.py', 'prepare_normal_epoch2_publication.py', 'publish_normal_epoch2_authorized.py']
names.extend(path.relative_to(root).as_posix() for path in (root/'normal_epoch2_final_observation').rglob('*') if path.is_file())
files={prefix+name:(root/name).read_bytes() for name in sorted(set(names))}
files[prefix+'.gitattributes']=b'** -text whitespace=-blank-at-eol,-space-before-tab,cr-at-eol,-blank-at-eof\n'
assert not any('STDERR' in name or 'run_mcln_authorized_20260908.py' in name for name in files)
file_hashes={name:hashlib.sha256(raw).hexdigest() for name,raw in files.items()}
assert all(not (repo/prefix).exists() for repo in repos[:2])
payload=dict(doc=doc,old_sha256=prior['doc_sha256'],new_sha256=digest,prefix=prefix,file_sha256=file_hashes)
(root/'NORMAL_EPOCH2_PUBLICATION_EXPECTED_AT_FAILURE.json').write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
code = r'''import hashlib,json,sys
from pathlib import Path
b=json.load(sys.stdin);p=Path('/home/gb/new butd/butd_detr-main/MCLN-main');d=p/b['doc'];leaf=p/b['prefix']
actual=hashlib.sha256(d.read_bytes()).hexdigest();present=0;missing=[];mismatch=[]
for n,h in b['file_sha256'].items():
 q=p/n
 if not q.is_file():missing.append(n)
 elif hashlib.sha256(q.read_bytes()).hexdigest()!=h:mismatch.append(n)
 else:present+=1
print(json.dumps(dict(doc_sha256=actual,old_document=actual==b['old_sha256'],new_document=actual==b['new_sha256'],
 prefix_exists=leaf.exists(),verified_files=present,expected_files=len(b['file_sha256']),missing=missing,mismatched=mismatch,
 training_status_queries=0)))
'''
witness = json.loads((root.parent/'SCP_TRANSPORT_WITNESS.json').read_bytes())
environment = dict(os.environ,SSH_ASKPASS='C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/NativeSshAskPass.exe',
    SSH_ASKPASS_REQUIRE='force',DISPLAY='codex-byte-transfer')
argv = ['C:/Windows/System32/OpenSSH/ssh.exe','-T','-p','33476','-o','ProxyCommand=none',
    '-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile=C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_scp_known_hosts',
    '-o','HostKeyAlgorithms='+witness['negotiated_host_key_algorithm'],'-o','NumberOfPasswordPrompts=1',
    'root@region-9.autodl.pro',shlex.join(['/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python','-B','-u','-c',code])]
response = subprocess.run(argv,env=environment,input=json.dumps(payload).encode(),stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,creationflags=subprocess.CREATE_NO_WINDOW)
(root/'NORMAL_EPOCH2_PUBLICATION_STATIC_PRIVATE_STDERR.txt').write_bytes(response.stderr)
(root/'NORMAL_EPOCH2_PUBLICATION_STATIC_RAW_STDOUT.json').write_bytes(response.stdout)
(root/'NORMAL_EPOCH2_PUBLICATION_STATIC_EXIT.json').write_text(json.dumps(dict(exit_code=response.returncode))+'\n')
assert response.returncode == 0
record = json.loads(response.stdout)
record.update(time_cst=datetime.datetime.now().astimezone().isoformat())
(root/'NORMAL_EPOCH2_PUBLICATION_STATIC_READ.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:v for k,v in record.items() if k not in ('missing','mismatched')}))
