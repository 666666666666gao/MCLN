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
prior = json.loads((root/'face_residual_cpu_publication.json').read_bytes())
assert prior['section']=='20.376.134' and prior['remote_handoff_sync_complete']
assert prior['github_main_verified']
prior_remote = json.loads(Path(prior['remote_sync_receipt']).read_bytes())
assert prior_remote['remote_sync_complete'] and prior_remote['doc_sha256']==prior['doc_sha256']
assert not (root/'normal_epoch1_publication.json').exists()
summary = json.loads((root/'NORMAL_FIRST_OBSERVATION_SUMMARY.json').read_bytes())
plan = json.loads((root/'NORMAL_NEXT_OBSERVATION_PLAN.json').read_bytes())
owner = json.loads((root/'NORMAL_EPOCH2_OBSERVER_OWNER.json').read_bytes())
assert owner['first_due_cst']==plan['due_cst'] and owner['local_pid']==51724
assert summary['epoch1_hits_delta']==[-25,-324] and summary['formal_results_unaudited']
assert json.loads((root/'NORMAL_FIRST_OBSERVATION_EXIT.json').read_bytes())['exit_code']==0
raw_observation = json.loads((root/'NORMAL_FIRST_OBSERVATION_STDOUT.json').read_bytes())
for row in raw_observation['files']:
    raw=base64.b64decode(row['base64'])
    assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
    assert (root/'normal_first_observation'/row['name']).read_bytes()==raw
assert base64.b64decode(raw_observation['train_tail_base64'])==(root/'normal_first_observation/train_tail.txt').read_bytes()
repos=[Path('C:/Users/gb')/name for name in (
    '.codex_mcln_g0_20260905','.codex_pvground_cs_20261002','.codex_mcln_v99_internal_20260928')]
doc='docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
desktop=Path('C:/Users/gb/Desktop/document')/Path(doc).name
old=(repos[0]/doc).read_bytes()
assert hashlib.sha256(old).hexdigest()==prior['doc_sha256'] and b'20.376.135' not in old
assert all((repo/doc).read_bytes()==old for repo in repos) and desktop.read_bytes()==old
for index,repo in enumerate(repos):
    assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip()==prior['heads'][index]
    assert not subprocess.check_output(['git','-C',str(repo),'-c','core.longpaths=true','status','--porcelain']).strip()
notes=(root/'HANDOFF_NORMAL_EPOCH1_20261010.md').read_text(encoding='utf-8')
document=old+('\n\n'+notes.replace('\r\n','\n')).replace('\n','\r\n').encode('utf-8')
digest=hashlib.sha256(document).hexdigest()
prefix='refine-logs/pvground_native_joint_training_20261009/normal_epoch1_observation_20261010/'
names=['HANDOFF_NORMAL_EPOCH1_20261010.md','NORMAL_FIRST_OBSERVATION.json',
    'NORMAL_FIRST_OBSERVATION_STDOUT.json','NORMAL_FIRST_OBSERVATION_EXIT.json',
    'NORMAL_FIRST_OBSERVATION_SUMMARY.json','NORMAL_NEXT_OBSERVATION_PLAN.json',
    'NORMAL_EPOCH2_OBSERVER_OWNER.json','prepare_normal_epoch2_observation.py',
    'observe_normal_epoch2_planned_authorized.py','prepare_normal_epoch1_publication.py',
    'publish_normal_epoch1_authorized.py']
names.extend(path.relative_to(root).as_posix() for path in (root/'normal_first_observation').rglob('*') if path.is_file())
files={prefix+name:(root/name).read_bytes() for name in sorted(set(names))}
files[prefix+'.gitattributes']=b'** -text whitespace=-blank-at-eol,-space-before-tab,cr-at-eol,-blank-at-eof\n'
assert not any('STDERR' in name or 'run_mcln_authorized_20260908.py' in name for name in files)
file_hashes={name:hashlib.sha256(raw).hexdigest() for name,raw in files.items()}
assert all(not (repo/prefix).exists() for repo in repos[:2])
code = r'''import base64,hashlib,json,sys
from pathlib import Path
b=json.load(sys.stdin);project=Path('/home/gb/new butd/butd_detr-main/MCLN-main');doc=project/b['doc']
assert hashlib.sha256(doc.read_bytes()).hexdigest()==b['old_sha']
leaf=(project/b['prefix']).resolve()
assert leaf==Path('/root/autodl-tmp/mcln_published_evidence_20261005/pvground_native_joint_training_20261009/normal_epoch1_observation_20261010') and not leaf.exists()
for name,data in b['files'].items():
 path=project/name;assert leaf in path.resolve().parents
 path.parent.mkdir(parents=True,exist_ok=True);raw=base64.b64decode(data);path.write_bytes(raw);assert path.read_bytes()==raw
raw=base64.b64decode(b['document']);temporary=doc.with_name(doc.name+'.tmp_normal_epoch1_observation')
assert not temporary.exists();temporary.write_bytes(raw);assert hashlib.sha256(temporary.read_bytes()).hexdigest()==b['sha'];temporary.replace(doc)
print(json.dumps(dict(doc_sha256=b['sha'],files=len(b['files']),neural_calls=0,training_status_reads=0)))
'''
witness = json.loads((root.parent / 'SCP_TRANSPORT_WITNESS.json').read_bytes())
environment = dict(os.environ, SSH_ASKPASS='C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/NativeSshAskPass.exe',
    SSH_ASKPASS_REQUIRE='force', DISPLAY='codex-byte-transfer')
runtime = '/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python'
argv = ['C:/Windows/System32/OpenSSH/ssh.exe', '-T', '-p', '33476', '-o', 'ProxyCommand=none',
    '-o', 'StrictHostKeyChecking=yes', '-o', 'UserKnownHostsFile=C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_scp_known_hosts',
    '-o', 'HostKeyAlgorithms=' + witness['negotiated_host_key_algorithm'], '-o', 'NumberOfPasswordPrompts=1',
    'root@region-9.autodl.pro', shlex.join([runtime, '-B', '-u', '-c', code])]
payload = dict(doc=doc, old_sha=prior['doc_sha256'], prefix=prefix, sha=digest,
    document=base64.b64encode(document).decode(),
    files={name: base64.b64encode(raw).decode() for name, raw in files.items()})
response = subprocess.run(argv, env=environment, input=json.dumps(payload).encode(),
    stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW)
(root / 'NORMAL_EPOCH1_PUBLICATION_REMOTE_STDOUT.json').write_bytes(response.stdout)
(root / 'NORMAL_EPOCH1_PUBLICATION_REMOTE_STDERR.txt').write_bytes(response.stderr)
(root / 'NORMAL_EPOCH1_PUBLICATION_REMOTE_EXIT.json').write_text(json.dumps(dict(exit_code=response.returncode)) + '\n', encoding='utf-8')
assert response.returncode == 0, 'Publication transport failed; inspect exact static state before any repair'
remote = json.loads(response.stdout)
assert remote['doc_sha256'] == digest and remote['files'] == len(files)
remote.update(time_cst=datetime.datetime.now().astimezone().isoformat(), remote_sync_complete=True)
(root / 'NORMAL_EPOCH1_PUBLICATION_REMOTE_RECEIPT.json').write_text(json.dumps(remote, indent=2) + '\n', encoding='utf-8')
heads = []
for index, repo in enumerate(repos):
    selected = [doc]
    if index < 2:
        for name, raw in files.items():
            target = Path('\\\\?\\' + str(repo / name))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        selected += sorted(files)
    (repo / doc).write_bytes(document)
    subprocess.check_call(['git', '-C', str(repo), '-c', 'core.longpaths=true', 'add', '-f', '--'] + selected)
    subprocess.check_call(['git', '-C', str(repo), '-c',
        'core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol', 'diff', '--cached', '--check'])
    assert subprocess.check_output(['git', '-C', str(repo), 'show', ':' + doc]) == document
    subprocess.check_call(['git', '-C', str(repo), 'commit', '--quiet', '-m',
        'Record first normal joint epoch and measured next observation time'])
    heads.append(subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip())
desktop.write_bytes(document)
local = dict(status='NORMAL_EPOCH1_ALL_LOCAL_AND_REMOTE_COPIES_COMMITTED_GITHUB_PENDING',
    section='20.376.135', time_cst=datetime.datetime.now().astimezone().isoformat(),
    heads=heads, doc_sha256=digest, prefix=prefix, new_files=len(files),
    file_sha256=file_hashes, remote_handoff_sync_complete=True, remote_evidence_sync_complete=True,
    remote_sync_receipt=str(root / 'NORMAL_EPOCH1_PUBLICATION_REMOTE_RECEIPT.json'),
    interim_results_unaudited=True, terminal_result=None,
    control_training_launched=False, Nr3D_or_Sr3D_training_launched=False,
    active_training_source_changed=False, normal_training_status_queries=0,
    observed_normal_metrics=summary['metrics'],
    next_observation_cst=plan['due_cst'], full_goal_complete=False)
(root / 'normal_epoch1_local_commit.json').write_text(json.dumps(local, indent=2) + '\n', encoding='utf-8')
subprocess.check_call(['git', '-C', str(repos[0]), '-c', 'http.version=HTTP/1.1', 'push', 'origin', 'HEAD:main'])
assert subprocess.check_output(['git', '-C', str(repos[0]), 'ls-remote', 'origin', 'refs/heads/main']).decode().split()[0] == heads[0]
assert all((repo / doc).read_bytes() == document for repo in repos) and desktop.read_bytes() == document
for repo in repos[:2]:
    for name, raw in files.items():
        assert Path('\\\\?\\' + str(repo / name)).read_bytes() == raw
assert all(not subprocess.check_output(['git', '-C', str(repo), '-c', 'core.longpaths=true', 'status', '--porcelain']).strip() for repo in repos)
guard = root.parent.parent / 'sync_cs_handoff_remote_20260923.py'
raw = guard.read_bytes()
assert prior['doc_sha256'].encode() in raw
guard.write_bytes(raw.replace(prior['doc_sha256'].encode(), digest.encode()))
local.update(status='NORMAL_EPOCH1_ALL_COPIES_AND_GITHUB_SYNCHRONIZED',
    github_main_verified=True, time_cst=datetime.datetime.now().astimezone().isoformat())
(root / 'normal_epoch1_publication.json').write_text(json.dumps(local, indent=2) + '\n', encoding='utf-8')
print(json.dumps({k:local[k] for k in ('status','section','heads','doc_sha256','new_files',
    'observed_normal_metrics','next_observation_cst','full_goal_complete')}),flush=True)
