"""Publish saved PV evidence and verified archive availability; no training/NN operations."""
import base64
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

root=Path(__file__).resolve().parent
prior=json.loads((root/'pv_terminal_span_lr_publication_20261014.json').read_bytes())
assert prior['section']=='20.376.175' and prior['github_main_verified'] and prior['remote_handoff_sync_complete']
assert not (root/'pv_initial_and_dedup_publication_20261014.json').exists()
data=root/'pv_initial_and_dedup_publication_20261014'
summary=json.loads((data/'CLOSEOUT.json').read_bytes())
assert summary['retained_best_hits']==[5677,4920] and summary['goal_hits']==[5705,4945]
repos=[Path('C:/Users/gb')/name for name in ('.codex_mcln_g0_20260905','.codex_pvground_cs_20261002','.codex_mcln_v99_internal_20260928')]
doc='docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
desktop=Path('C:/Users/gb/Desktop/document')/Path(doc).name
old=(repos[0]/doc).read_bytes()
assert hashlib.sha256(old).hexdigest()==prior['doc_sha256'] and b'20.376.176' not in old
assert all((repo/doc).read_bytes()==old for repo in repos) and desktop.read_bytes()==old
for i,repo in enumerate(repos):
 assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip()==prior['heads'][i]
 assert not subprocess.check_output(['git','-C',str(repo),'-c','core.longpaths=true','status','--porcelain']).strip()
notes='## §20.376.176｜低B学习率完整起点与重复权重去重（2026-10-14）\n\n'+(data/'CLOSEOUT.md').read_text(encoding='utf-8')
document=old+('\n\n'+notes.replace('\r\n','\n')).replace('\n','\r\n').encode('utf-8')
digest=hashlib.sha256(document).hexdigest()
prefix='refine-logs/pvground_native_joint_training_20261009/pv_initial_and_dedup_20261014/'
files={prefix+path.name:path.read_bytes() for path in data.iterdir() if path.is_file()}
files[prefix+'.gitattributes']=b'** -text whitespace=-blank-at-eol,-space-before-tab,cr-at-eol,-blank-at-eof\n'
file_hashes={name:hashlib.sha256(raw).hexdigest() for name,raw in files.items()}
assert not any('STDERR' in name or '.aris' in name or 'credential' in name for name in files)
assert all(not (repo/prefix).exists() for repo in repos[:2])
remote_code = r'''import base64,hashlib,json,sys
from pathlib import Path
b=json.load(sys.stdin);project=Path('/home/gb/new butd/butd_detr-main/MCLN-main');doc=project/b['doc']
assert hashlib.sha256(doc.read_bytes()).hexdigest()==b['old_sha']
leaf=(project/b['prefix']).resolve()
assert leaf==Path('/root/autodl-tmp/mcln_published_evidence_20261005/pvground_native_joint_training_20261009/pv_initial_and_dedup_20261014') and not leaf.exists()
for name,data in b['files'].items():
 path=project/name;assert (leaf in path.resolve().parents or (name=='idea-stage/docs/research_contract.md' and project.resolve() in path.resolve().parents))
 path.parent.mkdir(parents=True,exist_ok=True);raw=base64.b64decode(data);path.write_bytes(raw);assert path.read_bytes()==raw
raw=base64.b64decode(b['document']);temporary=doc.with_name(doc.name+'.tmp_pv_initial_and_dedup')
assert not temporary.exists();temporary.write_bytes(raw);assert hashlib.sha256(temporary.read_bytes()).hexdigest()==b['sha'];temporary.replace(doc)
print(json.dumps(dict(doc_sha256=b['sha'],files=len(b['files']),neural_calls=0,training_status_reads=0)))
'''
witness = json.loads((root.parent / 'SCP_TRANSPORT_WITNESS.json').read_bytes())
environment = dict(os.environ,
    SSH_ASKPASS='C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/NativeSshAskPass.exe',
    SSH_ASKPASS_REQUIRE='force', DISPLAY='codex-byte-transfer')
runtime = '/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python'
argv = ['C:/Windows/System32/OpenSSH/ssh.exe', '-T', '-p', '33476', '-o', 'ProxyCommand=none',
    '-o', 'StrictHostKeyChecking=yes',
    '-o', 'UserKnownHostsFile=C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_scp_known_hosts',
    '-o', 'HostKeyAlgorithms=' + witness['negotiated_host_key_algorithm'], '-o', 'NumberOfPasswordPrompts=1',
    'root@region-9.autodl.pro', ' '.join(shlex.quote(part) for part in [runtime, '-B', '-u', '-c', remote_code])]
payload = dict(doc=doc, old_sha=prior['doc_sha256'], prefix=prefix, sha=digest,
    document=base64.b64encode(document).decode(),
    files={name: base64.b64encode(raw).decode() for name, raw in files.items()})
response = subprocess.run(argv, env=environment, input=json.dumps(payload).encode(),
    stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW)
(root / 'PV_INITIAL_AND_DEDUP_PUBLICATION_STDOUT.json').write_bytes(response.stdout)
(root / 'PV_INITIAL_AND_DEDUP_PUBLICATION_STDERR.txt').write_bytes(response.stderr)
(root / 'PV_INITIAL_AND_DEDUP_PUBLICATION_EXIT.json').write_text(
    json.dumps(dict(exit_code=response.returncode)) + '\n', encoding='utf-8')
assert response.returncode == 0, 'Publication transport failed; inspect exact static state before any repair'
remote = json.loads(response.stdout)
assert remote['doc_sha256'] == digest and remote['files'] == len(files)
remote.update(time_cst=datetime.datetime.now().astimezone().isoformat(), remote_sync_complete=True)
receipt_path = root / 'PV_INITIAL_AND_DEDUP_PUBLICATION_RECEIPT.json'
receipt_path.write_text(json.dumps(remote, indent=2) + '\n', encoding='utf-8')
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
        'Record complete PV E0 and equal model checkpoint deduplication'])
    heads.append(subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip())
desktop.write_bytes(document)
publication = dict(status='PV_INITIAL_AND_DEDUP_ALL_LOCAL_AND_REMOTE_COPIES_COMMITTED_GITHUB_PENDING',
    section='20.376.176', time_cst=datetime.datetime.now().astimezone().isoformat(),
    heads=heads, doc_sha256=digest, prefix=prefix, new_files=len(files), file_sha256=file_hashes,
    remote_handoff_sync_complete=True, remote_evidence_sync_complete=True,
    remote_sync_receipt=str(receipt_path), diagnostic_result=summary,
    selected_epoch=0, selected_epoch_scope='Lower_B_LR_complete_E0_5677_4920_real_first_epoch_started',
    next_observation_cst=summary['next_observation_cst'], observer_native_session=summary['observer_native_session'], observer_pid=summary['observer_pid'],
    MCLN_three_arm_evaluation_started=True, MCLN_training_started=summary['MCLN_formal_training_launched'], normal_training_alive_at_recorded_observation=True, normal_training_alive_scope='lower_span_lr controller and child verified03:20:49; real first epoch35batches; no new trained complete metric', PV_preflight_complete=True, PV_normal_controller_launched=True, MCLN_formal_training_complete=True, Nr3D_or_Sr3D_training_launched=False,
    active_training_source_changed=False, publication_training_status_queries=0,
    training_restart=False, full_goal_complete=False)
(root / 'pv_initial_and_dedup_local_commit_20261014.json').write_text(
    json.dumps(publication, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
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
publication.update(status='PV_INITIAL_AND_DEDUP_ALL_COPIES_AND_GITHUB_SYNCHRONIZED',
    github_main_verified=True, time_cst=datetime.datetime.now().astimezone().isoformat())
(root / 'pv_initial_and_dedup_publication_20261014.json').write_text(
    json.dumps(publication, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({key: publication[key] for key in (
    'status', 'section', 'heads', 'doc_sha256', 'new_files', 'selected_epoch', 'full_goal_complete')}), flush=True)
