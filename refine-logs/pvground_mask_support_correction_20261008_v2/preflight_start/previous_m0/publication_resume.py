"""Finish the existing staged Doc98 after inherited-source whitespace failure."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

import paramiko


root = Path(__file__).resolve().parent
assert not (root / 'start_publication.json').exists()
previous = json.loads((root.parent / 'pvground_support_boundary_cases_20261007/terminal_publication.json').read_bytes())
workspace = Path('C:/Users/gb')
repos = [workspace / '.codex_mcln_g0_20260905',workspace / '.codex_pvground_cs_20261002',
         workspace / '.codex_mcln_v99_internal_20260928']
doc = 'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
prefix = 'refine-logs/pvground_mask_support_correction_20261008/preflight_start/'
copies = [repo / doc for repo in repos] + [workspace / 'Desktop/document' / Path(doc).name]
new = copies[0].read_bytes()
assert all(path.read_bytes() == new for path in copies)
git_old = subprocess.check_output(['git','-C',str(repos[0]),'show',previous['heads'][0]+':'+doc])
section_start = new.index(b'\n\n## 20.376.98 ')
old = new[:section_start]
assert hashlib.sha256(old).hexdigest() == previous['handoff_sha256']
assert git_old == old.replace(b'\r\n', b'\n')
assert new.startswith(old) and new.count(b'## 20.376.98 ') == 1
assert all(line.rstrip(b' \t') == line for line in new[section_start:].splitlines())
payload_names = [name[len(prefix):] for name in subprocess.check_output(
    ['git','-C',str(repos[0]),'diff','--cached','--name-only']).decode().splitlines() if name.startswith(prefix)]
assert len(payload_names) == 25
payloads = {prefix + name: (repos[0] / prefix / name).read_bytes() for name in payload_names}
assert all(not name.endswith(('.pth','.npz','.pt')) for name in payloads)
spec = json.loads((root / 'pair_spec.json').read_bytes())
for name, digest in spec['new_runner_files'].items():
    assert hashlib.sha256(payloads[prefix + name]).hexdigest() == digest
overlay_name = prefix + 'runtime_overlay/PV-Ground/models/pv_ground.py'
overlay = payloads[overlay_name]
assert overlay == (root / 'runtime_overlay/PV-Ground/models/pv_ground.py').read_bytes()
original = (root.parent / 'pvground_geometry_readback_20261004/revision2/source_preview/PV-Ground/models/pv_ground.py').read_bytes()
old_trailing = {line for line in original.splitlines() if line.rstrip(b' \t') != line}
new_trailing = {line for line in overlay.splitlines() if line.rstrip(b' \t') != line}
assert new_trailing.issubset(old_trailing)
digest = hashlib.sha256(new).hexdigest()
client = paramiko.SSHClient();client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
code = '''import hashlib,json,sys
from pathlib import Path
project=Path('/home/gb/new butd/butd_detr-main/MCLN-main');packet=json.load(sys.stdin)
assert hashlib.sha256((project/packet['doc']).read_bytes()).hexdigest()==packet['sha']
for name,digest in packet['files'].items():
    assert hashlib.sha256((project/name).read_bytes()).hexdigest()==digest,name
print(json.dumps(dict(remote_doc_and_payload_exact=True,remote_mutations=0,neural_forwards=0)))
'''
stdin,stdout,stderr=client.exec_command(shlex.join([spec['runtime']+'/venv/bin/python','-B','-c',code]),timeout=90)
stdin.write(json.dumps(dict(doc=doc,sha=digest,files={name:hashlib.sha256(raw).hexdigest() for name,raw in payloads.items()})))
stdin.channel.shutdown_write();raw=stdout.read()
assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
remote=json.loads(raw);client.close()
stamp=datetime.datetime.now().astimezone().isoformat()
heads=[]
for index,repo in enumerate(repos):
    assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip()==previous['heads'][index]
    attributes = repo / '.gitattributes'
    attribute_bytes = attributes.read_bytes()
    rule = (doc + ' -text').encode()
    if index == 0:
        # The first resume staged this rule before failing on the old index blob.
        assert rule in attribute_bytes.splitlines()
    else:
        assert rule not in attribute_bytes.splitlines()
        attributes.write_bytes(attribute_bytes + b'\n' + rule + b'\n')
    stage=[doc,'.gitattributes']
    if index<2:
        assert all((repo/name).read_bytes()==raw for name,raw in payloads.items())
        if index==1:
            with (repo/'MANIFEST.md').open('a',encoding='utf-8') as stream:
                stream.write('\n- '+stamp+' Mask-support correction implementation and initial GPU M0 start; no accuracy claim.\n')
        stage+=['MANIFEST.md',*payloads]
    changed=subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--untracked-files=all']).decode()
    assert {line[3:] for line in changed.splitlines()}.issubset(set(stage))
    subprocess.check_call(['git','-C',str(repo),'add','-f','--',*stage],stderr=subprocess.DEVNULL)
    subprocess.check_call(['git','-C',str(repo),'add','--renormalize','--',doc],stderr=subprocess.DEVNULL)
    # The exact old document and upstream PV snapshot retain inherited bytes.
    # New section whitespace and unchanged old prefix were checked above.
    lint=[name for name in stage if name not in (overlay_name,doc)]
    subprocess.check_call(['git','-C',str(repo),'-c','core.whitespace=cr-at-eol,-blank-at-eof','diff','--cached','--check','--',*lint])
    if index<2:
        assert all(subprocess.check_output(['git','-C',str(repo),'show',':'+name])==raw for name,raw in payloads.items())
    assert subprocess.check_output(['git','-C',str(repo),'show',':'+doc])==new
    subprocess.check_call(['git','-C',str(repo),'commit','--quiet','-m','Implement native Mask support correction and record initial GPU preflight start'])
    heads.append(subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip())
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'])
subprocess.check_call(['git','-C',str(repos[0]),'-c','http.version=HTTP/1.1','push','origin','HEAD:main'])
assert subprocess.check_output(['git','-C',str(repos[0]),'ls-remote','origin','refs/heads/main']).decode().split()[0]==heads[0]
guard=workspace/'.codex/tmp/sync_cs_handoff_remote_20260923.py';raw=guard.read_bytes()
assert raw.count(previous['handoff_sha256'].encode())==1
guard.write_bytes(raw.replace(previous['handoff_sha256'].encode(),digest.encode()))
wait=json.loads((root/'preflight_wait.json').read_bytes())
assert wait['observer_closed'] and not wait['controller_alive']
record=dict(time_cst=datetime.datetime.now().astimezone().isoformat(),section='20.376.98',heads=heads,
    handoff_sha256=digest,four_local_and_remote_equal=True,github_main=heads[0],payload_count=len(payloads),
    execution_scope='STAGED_SOURCE_START_PUBLICATION_RESUMED_ONLY',formal_accuracy_result=False,
    remote_source_or_doc_rewrites=0,repeated_nn_launches=0,inherited_overlay_whitespace_preserved=True,
    retained_best_hits=[5598,4848],overall_goal_complete=False,
    previous_git_doc_differs_only_by_crlf=True, per_document_raw_bytes_attribute_added=True,
    actual_initial_preflight_exitcode=wait['exitcode'],actual_initial_preflight_closed=True)
(root/'start_publication.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
state_path=root.parent/'pvground_query_supported_geometry_20261005/active_continuation_state.json'
state=json.loads(state_path.read_bytes())
state.update(latest_publication=str(root/'start_publication.json'),handoff_section='20.376.98',handoff_sha256=digest,
    published_heads=heads,status='MASK_SUPPORT_M0_IMPORT_FAILURE_CLOSED',owned_gpu_job_active=False,
    mask_support_correction_status='FIRST_M0_FAILED_BEFORE_MODEL_CONSTRUCTION_IMPORT_PATH',
    mask_support_correction_observer_closed=True,full_goal_status='ACTIVE_UNMET',
    next_action='Preserve failed original M0 exit1/observer57442 and Doc98 source/start receipt. Fix new loss import to use actual models.losses.scatter_mean; source-review one-line repair in a new immutable v2 preflight root before retry. No env install/rebuild/fallback, no formal scores or fit. Preserve best5598/4848 and Scan5620/4764+3effective then author-initNr/Sr.')
state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record),flush=True)
