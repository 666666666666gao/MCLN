"""Finish the partially staged launch publication after exact-byte Git failure."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess

import paramiko


local = Path(__file__).resolve().parent
workspace = Path('C:/Users/gb')
receipt_path = local/'launch_publication.json'
assert not receipt_path.exists()
previous = json.loads((local.parent/'pvground_geometry_readback_20261004/terminal_publication_20261005.json').read_bytes())
failure = json.loads((local/'PUBLISH_LINE_ENDING_FAILURE.json').read_bytes())
assert failure['gpu_source_changed'] is False and failure['inference_or_optimizer_replayed'] is False
repos = [workspace/'.codex_mcln_g0_20260905', workspace/'.codex_pvground_cs_20261002', workspace/'.codex_mcln_v99_internal_20260928']
doc = 'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
prefix = 'refine-logs/pvground_final_quality_20261005/'
copies = [r/doc for r in repos]+[workspace/'Desktop/document'/Path(doc).name]
new = copies[0].read_bytes()
assert hashlib.sha256(new[:previous['handoff_bytes']]).hexdigest()==previous['handoff_sha256']
assert new.count(b'## 20.376.58 ')==1 and all(p.read_bytes()==new for p in copies)
for repo, head in zip(repos, previous['heads']):
    assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip()==head
staged = subprocess.check_output(['git','-C',str(repos[0]),'diff','--cached','--name-only']).decode().splitlines()
names = [p[len(prefix):] for p in staged if p.startswith(prefix)]
assert len(names)==47
payloads = {prefix+n:(local/n).read_bytes() for n in names}
for name, raw in payloads.items():
    assert all((repo/name).read_bytes()==raw for repo in repos[:2])
extra = ['PUBLISH_LINE_ENDING_FAILURE.json', Path(__file__).name]
payloads.update({prefix+n:(local/n).read_bytes() for n in extra})
stamp = datetime.datetime.now().astimezone().isoformat()
rule = b'refine-logs/pvground_final_quality_20261005/** -text whitespace=cr-at-eol\n'
for index, repo in enumerate(repos[:2]):
    attr = repo/'.gitattributes'
    raw = attr.read_bytes()
    assert b'refine-logs/pvground_final_quality_20261005/**' not in raw
    attr.write_bytes(raw+b'\n# Preserve exact reviewed and collected final-quality evidence bytes.\n'+rule)
    for name in extra:
        (repo/prefix/name).write_bytes((local/name).read_bytes())
    with (repo/'MANIFEST.md').open('a',encoding='utf-8') as stream:
        if index==1:
            stream.write('\n- '+stamp+' final-native-bbs quality supervision: sourcePASS, actual2step preflightPASS, one3723update fit LAUNCHED; protected4506/control4477; no new accuracy.\n')
        stream.write('\n- '+stamp+' Exact-byte publication repair: scoped final-quality -text attribute; automatic CRLF conversion stopped the first publication before commit. No GPU source change or NN replay.\n')
    subprocess.check_call(['git','-C',str(repo),'add','--','.gitattributes'])
    subprocess.check_call(['git','-C',str(repo),'add','--renormalize','--',prefix])
    subprocess.check_call(['git','-C',str(repo),'add','-f','--','MANIFEST.md',doc,*payloads])
    stage = ['.gitattributes','MANIFEST.md',doc,*payloads]
    changed = subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--untracked-files=all']).decode()
    assert {line[3:] for line in changed.splitlines()}.issubset(set(stage))
    subprocess.check_call(['git','-C',str(repo),'diff','--cached','--check','--',*stage])
    for name, raw in payloads.items():
        assert subprocess.check_output(['git','-C',str(repo),'show',':'+name])==raw, name
    assert subprocess.check_output(['git','-C',str(repo),'show',':'+doc])==new
subprocess.check_call(['git','-C',str(repos[2]),'add','--',doc])
assert subprocess.check_output(['git','-C',str(repos[2]),'show',':'+doc])==new
assert {line[3:] for line in subprocess.check_output(['git','-C',str(repos[2]),'status','--porcelain']).decode().splitlines()}=={doc}

client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
sftp = client.open_sftp()
remote_project = '/home/gb/new butd/butd_detr-main/MCLN-main'
with sftp.open(remote_project+'/'+doc,'rb') as stream:
    assert stream.read()==new
for name, raw in payloads.items():
    if name[len(prefix):] in extra:
        with sftp.open(remote_project+'/'+name,'wb') as stream:
            stream.write(raw)
    with sftp.open(remote_project+'/'+name,'rb') as stream:
        assert stream.read()==raw, name
sftp.close()
client.close()
heads = []
for repo in repos:
    subprocess.check_call(['git','-C',str(repo),'commit','--quiet','-m','Publish real final-quality preflight and launch with exact evidence bytes'])
    heads.append(subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip())
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'])
subprocess.check_call(['git','-C',str(repos[0]),'-c','http.version=HTTP/1.1','push','origin','HEAD:main'])
assert subprocess.check_output(['git','-C',str(repos[0]),'ls-remote','origin','refs/heads/main']).decode().split()[0]==heads[0]
digest = hashlib.sha256(new).hexdigest()
guard = workspace/'.codex/tmp/sync_cs_handoff_remote_20260923.py'
raw = guard.read_bytes()
assert raw.count(previous['handoff_sha256'].encode())==1
guard.write_bytes(raw.replace(previous['handoff_sha256'].encode(),digest.encode()))
assert all(p.read_bytes()==new for p in copies)
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(),section='20.376.58',heads=heads,
    github_main=heads[0],handoff_bytes=len(new),handoff_sha256=digest,four_local_and_remote_equal=True,
    payload_count=len(payloads),source_review='PASS_SOURCE_ONLY_same_family_provisional',
    actual_two_step_preflight=True,formal_training_started=True,new_accuracy_result=False,
    protected_best_hits=[5616,4506],target_hits=[5615,4754],control_hits=[5615,4477],goal_achieved=False,
    publication_repair='Scoped Git -text after actual staged byte mismatch; no training source change')
receipt_path.write_bytes((json.dumps(record,indent=2)+'\n').encode())
state_path = local/'active_continuation_state.json'
state = json.loads(state_path.read_bytes())
state.update(time_cst=record['time_cst'],status='FORMAL_QUALITY_FIT_ACTIVE_PUBLISHED',active_reviewer=None,
    source_only=False,gpu_preflight_executed=True,formal_training_started=True,latest_publication=str(receipt_path),
    published_heads=heads,handoff_section='20.376.58',handoff_sha256=digest,handoff_bytes=len(new),
    last_goal_turn_classification='PROGRESS_ACTUAL_QUALITY_PREFLIGHT_AND_FORMAL_LAUNCH_PUBLISHED')
state_path.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
(local/'NEXT_CONTINUATION.md').write_text(
    '# Actual active continuation\n\n'
    'Sole actual formal-quality fit controller 546885; screen pvg_final_quality_fit_20261005. '
    'Sole native observer session 4460 is ACTIVE. Do not relaunch it. '
    'First remote check 2026-10-05 06:30:16 CST, then 240 seconds; estimated finish 07:03:36 CST.\n\n'
    'Read active_continuation_state.json and launch_publication.json. Source review and actual two-step preflight PASS; '
    'formal accuracy unknown. Reuse completed control4477; parents protected4506; target4754 ACTIVE/UNMET. '
    'No larger R or positive-count/normalization sweeps. Actual fit B8/29778 once/3723 updates. '
    'After terminal collect existing small results, CPU analyze rows, then required fresh terminal audit. '
    'Do not read authorization wrapper or raw goal objective. No negative weight archive.\n', encoding='utf-8')
for sibling in ('pvground_geometry_readback_20261004','pvground_face_conditioned_20261004'):
    with (local.parent/sibling/'NEXT_CONTINUATION.md').open('a',encoding='utf-8') as stream:
        stream.write('\n\nSuccessor active actual run: '+str(state_path)+'. Readback pair CLOSED; final-quality fit is sole active run. Publication '+str(receipt_path)+', main '+heads[0]+', doc58. Do not relaunch old observers/controllers.\n')
with (workspace/'memory/2026-10-05.md').open('a',encoding='utf-8') as stream:
    stream.write('\nPVGround '+record['time_cst']+': signed native-bbs/final-IoU difference supervision in originalG pool sourcePASS, real2step isolated-gradient/restore proofPASS. Actual3723update fit launched04:46:56, soleobserver4460 first06:30 then240sec. Reuses4477control/all256/oneScore/frozen4506. Published main '+heads[0]+' doc58, fourlocal/remote exact. Scoped -text repaired actual Git evidence-byte rejection before commit; GPU source unchanged. No newaccuracy; targetACTIVE_UNMET. Cursor '+str(state_path)+'.\n')
print(json.dumps(record),flush=True)
