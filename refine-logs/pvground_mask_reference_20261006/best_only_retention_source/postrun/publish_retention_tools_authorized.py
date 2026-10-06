"""Publish reviewed best-only retention sources without executing retention."""
import base64
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

import paramiko


root = Path(__file__).resolve().parents[1]
source_root = root/'postrun'
receipt_path = root/'retention_tools_publication.json'
assert not receipt_path.exists()
for filename in ('RETENTION_SOURCE_REVIEW.json', 'PUBLISH_RETENTION_TOOLS_REVIEW.json'):
    review = json.loads((source_root/filename).read_bytes())
    assert review['execution_scope']=='SOURCE_ONLY'
    assert review['verdict'] in ('PASS', 'WARN') and not review['blocking_findings']
    for item in review['reviewed_files']:
        assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256'], item['path']
previous = json.loads((root/'restore_tools_publication.json').read_bytes())
assert previous['section']=='20.376.84'
state_path = root.parent/'pvground_query_supported_geometry_20261005/active_continuation_state.json'
state = json.loads(state_path.read_bytes())
assert Path(state['latest_publication']).resolve()==(root/'restore_tools_publication.json').resolve()
assert state['owned_gpu_job_active'] and state['reference_fit_observer_native_session_id']==27853
assert not (root/'complete_fit').exists() and not (root/'analysis').exists()
workspace = Path('C:/Users/gb')
repos = [workspace/'.codex_mcln_g0_20260905', workspace/'.codex_pvground_cs_20261002',
         workspace/'.codex_mcln_v99_internal_20260928']
doc = 'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
copies = [repo/doc for repo in repos]+[workspace/'Desktop/document'/Path(doc).name]
old = copies[0].read_bytes()
assert hashlib.sha256(old).hexdigest()==previous['handoff_sha256']
assert all(path.read_bytes()==old for path in copies)
for repo, head in zip(repos, previous['heads']):
    assert subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip()==head
    assert not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'])
stamp = datetime.datetime.now().astimezone().isoformat()
prefix = 'refine-logs/pvground_mask_reference_20261006/best_only_retention_source/'
section = f'''

## 20.376.85 最佳权重保留工具源码已审查，尚未执行清理（{stamp}）

承接§84，当前Mask参考范围对照与既有观察器保持不变；本次仅发布训练结束后的清理实现和源码审查。未取得M1完整精度，未执行候选状态恢复或权重删除。当前保留的trained best仍为5616/4511（59.0660%/47.4443%），目标ACTIVE_UNMET；首次远端进度检查仍为17:14:14 UTC+8，之后240秒。

用户已授权清理本项目产生的已结束非最佳权重，不再为负结果创建权重归档。工具只处理五个明确路径：受保护的4511几何父权重，以及native_reference/fused_mask_reference各自initial.pth和terminal.pth。以实际9508条Acc@0.5选优，同严格命中时优先保留原最佳；其余证据、官方PV、原G及V99不在删除范围。

执行前必须有真实controller exit0与观察器闭合、四份9508结果及全256候选CPU复算、当前摘要身份绑定的fresh终态审计。若新候选胜出，还须实际严格CPU全模型/优化器恢复证明它不依赖即将删除的旧几何父权重。五个固定候选先核对身份；四个新候选核对头状态和正式结果，两份terminal另核对对应训练日志SHA；之后才删除非最佳项。未用历史审计替代本轮未来审计，也未提前运行这些门槛。

两份清理源码已通过SOURCE_ONLY审查，遗漏的terminal训练日志SHA核对已用一行修正，当前0个未解决阻断项。源码审查属于same-family/provisional，无backend attestation；这不是实际清理、实际恢复或新精度证据。发布目录为{prefix}，没有权重、原始NPZ或凭据。§84既有源码审查和历史快照保持原字节。
'''
assert chr(65533) not in section
new = old+section.encode('utf-8')
assert new.startswith(old) and new.count(b'## 20.376.85 ')==1
names = ['postrun/retain_metric_best.py', 'postrun/retain_metric_best_authorized.py',
         'postrun/RETENTION_SOURCE_REVIEW.json', 'postrun/RETENTION_SOURCE_REVIEW.md',
         'postrun/RETENTION_PREPARATION.json', 'postrun/publish_retention_tools_authorized.py',
         'postrun/PUBLISH_RETENTION_TOOLS_REVIEW.json', 'postrun/PUBLISH_RETENTION_TOOLS_REVIEW.md']
payloads = {prefix+name:(root/name).read_bytes() for name in names}
payloads[prefix+'.gitattributes'] = b'** -text\n'
assert all(not name.endswith(('.pth', '.pt', '.npz')) and '.aris' not in name for name in payloads)
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root',
               password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
spec = json.loads((root/'native_reference_spec.json').read_bytes())
remote_code = '''import base64,hashlib,json,sys
from pathlib import Path
project=Path(sys.argv[1]);bundle=json.load(sys.stdin)
doc=project/bundle['doc'];old=doc.read_bytes()
assert hashlib.sha256(old).hexdigest()==bundle['old_sha256']
evidence=(project/bundle['prefix']).resolve()
assert str(evidence)=='/root/autodl-tmp/mcln_published_evidence_20261005/pvground_mask_reference_20261006/best_only_retention_source'
assert not evidence.exists()
for name,encoded in bundle['files'].items():
    assert name.startswith(bundle['prefix']) and not name.endswith(('.pt','.pth','.npz')) and '.aris' not in name
    path=project/name;assert evidence in path.resolve().parents
    path.parent.mkdir(parents=True,exist_ok=True)
    raw=base64.b64decode(encoded)
    with path.open('xb') as stream:stream.write(raw)
    assert path.read_bytes()==raw
new=base64.b64decode(bundle['new_doc']);assert new.startswith(old)
doc.write_bytes(new);assert doc.read_bytes()==new
print(json.dumps(dict(files=len(bundle['files']),handoff_sha256=hashlib.sha256(new).hexdigest())))
'''
bundle = dict(doc=doc, prefix=prefix, old_sha256=previous['handoff_sha256'],
              new_doc=base64.b64encode(new).decode(),
              files={name:base64.b64encode(raw).decode() for name, raw in payloads.items()})
stdin, stdout, stderr = client.exec_command(shlex.join([spec['runtime']+'/venv/bin/python', '-B',
    '-c', remote_code, '/home/gb/new butd/butd_detr-main/MCLN-main']), timeout=180)
stdin.write(json.dumps(bundle).encode())
stdin.flush()
stdin.channel.shutdown_write()
raw = stdout.read()
assert stdout.channel.recv_exit_status()==0, stderr.read().decode()
remote = json.loads(raw)
client.close()
for name, raw in payloads.items():
    for repo in repos[:2]:
        destination = repo/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
for path in copies:
    path.write_bytes(new)
heads = []
for index, repo in enumerate(repos):
    stage = [doc]
    if index<2:
        with (repo/'MANIFEST.md').open('a', encoding='utf-8') as stream:
            stream.write('\n- '+stamp+' Best-only fixed-path retention tools SOURCE_ONLY reviewed; not executed.\n')
        stage += ['MANIFEST.md', *payloads]
    changed = subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain',
                                       '--untracked-files=all']).decode()
    assert {line[3:] for line in changed.splitlines()}.issubset(set(stage))
    subprocess.check_call(['git', '-C', str(repo), 'add', '-f', '--', *stage], stderr=subprocess.DEVNULL)
    subprocess.check_call(['git', '-C', str(repo), '-c', 'core.whitespace=cr-at-eol,-blank-at-eof',
                           'diff', '--cached', '--check', '--', *stage])
    if index<2:
        assert all(subprocess.check_output(['git', '-C', str(repo), 'show', ':'+name])==raw
                   for name, raw in payloads.items())
    assert subprocess.check_output(['git', '-C', str(repo), 'show', ':'+doc]).startswith(
        subprocess.check_output(['git', '-C', str(repo), 'show', previous['heads'][index]+':'+doc]))
    subprocess.check_call(['git', '-C', str(repo), 'commit', '--quiet', '-m',
                           'Record reviewed best-only Mask-reference weight retention tools'])
    heads.append(subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip())
    assert not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'])
subprocess.check_call(['git', '-C', str(repos[0]), '-c', 'http.version=HTTP/1.1',
                       'push', 'origin', 'HEAD:main'])
assert subprocess.check_output(['git', '-C', str(repos[0]), 'ls-remote', 'origin',
                                'refs/heads/main']).decode().split()[0]==heads[0]
digest = hashlib.sha256(new).hexdigest()
assert digest==remote['handoff_sha256'] and all(path.read_bytes()==new for path in copies)
guard = workspace/'.codex/tmp/sync_cs_handoff_remote_20260923.py'
raw = guard.read_bytes()
assert raw.count(previous['handoff_sha256'].encode())==1
guard.write_bytes(raw.replace(previous['handoff_sha256'].encode(), digest.encode()))
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(), section='20.376.85',
    heads=heads, handoff_sha256=digest, four_local_and_remote_equal=True, github_main=heads[0],
    payload_count=len(payloads), execution_scope='SOURCE_ONLY_RETENTION_TOOLS_PUBLISHED',
    retention_executed=False, candidate_restore_executed=False, new_accuracy_result=False,
    raw_npz_published=False, weights_changed=False, observer_native_session_id=27853)
receipt_path.write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
state.update(time_cst=record['time_cst'], latest_publication=str(receipt_path),
    handoff_section=record['section'], handoff_sha256=digest, published_heads=heads,
    reference_retention_tools_source_published=True, reference_retention_executed=False)
state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
with (workspace/'memory/2026-10-06.md').open('a', encoding='utf-8') as stream:
    stream.write('\nPV-Ground '+record['time_cst']+': Doc85 reviewed best-only retention tools published; no deletion/restore/new accuracy. Main '+heads[0]+'. M1 controller/sole observer unchanged; first remote check17:14:14 then240s. Goal active/unmet; trained best5616/4511 unchanged.\n')
print(json.dumps(record), flush=True)
