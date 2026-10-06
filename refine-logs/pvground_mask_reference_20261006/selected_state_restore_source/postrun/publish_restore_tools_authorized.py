"""Publish reviewed reconstruction tools, explicitly before any actual restore."""
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
receipt_path = root/'restore_tools_publication.json'
assert not receipt_path.exists()
for filename in ('RESTORE_SOURCE_REVIEW.json','PUBLISH_RESTORE_TOOLS_REVIEW.json'):
    review = json.loads((source_root/filename).read_bytes())
    assert review['execution_scope']=='SOURCE_ONLY' and review['verdict'] in ('PASS','WARN') and not review['blocking_findings']
    for item in review['reviewed_files']:
        assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256'],item['path']
previous = json.loads((root/'fit_launch_publication.json').read_bytes())
assert previous['section']=='20.376.83'
state_path = root.parent/'pvground_query_supported_geometry_20261005/active_continuation_state.json'
state = json.loads(state_path.read_bytes())
assert Path(state['latest_publication']).resolve()==(root/'fit_launch_publication.json').resolve()
assert state['owned_gpu_job_active'] and state['reference_fit_observer_native_session_id']==27853
assert not (root/'complete_fit').exists() and not (root/'analysis').exists()
workspace = Path('C:/Users/gb')
repos = [workspace/'.codex_mcln_g0_20260905',workspace/'.codex_pvground_cs_20261002',workspace/'.codex_mcln_v99_internal_20260928']
doc = 'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
copies = [repo/doc for repo in repos]+[workspace/'Desktop/document'/Path(doc).name]
old = copies[0].read_bytes()
assert hashlib.sha256(old).hexdigest()==previous['handoff_sha256'] and all(path.read_bytes()==old for path in copies)
for repo,head in zip(repos,previous['heads']):
    assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip()==head
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'])
stamp = datetime.datetime.now().astimezone().isoformat()
prefix = 'refine-logs/pvground_mask_reference_20261006/selected_state_restore_source/'
section = f'''

## 20.376.84 声明候选的严格重建工具已做源码复核，尚未执行终态重载（{stamp}）

承接§83的活动完整对照，未重复启动GPU实验，也未修改训练、参考、损失或观察器。11:33:10 UTC+8同一工具批次的现有native观察器27853返回running句柄；本次没有提前SSH查进度，首次远端观察仍17:14:14，之后240秒。当前仍未取得M1新9508结果，trainedbest仍5616/4511，目标ACTIVE_UNMET。

新增的postrun分析器按四份actual9508分开复算initial_formal与formal、全部256 prior/reference/final和原生分数；CPU浮点阈值差异按三类框分别报告，不替代原生主指标。声明最佳选择继续以Acc@0.5优先、同严格命中保留原保护权重。其SOURCE_ONLY复核通过，不代表终态数据已经复算。

当前旧工厂明确只接受step3723，不能把新initial.pth的真实step0改成3723来重载。新增selected_mask_reference_factory保留同样PV/G、几何头和冻结零输出R的构造顺序，直接严格加载候选完整10个几何状态。后续CPU检查会分别验证step0的hidden8与真实父权重一致、output2全零和emptyAdam，或terminal的3723步、10个moment及完整模型状态与原构造路径逐张量一致。旧几何文件仅用于这次构造比较见证；是否确能解除依赖仍须实际执行证明。

三份重载工具及独立发布器已SOURCE_ONLY复核；它们尚未部署或重载候选，没有新的模型CPU重载、GPU前向、完整评估、权重生成或删除。实际调用须在当前controller exit0和两组完整结果之后；CPU重载不宣称独立GPU重放。后续仍需fresh终态审计、实际选中状态恢复和best-only清理。原PV/G、V99与活动恢复保持保护，不建立负结果权重归档。

本节仅公布源码与范围：{prefix}；未上传原始NPZ、权重或凭据。四本地与远端交接按原字节前缀续写，GitHub同步；不将预检/源码审查写成精度提升。
'''
assert chr(65533) not in section
new = old+section.encode('utf-8')
assert new.startswith(old) and new.count(b'## 20.376.84 ')==1
names = ['postrun/analyze_mask_reference_formal.py','postrun/ANALYSIS_SOURCE_REVIEW.json','postrun/ANALYSIS_SOURCE_REVIEW.md',
    'postrun/selected_mask_reference_factory.py','postrun/restore_candidate_state.py','postrun/restore_selected_candidate_authorized.py',
    'postrun/RESTORE_SOURCE_REVIEW.json','postrun/RESTORE_SOURCE_REVIEW.md','postrun/publish_restore_tools_authorized.py',
    'postrun/PUBLISH_RESTORE_TOOLS_REVIEW.json','postrun/PUBLISH_RESTORE_TOOLS_REVIEW.md',
    'POSTRUN_TASKS.md','POSTRUN_PREPARATION.json','LIVE_OBSERVER_HANDLE_WITNESS_20261006_113310.json']
payloads = {prefix+name:(root/name).read_bytes() for name in names}
payloads[prefix+'.gitattributes'] = b'** -text\n'
assert all(not name.endswith(('.pth','.pt','.npz')) and '.aris' not in name for name in payloads)
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
spec = json.loads((root/'native_reference_spec.json').read_bytes())
remote_code = '''import base64,hashlib,json,sys
from pathlib import Path
project=Path(sys.argv[1]);bundle=json.load(sys.stdin)
doc=project/bundle['doc'];old=doc.read_bytes()
assert hashlib.sha256(old).hexdigest()==bundle['old_sha256']
evidence=(project/bundle['prefix']).resolve()
assert str(evidence)=='/root/autodl-tmp/mcln_published_evidence_20261005/pvground_mask_reference_20261006/selected_state_restore_source'
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
bundle = dict(doc=doc,prefix=prefix,old_sha256=previous['handoff_sha256'],new_doc=base64.b64encode(new).decode(),
    files={name:base64.b64encode(raw).decode() for name,raw in payloads.items()})
stdin,stdout,stderr = client.exec_command(shlex.join([spec['runtime']+'/venv/bin/python','-B','-c',remote_code,
    '/home/gb/new butd/butd_detr-main/MCLN-main']),timeout=180)
stdin.write(json.dumps(bundle).encode());stdin.flush();stdin.channel.shutdown_write()
raw = stdout.read()
assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
remote = json.loads(raw)
client.close()
for name,raw in payloads.items():
    for repo in repos[:2]:
        destination = repo/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes(raw)
for path in copies:
    path.write_bytes(new)
heads = []
for index,repo in enumerate(repos):
    stage = [doc]
    if index<2:
        with (repo/'MANIFEST.md').open('a',encoding='utf-8') as stream:
            stream.write('\n- '+stamp+' Mask-reference closed-state reconstruction and all256 recount tools SOURCE_ONLY reviewed; not executed.\n')
        stage += ['MANIFEST.md',*payloads]
    changed = subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--untracked-files=all']).decode()
    assert {line[3:] for line in changed.splitlines()}.issubset(set(stage))
    subprocess.check_call(['git','-C',str(repo),'add','-f','--',*stage],stderr=subprocess.DEVNULL)
    subprocess.check_call(['git','-C',str(repo),'-c','core.whitespace=cr-at-eol,-blank-at-eof','diff','--cached','--check','--',*stage])
    if index<2:
        assert all(subprocess.check_output(['git','-C',str(repo),'show',':'+name])==raw for name,raw in payloads.items())
    assert subprocess.check_output(['git','-C',str(repo),'show',':'+doc]).startswith(
        subprocess.check_output(['git','-C',str(repo),'show',previous['heads'][index]+':'+doc]))
    subprocess.check_call(['git','-C',str(repo),'commit','--quiet','-m','Record reviewed declared Mask-reference state reconstruction tools'])
    heads.append(subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip())
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'])
subprocess.check_call(['git','-C',str(repos[0]),'-c','http.version=HTTP/1.1','push','origin','HEAD:main'])
assert subprocess.check_output(['git','-C',str(repos[0]),'ls-remote','origin','refs/heads/main']).decode().split()[0]==heads[0]
digest = hashlib.sha256(new).hexdigest()
assert digest==remote['handoff_sha256'] and all(path.read_bytes()==new for path in copies)
guard = workspace/'.codex/tmp/sync_cs_handoff_remote_20260923.py'
raw = guard.read_bytes()
assert raw.count(previous['handoff_sha256'].encode())==1
guard.write_bytes(raw.replace(previous['handoff_sha256'].encode(),digest.encode()))
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(),section='20.376.84',heads=heads,
    handoff_sha256=digest,four_local_and_remote_equal=True,github_main=heads[0],payload_count=len(payloads),
    execution_scope='SOURCE_ONLY_TOOLS_PUBLISHED',candidate_restore_executed=False,new_accuracy_result=False,
    raw_npz_published=False,weights_changed=False,observer_native_session_id=27853)
receipt_path.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
state.update(time_cst=record['time_cst'],latest_publication=str(receipt_path),handoff_section=record['section'],
    handoff_sha256=digest,published_heads=heads,reference_restore_tools_source_published=True,
    reference_candidate_restore_executed=False,reference_restore_source_review=str(source_root/'RESTORE_SOURCE_REVIEW.json'))
state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
with (workspace/'memory/2026-10-06.md').open('a',encoding='utf-8') as stream:
    stream.write('\nPV-Ground '+record['time_cst']+': Doc84 reviewed candidate state reconstruction/all256 recount tools published; not executed. Main '+heads[0]+'. Actual observer27853 live handle checked11:33; no earlySSH progress check, first17:14:14 then240s. Active/unmet; trainedbest5616/4511 unchanged.\n')
print(json.dumps(record),flush=True)
