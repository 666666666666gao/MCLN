"""Publish reviewed sources, accepted M0 and actual read-only formal launch."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import base64
import paramiko

local=Path(__file__).resolve().parent
workspace=Path('C:/Users/gb')
assert not (local/'terminal_publication.json').exists()
audit=json.loads((local/'analysis/EXPERIMENT_AUDIT.json').read_bytes())
acceptance=json.loads((local/'analysis/TERMINAL_ACCEPTANCE.json').read_bytes())
summary=json.loads((local/'analysis/SUMMARY.json').read_bytes())
publication_review=json.loads((local/'PUBLISH_TERMINAL_REVIEW.json').read_bytes())
assert audit['verdict'] in ('PASS','WARN') and not audit['blocking_findings']
assert audit['execution_scope']=='TERMINAL_SOURCE_AND_ACTUAL_ARTIFACTS'
assert acceptance['offline_not_new_trained_result'] and acceptance['trained_best_hits']==[5616,4511]
assert publication_review['execution_scope']=='SOURCE_ONLY'
assert publication_review['verdict'] in ('PASS','WARN') and not publication_review['blocking_findings']
for item in publication_review['reviewed_files']:
    assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256'],item['path']
state_path=local.parent/'pvground_query_supported_geometry_20261005/active_continuation_state.json'
state=json.loads(state_path.read_bytes())
previous=json.loads(Path(state['latest_publication']).read_bytes())
assert previous['section']=='20.376.80'
repos=[workspace/'.codex_mcln_g0_20260905',workspace/'.codex_pvground_cs_20261002',workspace/'.codex_mcln_v99_internal_20260928']
doc='docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
copies=[repo/doc for repo in repos]+[workspace/'Desktop/document'/Path(doc).name]
old=copies[0].read_bytes()
assert hashlib.sha256(old).hexdigest()==previous['handoff_sha256'] and all(path.read_bytes()==old for path in copies)
for repo,head in zip(repos,previous['heads']):
    assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip()==head
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'])
stamp=datetime.datetime.now().astimezone().isoformat()
prefix='refine-logs/pvground_mask_extent_diagnostic_20261006/terminal/'
section=f'''

## 20.376.81 当前4511模型完整Mask范围诊断闭合：同Query exact4848，未当作训练成绩（{stamp}）

承接§80。实际9508只读任务于{summary['actual_controller']['finished_cst']}闭合，controller699895、child699896、exit0；sole observer59532和CPU分组34764均已闭合，不重启旧进程。重建同一官方PV＋原G＋受保护4511几何头，eval/no_grad；0优化器、0更新、0新权重。完整256候选保留，原生bbs选中的同一Query，原语义头每批仅调用一次，Box与Mask身份一致。GT只在前向及选择之后用于IoU/Mask诊断。

| 同Query范围 | Acc@0.25命中 | Acc@0.50命中 | 严格修复/破坏/净变化 |
|---|---:|---:|---:|
| 当前学习框 |5616|4511|—|
| 全融合Mask sigmoid>0.5实际成员精确范围 |5598|4848|713/376/+337|
| 同一前景各轴0.5%/99.5%线性分位范围 |5590|4781|663/393/+270|

exact@0.25修复174/破坏192/净−18；q005修复167/破坏193/净−26。两种Mask范围均39条无效，实际均为空支撑，诊断计IoU0，没有用原框补齐。原历史9508逐行身份、点hash、rootGT、选中Query和严格标签一致；505行IoU有微小浮点差异、最大约8.106e−6，不能写成逐位复现。

自身Query与融合Mask都>0.5的5124条中，原严格4169→exact4577，修复644/破坏236/净408；其中原框不合格955条，exact修复644。公共融合好、自身Query不好仅9条，严格5→5；两Mask都不合格4366条，exact严格328→258，净−70。此为当前模型离线GT分组，资格不进入部署；不能把GT分组变成推理门控，也不能把644条换算成新网络预期收益。目标体积四分位下exact严格净变化170/93/10/64，分组阈值/并列处理保留在实际CPU报告。

实际推理1755.08秒、总1882.98秒；峰值allocator4321393664B、reserved5739905024B。1189份NPZ共450284771B，完整收集1197文件462713589B，权重0。独立fresh-context terminal SOURCE_AND_ACTUAL审查{audit['verdict']}、0阻断，全部1197 SHA/bytes、9508成员极值/Mask交并/分位邻节点/IoU/修复破坏及分组均重算，阈值翻转0。完整原始50000点排序重放仅M0的8条；正式9508核验保存的成员证据，并非全原始点复放。审查属于same-family/provisional、backend未attest。单seed与反复开发验证的范围限制保留。数据和NPZ留本地/服务器，不上传Git，仅源与摘要、实际收集清单和审查记录。

SUMMARY、NARRATIVE_REPORT与TERMINAL_EXECUTION_TRACKER保留审查前的pending/in-progress快照以保持被审原字节；最终审查状态以TERMINAL_ACCEPTANCE.json和已返回的EXPERIMENT_AUDIT.json/md为准，不将旧启动快照写成服务器当前状态。

结论：完整预测Mask中存在现有学习框未充分利用的严格几何信息，但376条破坏不支持无条件换框。本次不是新训练模型、不是V99能力已内化，metricbest仍5616/4511，距4754尚差243条。下一项源码正在fresh SOURCE_ONLY审查，计划以完整Mask范围作几何参考，对照原粗框参考；两组共同保留hidden8并重置output2，原粗框保留6维先验，参考改变采样位置及六面坐标系，监督预算保持。39条实际空支撑说明网络无法形成参考时需要保留原粗框先验；这是有效性处理，不是GT质量/双源/版本选择。该新实验尚未部署或获得精度，先真实两步GPU预检，再固定预算训练。研究目标ACTIVE_UNMET，PV-Ground主线、原生唯一bbs及全256候选不变。
'''
assert chr(65533) not in section
new=old+section.encode('utf-8')
assert new.startswith(old) and new.count(b'## 20.376.81 ')==1
names=['publish_terminal.py','prepare_terminal_publication.py','record_terminal_acceptance.py',
       'TERMINAL_EXECUTION_TRACKER.md','FORMAL_CLOSED_SESSIONS.json','PUBLICATION_EXECUTION_SESSIONS.json',
       'FORMAL_ESTIMATE_EXPIRED_PROGRESS.json','formal_wait.json',
       'analysis/SUMMARY.json','analysis/NARRATIVE_REPORT.md','analysis/TERMINAL_ACCEPTANCE.json',
       'analysis/TERMINAL_REVIEW_CALL.json','analysis/EXPERIMENT_AUDIT.json','analysis/EXPERIMENT_AUDIT.md',
       'complete/formal_status.json','complete/formal.exit','complete/formal/receipt.json',
       'complete/formal/CPU_SUMMARY.json','complete/formal/FAILURE_BREAKDOWN.json',
       'complete/formal/imports.json','complete/formal/load.json',
       'PUBLISH_TERMINAL_REVIEW.json','PUBLISH_TERMINAL_REVIEW.md']
assert len(names)==len(set(names))
assert all('.aris' not in name and not name.endswith(('.pth','.pt','.npz')) for name in names)
payloads={prefix+name:(local/name).read_bytes() for name in names}
payloads[prefix+'.gitattributes']=b'** -text\n'
client=paramiko.SSHClient();client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
project='/home/gb/new butd/butd_detr-main/MCLN-main'
code='''import base64,hashlib,json,sys
from pathlib import Path
project=Path(sys.argv[1]);bundle=json.load(sys.stdin)
doc=project/bundle['doc'];old=doc.read_bytes()
assert hashlib.sha256(old).hexdigest()==bundle['old_sha256']
evidence=(project/bundle['prefix']).resolve()
assert str(evidence)=='/root/autodl-tmp/mcln_published_evidence_20261005/pvground_mask_extent_diagnostic_20261006/terminal'
assert not evidence.exists()
for name,encoded in bundle['files'].items():
    assert name.startswith(bundle['prefix']) and '.aris' not in name and not name.endswith(('.pt','.pth','.npz'))
    path=project/name;assert evidence in path.resolve().parents
    path.parent.mkdir(parents=True,exist_ok=True)
    raw=base64.b64decode(encoded)
    with path.open('xb') as stream:stream.write(raw)
    assert path.read_bytes()==raw
new=base64.b64decode(bundle['new_doc']);assert new.startswith(old)
doc.write_bytes(new);assert doc.read_bytes()==new
print(json.dumps(dict(files=len(bundle['files']),handoff_sha256=hashlib.sha256(new).hexdigest())))
'''
bundle=dict(doc=doc,prefix=prefix,old_sha256=previous['handoff_sha256'],
            files={name:base64.b64encode(raw).decode() for name,raw in payloads.items()},new_doc=base64.b64encode(new).decode())
runtime=json.loads((local/'spec.json').read_bytes())['runtime']
stdin,stdout,stderr=client.exec_command(shlex.join([runtime+'/venv/bin/python','-B','-c',code,project]),timeout=180)
stdin.write(json.dumps(bundle).encode());stdin.flush();stdin.channel.shutdown_write()
remote_raw=stdout.read();remote_exit=stdout.channel.recv_exit_status();remote_error=stderr.read().decode()
assert remote_exit==0,remote_error
remote=json.loads(remote_raw)
client.close()
for name,raw in payloads.items():
    for repo in repos[:2]:
        destination=repo/name;destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(raw)
for path in copies:path.write_bytes(new)
heads=[]
for index,repo in enumerate(repos):
    stage=[doc]
    if index<2:
        with (repo/'MANIFEST.md').open('a',encoding='utf-8') as stream:
            stream.write('\n- '+stamp+' Protected4511 fullMask diagnostic9508 closed: offline exact4848/q4781, no trainedbest promotion; fresh terminal WARN0blocks.\n')
        stage += ['MANIFEST.md',*payloads]
    changed=subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--untracked-files=all']).decode()
    assert {line[3:] for line in changed.splitlines()}.issubset(set(stage))
    subprocess.check_call(['git','-C',str(repo),'add','-f','--',*stage],stderr=subprocess.DEVNULL)
    reports=[name for name in payloads if Path(name).name in ('SOURCE_REVIEW.json','SOURCE_REVIEW.md','FORMAL_LAUNCH_REVIEW.json','FORMAL_LAUNCH_REVIEW.md')] if index<2 else []
    subprocess.check_call(['git','-C',str(repo),'-c','core.whitespace=cr-at-eol','diff','--cached','--check','--',*stage,*[':(exclude)'+name for name in reports]])
    subprocess.check_call(['git','-C',str(repo),'-c','core.whitespace=cr-at-eol,-blank-at-eof','diff','--cached','--check','--',*stage])
    if index<2:
        assert all(subprocess.check_output(['git','-C',str(repo),'show',':'+name])==raw for name,raw in payloads.items())
    assert subprocess.check_output(['git','-C',str(repo),'show',':'+doc]).startswith(subprocess.check_output(['git','-C',str(repo),'show',previous['heads'][index]+':'+doc]))
    subprocess.check_call(['git','-C',str(repo),'commit','--quiet','-m','Record closed9508 Mask extent diagnostic and fresh integrity audit'])
    heads.append(subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip())
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'])
subprocess.check_call(['git','-C',str(repos[0]),'-c','http.version=HTTP/1.1','push','origin','HEAD:main'])
assert subprocess.check_output(['git','-C',str(repos[0]),'ls-remote','origin','refs/heads/main']).decode().split()[0]==heads[0]
digest=hashlib.sha256(new).hexdigest();assert digest==remote['handoff_sha256']
guard=workspace/'.codex/tmp/sync_cs_handoff_remote_20260923.py';raw=guard.read_bytes()
assert raw.count(previous['handoff_sha256'].encode())==1
guard.write_bytes(raw.replace(previous['handoff_sha256'].encode(),digest.encode()))
record=dict(time_cst=datetime.datetime.now().astimezone().isoformat(),section='20.376.81',heads=heads,
            handoff_sha256=digest,four_local_and_remote_equal=True,github_main=heads[0],
            new_trained_accuracy_result=False,offline_extent_diagnostic_complete=True,
            protected_trained_model_hits=[5616,4511],offline_exact_hits=[5598,4848],offline_quantile_hits=[5590,4781],
            terminal_verdict=audit['verdict'],payload_count=len(payloads),raw_npz_published=False)
(local/'terminal_publication.json').write_text(json.dumps(record,indent=2)+'\n')
state.update(time_cst=record['time_cst'],latest_publication=str(local/'terminal_publication.json'),
             status='MASK_EXTENT_CLOSED_ACCEPTED_PUBLISHED_REFERENCE_SOURCE_REVIEW',owned_gpu_job_active=False,
             active_reviewer='/root/pvg_mask_reference_source_review',
             handoff_section=record['section'],handoff_sha256=digest,published_heads=heads,
             next_action='Complete new reference SOURCE review; actual2-update checks before training. Best4511 unchanged.')
state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
with (workspace/'memory/2026-10-06.md').open('a',encoding='utf-8') as stream:
    stream.write('\nPVGround '+record['time_cst']+': Doc81 actual9508 extent diagnostic closed and fresh audit accepted; offline4848/q4781 not trainedbest; source reference pair pending. Fourlocal/remote exact, main '+heads[0]+'. No training/new weights, protected5616/4511 unchanged.\n')
print(json.dumps(record),flush=True)
