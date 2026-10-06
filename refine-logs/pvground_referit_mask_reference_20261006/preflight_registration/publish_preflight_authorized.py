"""Append only registered ReferIt preflight preparation and actual launch to Doc86."""
import base64
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

import paramiko

root=Path(__file__).resolve().parent
assert not (root/'preflight_publication.json').exists()
review=json.loads((root/'PUBLISH_PREFLIGHT_SOURCE_REVIEW.json').read_bytes())
assert review['execution_scope']=='SOURCE_ONLY' and review['verdict'] in ('PASS','WARN') and not review['blocking_findings']
for item in review['reviewed_files']:
    assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256']
source_review=json.loads((root/'SOURCE_REVIEW.json').read_bytes())
assert source_review['execution_scope']=='SOURCE_ONLY' and source_review['verdict'] in ('PASS','WARN') and not source_review['blocking_findings']
for item in source_review['reviewed_files']:
    assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256']
launch=json.loads((root/'preflight_launch.json').read_bytes())
assert launch['actual_preflight_terminal_pending'] and launch['formal_rows']==0
cleanup=json.loads((root/'archived_arrays_delete_receipt.json').read_bytes())
assert cleanup['status']=='EXACT_LOCAL_ARCHIVED_REMOTE_NPZ_REMOVED' and cleanup['count']==4756
assert cleanup['logical_bytes']==571823151 and cleanup['best_weight_preserved']
previous_path=root.parent/'pvground_mask_reference_20261006/terminal_publication.json'
previous=json.loads(previous_path.read_bytes());assert previous['section']=='20.376.86'
state_path=root.parent/'pvground_query_supported_geometry_20261005/active_continuation_state.json'
state=json.loads(state_path.read_bytes());assert Path(state['latest_publication']).resolve()==previous_path.resolve()
workspace=Path('C:/Users/gb')
repos=[workspace/'.codex_mcln_g0_20260905',workspace/'.codex_pvground_cs_20261002',workspace/'.codex_mcln_v99_internal_20260928']
doc='docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
copies=[repo/doc for repo in repos]+[workspace/'Desktop/document'/Path(doc).name]
old=copies[0].read_bytes();assert hashlib.sha256(old).hexdigest()==previous['handoff_sha256']
assert all(path.read_bytes()==old for path in copies)
for repo,head in zip(repos,previous['heads']):
    assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip()==head
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'])
stamp=datetime.datetime.now().astimezone().isoformat()
prefix='refine-logs/pvground_referit_mask_reference_20261006/preflight_registration/'
section=f'''

## 20.376.87 Nr3D／Sr3D对应初始化与真实混合输入预检启动（{stamp}）

§86 ScanRefer best5598/4848保留，零更新与训练效果的结论不改。现在推进Nr3D/Sr3D对应作者初始化，不使用ScanRefer终点直接评估冒充独立训练。原作者Nr/Sr权重实际存在，各829830168字节且身份已核对；既有物理场景fit/holdout划分及原生实例框＋预测类别butd_cls输入复用。Nr fit36747/holdout6172（完整fit一次4594步），Sr fit65558/holdout10328（8195步）只是后续注册预算，正式训练尚未启动。

同一固定结构保留PV点—体素、文本Decoder、六源观测/任务读取、预测融合Mask成员范围空间参考、109维全局/7×16局部支撑、六面33节点分布及冻结全零R。一套last/bbs输出全部256候选，每候选一个最终框与自身Mask。各数据集严格加载对应作者1235状态，新增37读取/10几何/23零R重新初始化，无Scan权重或教师。核心和新读取/几何参数在预检中设为可训练，原生冻结参数及零R保持冻结；实际梯度和更新由M0结果核对。

训练修正适配真实接口：Nr CE token权重.6/.2/.2/.1，Sr .625/.125/.125/.125；语义与对比沿原生1/7系数。G标签替换及Query自身＋融合支撑确认的额外几何目标仅作用于sample_dataset为真实Nr/Sr表达行，检测行不扩展root目标，全部原匹配候选保护。language_dataset和sample_dataset职责分开；GT仅在训练资格/监督及评估中使用，模型不输入这些资格或dataset ID。

四项真实M0已由唯一控制器{launch['controller_pid']}于{launch['time_cst']}启动，顺序Nr-native、Nr-fused、Sr-native、Sr-fused，各4指代＋4检测增强输入、batch8、seed2027、两次更新；lr/backbone lr1e-5、WD.0005、clip.1。检查全256原始成员范围、native CE/梯度重建、检测/其他匹配保护、完整模型与Adam实际内存序列化恢复。预检不保存权重，更新不承接到正式训练。当前仅启动状态，不是M0通过或新精度；0正式验证结果。

fresh SOURCE_ONLY审查修正两处真实问题：GumbelSampling在eval仍执行gumbel_softmax，两次零初始化比较前必须重置同一RNG，不能比较不同随机Query；观测移植源码先安装成员观测包装后才能前向，原生对照暂时绕过新reader并保持非task分支，之后恢复同一reader。修正后源码审查0个未解决阻断，same-family/provisional、backend未attest；审查不能替代真实M0。

资源快照22:10:53：A10040GB、GPU仅1MiB，warm env SHA966235b2ead7fa5a1de63e9537745b457374a4e5749ac4a7a26566b7b630c82c不变，不安装新包。数据盘当时504991744字节、系统盘445059072字节。随后用户明确允许仅删除本地已核对的4756份旧远端NPZ：实际清理时间{cleanup['time_cst']}，逻辑571823151字节，数据盘清理后{cleanup['free_after']}字节；本地原始候选数组、全部文本记录及最佳权重保留。这只是旧远端副本清理，不改变实验结果；正式保存与评估空间仍需按实际配方检查。

M0四项初估20～40分钟；唯一观察器首次在启动后15分钟检查，根据已完成阶段实测耗时修正后续时间，接近结束再以180～300秒复查。正式同起点native/fused适配及完整7899/17726结果待M0和实际空间通过后启动。目标仍是三个基准两阈值超过明确baseline，整体未完成。源码、计划、身份和实际launch见{prefix}；无权重、原始NPZ、凭据或.aris发布。
'''
new=old+section.encode('utf-8');assert new.startswith(old) and new.count(b'## 20.376.87 ')==1
names=['EXPERIMENT_PLAN.md','EXPERIMENT_TRACKER.md','preflight_spec.json','prerequisites.json','preflight_launch.json',
    'SOURCE_REVIEW.json','SOURCE_REVIEW.md','referit_model_preflight.py','referit_training_targets.py',
    'pvground_referit_fit_dataset.py','mask_reference.py','query_supported_geometry.py','preflight_controller.py',
    'launch_preflight_authorized.py','publish_preflight_authorized.py','PUBLISH_PREFLIGHT_SOURCE_REVIEW.json',
    'PUBLISH_PREFLIGHT_SOURCE_REVIEW.md','archived_arrays_delete_receipt.json',
    'archived_arrays_delete_approval.json','ARRAY_CLEANUP_SOURCE_REVIEW.json','ARRAY_CLEANUP_SOURCE_REVIEW.md',
    'remove_approved_archived_arrays_authorized.py']
payloads={prefix+name:(root/name).read_bytes() for name in names};payloads[prefix+'.gitattributes']=b'** -text\n'
assert all(not name.endswith(('.pt','.pth','.npz')) and '.aris' not in name for name in payloads)
client=paramiko.SSHClient();client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
spec=json.loads((root/'preflight_spec.json').read_bytes())
code='''import base64,hashlib,json,sys
from pathlib import Path
project=Path(sys.argv[1]);b=json.load(sys.stdin);doc=project/b['doc'];old=doc.read_bytes()
assert hashlib.sha256(old).hexdigest()==b['old_sha256']
evidence=(project/b['prefix']).resolve()
assert str(evidence)=='/root/autodl-tmp/mcln_published_evidence_20261005/pvground_referit_mask_reference_20261006/preflight_registration'
assert not evidence.exists()
for name,encoded in b['files'].items():
    assert name.startswith(b['prefix']) and not name.endswith(('.pt','.pth','.npz')) and '.aris' not in name
    p=project/name;assert evidence in p.resolve().parents;p.parent.mkdir(parents=True,exist_ok=True)
    raw=base64.b64decode(encoded)
    with p.open('xb') as f:f.write(raw)
    assert p.read_bytes()==raw
new=base64.b64decode(b['new_doc']);assert new.startswith(old);doc.write_bytes(new)
assert doc.read_bytes()==new
print(json.dumps(dict(files=len(b['files']),handoff_sha256=hashlib.sha256(new).hexdigest())))
'''
bundle=dict(doc=doc,prefix=prefix,old_sha256=previous['handoff_sha256'],new_doc=base64.b64encode(new).decode(),
    files={name:base64.b64encode(raw).decode() for name,raw in payloads.items()})
stdin,stdout,stderr=client.exec_command(shlex.join([spec['runtime']+'/venv/bin/python','-B','-c',code,
    '/home/gb/new butd/butd_detr-main/MCLN-main']),timeout=180)
stdin.write(json.dumps(bundle).encode());stdin.flush();stdin.channel.shutdown_write()
raw=stdout.read();assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
remote=json.loads(raw);client.close()
for name,raw in payloads.items():
    for repo in repos[:2]:
        path=repo/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
for path in copies:path.write_bytes(new)
heads=[]
for index,repo in enumerate(repos):
    stage=[doc]
    if index<2:
        with (repo/'MANIFEST.md').open('a',encoding='utf-8') as stream:
            stream.write('\n- '+stamp+' Corresponding Nr/Sr initialization, native training roles and four real mixed-row preflights launched; no new formal metrics.\n')
        stage+=['MANIFEST.md',*payloads]
    changed=subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--untracked-files=all']).decode()
    assert {line[3:] for line in changed.splitlines()}.issubset(set(stage))
    subprocess.check_call(['git','-C',str(repo),'add','-f','--',*stage],stderr=subprocess.DEVNULL)
    subprocess.check_call(['git','-C',str(repo),'-c','core.whitespace=cr-at-eol,-blank-at-eof','diff','--cached','--check','--',*stage])
    if index<2:
        assert all(subprocess.check_output(['git','-C',str(repo),'show',':'+name])==raw for name,raw in payloads.items())
    assert subprocess.check_output(['git','-C',str(repo),'show',':'+doc]).startswith(
        subprocess.check_output(['git','-C',str(repo),'show',previous['heads'][index]+':'+doc]))
    subprocess.check_call(['git','-C',str(repo),'commit','--quiet','-m','Register corresponding Nr3D Sr3D Mask-reference real preflight'])
    heads.append(subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip())
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'])
subprocess.check_call(['git','-C',str(repos[0]),'-c','http.version=HTTP/1.1','push','origin','HEAD:main'])
assert subprocess.check_output(['git','-C',str(repos[0]),'ls-remote','origin','refs/heads/main']).decode().split()[0]==heads[0]
digest=hashlib.sha256(new).hexdigest();assert digest==remote['handoff_sha256'] and all(p.read_bytes()==new for p in copies)
guard=workspace/'.codex/tmp/sync_cs_handoff_remote_20260923.py';raw=guard.read_bytes()
assert raw.count(previous['handoff_sha256'].encode())==1
guard.write_bytes(raw.replace(previous['handoff_sha256'].encode(),digest.encode()))
record=dict(time_cst=datetime.datetime.now().astimezone().isoformat(),section='20.376.87',heads=heads,
    handoff_sha256=digest,four_local_and_remote_equal=True,github_main=heads[0],payload_count=len(payloads),
    execution_scope='REAL_PREFLIGHT_LAUNCH_REGISTERED_NOT_TERMINAL',new_formal_result=False,
    controller_pid=launch['controller_pid'],raw_npz_published=False,weights_published=False)
(root/'preflight_publication.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
state.update(time_cst=record['time_cst'],latest_publication=str(root/'preflight_publication.json'),
    handoff_section=record['section'],handoff_sha256=digest,published_heads=heads,
    referit_preflight_launch=str(root/'preflight_launch.json'),referit_preflight_controller_pid=launch['controller_pid'],
    owned_gpu_job_active=True,overall_goal_complete=False,next_action='Wait until scheduled ReferIt preflight observation; no formal training launched')
state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
with (workspace/'memory/2026-10-06.md').open('a',encoding='utf-8') as stream:
    stream.write('\nPV-Ground '+record['time_cst']+': Doc87 corresponding Nr/Sr official1235 and fresh70 modules four actualmixed8x2 M0 launched controller'+str(launch['controller_pid'])+'; G weights/detection protection adapted, Gumbel paired RNG bug fixed before launch. No newformal metrics orScanweight transfer. Main '+heads[0]+'. Userapproved4756 oldremoteNPZ actuallyretired571823151B after localSHA; alllocalarrays/text/bestweights retained. Formalstorage capacity still needs actualrecipe check.\n')
print(json.dumps(record),flush=True)
