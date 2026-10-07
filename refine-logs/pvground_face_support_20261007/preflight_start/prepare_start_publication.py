"""Reuse the existing append-only publication path for actual M0 start only."""
import ast
from pathlib import Path

root = Path(__file__).resolve().parent
old = root.parent / 'pvground_reference_keep_20261006'
template = (old / 'publish_terminal_authorized.py').read_text(encoding='utf-8')
imports = template[template.index('import base64'):template.index('root=Path(__file__).resolve().parent')]
prefix = '''"""Publish reviewed face-member source and actual M0 start, no accuracy claim."""
'''+imports+'''
root=Path(__file__).resolve().parent
assert not (root/'start_publication.json').exists()
review=json.loads((root/'PUBLISH_START_SOURCE_REVIEW.json').read_bytes())
assert review['execution_scope']=='SOURCE_ONLY' and review['verdict'] in ('PASS','WARN') and not review['blocking_findings']
for item in review['reviewed_files']:
    assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256']
source_review=json.loads((root/'SOURCE_REVIEW.json').read_bytes())
assert source_review['execution_scope']=='SOURCE_ONLY' and source_review['verdict']=='PASS'
assert not source_review['blocking_findings']
for item in source_review['reviewed_files']:
    assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256']
launch=json.loads((root/'preflight_launch.json').read_bytes())
observer=json.loads((root/'observer_wait.json').read_bytes())
assert launch['status']=='PREFLIGHT_LAUNCHED_NOT_PASSED' and launch['accuracy_result'] is False
assert launch['controller_pid']==observer['controller_pid']==811380
assert launch['root']=='/root/autodl-tmp/pvground_face_support_20261007'
assert launch['first_check_seconds']==660 and launch['later_poll_seconds']==240
assert not (root/'fit_launch.json').exists()
previous_path=root.parent/'pvground_reference_keep_20261006/terminal_publication.json'
previous=json.loads(previous_path.read_bytes());assert previous['section']=='20.376.92'
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
prefix='refine-logs/pvground_face_support_20261007/preflight_start/'
section=f"""

## 20.376.93 六面边界观测对照源码与真实预检启动：尚无新精度（{stamp}）

上一项参考保持已完整结束，control/keep两组正式5593/4832，均低于保留的5598/4848；本节不改写该终态。下一项已实现仅改变边界成员选择的直接对照：face_center保留原中心与六面中心各16近邻；face_region中心近邻不变，六面各读取距离有界面矩形最近的16个实际输入成员。法向距离与超出矩形的切向距离决定选择；编码仍保留原面中心欧氏距离、XYZ/RGB/Mask通道、14维成员MLP、109维范围输入及原456102参数/10状态/6×33分布头。没有新参数、半径门控、GT采样、裁剪候选、质量排名、文本读取或教师。最近16成员不保证整面均匀覆盖、固定薄层或前景背景平衡；实际性能仍须正式实验回答。

两组每个batch共享一次实际冻结PV/原G父模型前向，获得相同Query、原粗框、Text/Query Mask、融合权重及唯一原生bbs；两个几何头和AdamW独立更新。源码核对原生criterion后，明确detach冻结零输出R产生的公共last_sem_cls_scores，避免第二组loss经过第一组已使用的几何图。第二组覆盖自身几何字段；两组原生匹配各自计算，保存后只重建一个声明的sampler及一个几何头，推理不保留两个实验答案。

新鲜SOURCE_ONLY审查PASS/49实际文件/0blocking，same-family/provisional，实际模型后端身份未认证；不是GPU结果。真实A100两步预检实际于{launch['time_cst']}启动，controller{launch['controller_pid']}，screen{launch['screen']}；两组各计划2次更新。检查真实点的独立矩形距离、中心成员精确一致、中性输出等于参考、相同分数/Mask/参考、独立优化器与梯度、父模型/零R保持，以及各sampler的完整CPU模型/Adam恢复。预检不落地权重、不承接预检状态启动正式fit。

仅一个观察器，WindowsPID{observer['observer_local_pid']}，首次计划{observer['first_observation_cst']}，之后240秒；不提前重复NN/日志查询。启动核验数据盘空闲{launch['resource']['data_free_bytes']}字节、系统盘{launch['resource']['system_free_bytes']}字节；预检仅需小量文本输出。完整fit尚未启动，仍要求真实预检通过及原保存预算满足，不降低容量门槛。当前另一已闭合目录4756NPZ/505731567字节已经完整本地归档，精确删除许可问题尚未获得新回复；旧目录许可不能套用。本节没有执行候选数组删除。

正式计划仍为每组29778fit输入一次/3723更新、batch8/累积1、LR1e-5、WD5e-4、clip0.1、seed2027；native＋G＋matched DFL/7＋Query自身支撑额外几何为共同目标，失败参考保持权重为0。9508初始/终点formal共用一份公共GT/score/prior/reference和各自final的NPZ；6887已见训练场景留出保留逐行与Full-256覆盖摘要，不扩充原始数组。仅预检启动，尚无新fit或正式指标，不能计为有效M2。

用户目标保持：同一个完整ScanRefer模型Acc@0.25>59.1且Acc@0.5>50.1（5620/4764），并有三个真实有效贡献/直接消融，再固定完整模型训练Sr3D和Nr3D。用户明确允许对应作者预训练权重：各数据集baseline/完整方法同对应核心初始化，独立训练、单seed2027；不把Scan权重直测称为各自训练结果。当前最好5598/4848与必要PV/原G/V99依赖保留，宽松还差22条，三个贡献尚未成立。
"""
new=old+section.encode('utf-8');assert new.startswith(old) and new.count(b'## 20.376.93 ')==1
names=['face_region_box_refiner.py','face_support_model_factory.py','paired_geometry_loop.py',
    'run_face_support_pair.py','selected_mask_reference_factory.py','mask_reference.py',
    'query_supported_geometry.py','pair_spec.json','controller.py','EXPERIMENT_PLAN.md',
    'SOURCE_REVIEW.json','SOURCE_REVIEW.md','deploy_preflight_authorized.py',
    'observe_preflight_authorized.py','preflight_launch.json','prepare_start_publication.py',
    'publish_start_authorized.py','PUBLISH_START_SOURCE_REVIEW.json','PUBLISH_START_SOURCE_REVIEW.md']
payloads={prefix+name:(root/name).read_bytes() for name in names};payloads[prefix+'.gitattributes']=b'** -text\\n'
assert all(not name.endswith(('.pt','.pth','.npz')) and '.aris' not in name for name in payloads)
'''
tail = template[template.index('client=paramiko.SSHClient();client.load_system_host_keys()'):]
tail = tail.replace("spec=json.loads((root/'control_spec.json').read_bytes())", "spec=json.loads((root/'pair_spec.json').read_bytes())")
tail = tail.replace('/root/autodl-tmp/mcln_published_evidence_20261005/pvground_reference_keep_20261006/completed_comparison',
                    '/root/autodl-tmp/mcln_published_evidence_20261005/pvground_face_support_20261007/preflight_start')
tail = tail.replace('Closed reference-keep comparison: no metric gain; original best retained and closed nonbest weights cleared.',
                    'Face-member observation contrast: SOURCE reviewed and actual two-step M0 started; no new accuracy.')
tail = tail.replace('Record completed reference-preservation comparison', 'Prepare paired face observation and launch actual sanity')
start = tail.index("record=dict(time_cst=")
tail = tail[:start]+'''
record=dict(time_cst=datetime.datetime.now().astimezone().isoformat(),section='20.376.93',heads=heads,
    handoff_sha256=digest,four_local_and_remote_equal=True,github_main=heads[0],payload_count=len(payloads),
    execution_scope='SOURCE_REVIEWED_ACTUAL_PREFLIGHT_STARTED',new_formal_result=False,
    controller_pid=launch['controller_pid'],raw_npz_published=False,weights_published=False)
(root/'start_publication.json').write_text(json.dumps(record,indent=2)+'\\n',encoding='utf-8')
state.update(time_cst=record['time_cst'],latest_publication=str(root/'start_publication.json'),
    handoff_section=record['section'],handoff_sha256=digest,published_heads=heads,
    face_support_preflight_controller_pid=launch['controller_pid'],face_support_observer_pid=observer['observer_local_pid'],
    face_support_first_observation_cst=observer['first_observation_cst'],owned_gpu_job_active=True,
    overall_goal_complete=False,next_action='Wait sole M0 observer at08:48:09 CST, then240s. Formal fit only after actual M0 pass and save reserve; NPZ permission pending. Scan5620/4764+3 effective contributions before author-init Sr/Nr.')
state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\\n',encoding='utf-8')
with (workspace/'memory/2026-10-07.md').open('a',encoding='utf-8') as stream:
    stream.write('\\nPV-Ground '+record['time_cst']+': Doc93 actual face-support SOURCE49 PASS +M0 launched811380/sole39744 first08:48:09; no newformal result/fit. MAIN'+heads[0]+' ONEdocSHA'+digest+' fourlocal+remote+Git exact. Best5598/4848 protected; corresponding-author Nr/Sr after5620/4764+3effective modules. Precise current NPZ approval pending.\\n')
print(json.dumps(record),flush=True)
'''
source = prefix + tail
ast.parse(source)
(root / 'publish_start_authorized.py').write_text(source, encoding='utf-8')
print('DOC93_ACTUAL_M0_START_PUBLISHER_PREPARED_NOT_EXECUTED')
