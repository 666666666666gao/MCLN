"""Publish the actual closed engineering result; reuse the existing Doc path."""
import ast
from pathlib import Path

root = Path(__file__).resolve().parent
template = (root / 'publish_start_authorized.py').read_text(encoding='utf-8')
source = template[:template.index("root=Path(__file__).resolve().parent")]
source = source.replace('Publish reviewed face-member source and actual M0 start, no accuracy claim.',
                        'Publish actual completed face-member M0; formal fit still gated.')
source += '''root=Path(__file__).resolve().parent
assert not (root/'completed_preflight_publication.json').exists()
review=json.loads((root/'PUBLISH_COMPLETE_SOURCE_REVIEW.json').read_bytes())
assert review['execution_scope']=='SOURCE_ONLY' and review['verdict'] in ('PASS','WARN') and not review['blocking_findings']
for item in review['reviewed_files']:
    assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256']
for name in ('SOURCE_REVIEW.json','FIT_SOURCE_REVIEW.json'):
    result=json.loads((root/name).read_bytes())
    assert result['execution_scope']=='SOURCE_ONLY' and result['verdict']=='PASS' and not result['blocking_findings']
    for item in result['reviewed_files']:
        assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256']
wait=json.loads((root/'preflight_wait.json').read_bytes())
proof=json.loads((root/'preflight_complete/preflight.json').read_bytes())
assert wait['observer_closed'] and not wait['controller_alive'] and wait['exitcode']==0
assert wait['status']['status']=='complete' and wait['status']['protected_parents_exact']
assert proof['status']=='pass' and proof['optimizer_steps_per_arm']==2
assert proof['frozen_parent_forward_calls']==2 and proof['weight_files_created']==0
assert proof['accuracy_result'] is False and proof['same_forward_native_score_mask_reference_exact']
assert proof['separate_optimizers_and_gradients']
for arm in ('face_center','face_region'):
    restored=proof['cpu_restore'][arm]
    assert restored['full_cpu_model_exact'] and restored['declared_sampler_restored']
    assert restored['full_state_tensors']==1304 and restored['deployed_geometry_heads']==1
    assert restored['optimizer']['all_keys_moments_steps_and_groups_exact']
    assert proof['witnesses'][0]['arms'][arm]['neutral_decode_equals_reference']
    assert all(value>0 for value in proof['witnesses'][1]['arms'][arm]['raw_parameter_gradient_norms'].values())
capacity=json.loads((root/'closed_fit_capacity.json').read_bytes())
assert capacity['preflight_closed'] and not capacity['formal_fit_started']
assert capacity['save_reserve_pass'] is False and not (root/'fit_launch.json').exists()
assert capacity['files_deleted']==0 and capacity['optimizer_or_inference_executed'] is False
previous_path=root/'start_publication.json'
previous=json.loads(previous_path.read_bytes());assert previous['section']=='20.376.93'
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
prefix='refine-logs/pvground_face_support_20261007/preflight_complete/'
section=f"""

## 20.376.94 六面观测共享前向真实两步预检闭合：恢复与梯度通过，尚无新正式精度（{stamp}）

承接§20.376.93的启动记录，新对照controller811380实际于{wait['status']['finished_cst']}结束，退出码0、耗时{wait['status']['elapsed_seconds']:.3f}秒。唯一观察器按计划08:48首次查看后收集实际终态，没有提前反复NN查询；native33836已闭合并消费。当前没有本实验仍在运行的NN/观察器，也没有启动完整fit。本节是工程终态，不能写成新增定位收益或有效论文模块。

真实PV/ScanRefer输入上，两组各更新2次，共享父模型完整前向2次；各456102参数/10状态，优化器与梯度独立。每步384个真实矩形投影距离的独立NumPy检查通过，中心KNN索引/距离精确相同，六面采样输出确有差异；索引位置差异包含排序，不当成相互独立点数或均匀覆盖证明。两组首步中性解码等于Mask空间参考；共享原生bbs/Mask/参考精确一致，无跨头梯度与更新，冻结父模型及零R状态保持。首步输出层有梯度、内部层为0符合零初始化链式求导；第二步各10参数的梯度范数都非零。

实际序列化/CPU重建了两个各自只包含一个声明sampler/几何头的完整模型，1304状态全部一致，Adam的10个状态/参数组/step2精确恢复；没有生成权重文件。峰值分配{proof['peak_allocated_bytes']}字节、保留{proof['peak_reserved_bytes']}字节，系冻结父模型/仅几何头预检测量，不能作为全模型联合训练显存。没有做恢复后新的完整GPU验证，也没有把这些两步输入当正式精度。

完整fit入口、定时观察器与闭合后归档已由同一新鲜评审跟进SOURCE_ONLY审查57实际文件PASS/0blocking（same-family/provisional，实际后端身份未认证）。拟保持§20.376.93的结构与预算，每组29778输入一次/3723更新，从保护5598/4848重新构建并重置优化器，绝不继续M0状态；共享一次冻结父模型前向、各自原生匹配及几何目标，部署仍一个几何头与一个bbs评分。

{capacity['time_cst']}实际只读核验：数据盘空闲{capacity['data_free_bytes']}字节、系统盘{capacity['system_free_bytes']}字节，原保存预算需{capacity['required_reserve_bytes']}字节，目前未通过。该核验没有运行NN、删除文件或尝试训练，容量门槛没有降低。另一已闭合目录的4756份NPZ/505731567字节已有完整本地副本及独立字节数/SHA核验，具体清理许可仍按本轮精确范围处理，不复用旧目录许可；本节没有执行NPZ删除。此前4份非最佳权重清理与最佳5598/4848保留仍是§20.376.92的实际结果，不重复执行。

现在所欠的是本轮清理许可/保存空间，继而真实3723更新与完整9508比较；并非还有一轮工程预检需要重跑。ScanRefer同一完整模型5620/4764及三个有效贡献门槛仍未满足，当前严格4848已达标、宽松5598还差22。Sr3D/Nr3D对应作者预训练权重已获用户允许，但各自完整训练仍在该门槛之后，baseline/完整方法同对应核心起点、独立训练、单seed2027。
"""
new=old+section.encode('utf-8');assert new.startswith(old) and new.count(b'## 20.376.94 ')==1
names=['preflight_wait.json','preflight_complete/preflight.json','preflight_complete/preflight_status.json',
    'preflight_complete/preflight_controller.exit','PREFLIGHT_RESULT.md','FIT_SOURCE_REVIEW.json','FIT_SOURCE_REVIEW.md',
    'launch_fit_authorized.py','observe_fit_authorized.py','collect_closed_fit_authorized.py',
    'prepare_fit_collection.py','closed_fit_capacity.json','check_fit_capacity_authorized.py',
    'prepare_completed_preflight_publication.py','publish_completed_preflight_authorized.py',
    'PUBLISH_COMPLETE_SOURCE_REVIEW.json','PUBLISH_COMPLETE_SOURCE_REVIEW.md']
payloads={prefix+name:(root/name).read_bytes() for name in names};payloads[prefix+'.gitattributes']=b'** -text\\n'
assert all(not name.endswith(('.pt','.pth','.npz')) and '.aris' not in name for name in payloads)
'''
tail = template[template.index('client=paramiko.SSHClient();client.load_system_host_keys()'):]
tail = tail.replace('/root/autodl-tmp/mcln_published_evidence_20261005/pvground_face_support_20261007/preflight_start',
                    '/root/autodl-tmp/mcln_published_evidence_20261005/pvground_face_support_20261007/preflight_complete')
tail = tail.replace('Face-member observation contrast: SOURCE reviewed and actual two-step M0 started; no new accuracy.',
                    'Face-member actual M0 completed: two independent heads and CPU restore passed; formal fit awaits capacity.')
tail = tail.replace('Prepare paired face observation and launch actual sanity',
                    'Record actual paired geometry sanity and full fit capacity gate')
tail = tail[:tail.index('record=dict(time_cst=')]+'''
record=dict(time_cst=datetime.datetime.now().astimezone().isoformat(),section='20.376.94',heads=heads,
    handoff_sha256=digest,four_local_and_remote_equal=True,github_main=heads[0],payload_count=len(payloads),
    execution_scope='ACTUAL_PREFLIGHT_CLOSED_PASS_FIT_CAPACITY_PENDING',new_formal_result=False,
    raw_npz_published=False,weights_published=False,preflight_controller_exitcode=0,formal_fit_started=False)
(root/'completed_preflight_publication.json').write_text(json.dumps(record,indent=2)+'\\n',encoding='utf-8')
state.update(time_cst=record['time_cst'],latest_publication=str(root/'completed_preflight_publication.json'),
    handoff_section=record['section'],handoff_sha256=digest,published_heads=heads,owned_gpu_job_active=False,
    overall_goal_complete=False,next_action='Actual M0 PASS and formal SOURCE57 PASS; await precise current4756NPZ permission and unchanged save capacity, then launch fresh paired fit. No active NN/observer. Scan5620/4764+3effective contributions before author-init Sr/Nr.')
state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\\n',encoding='utf-8')
with (workspace/'memory/2026-10-07.md').open('a',encoding='utf-8') as stream:
    stream.write('\\nPV-Ground '+record['time_cst']+': Doc94 actual M0 closed0/two updates each/one frozen parent/no crosshead grads/full1304+Adamrestore PASS; no newaccuracy/fullfit. MAIN'+heads[0]+' ONEdocSHA'+digest+' fourlocal+remote+Git exact. Actualcapacity491032576B<960209103B, newprecise4756NPZpermitpending, best5598/4848 retained. Author-init Nr/Sr after5620/4764+3effective contributions, no liveNN/observer.\\n')
print(json.dumps(record),flush=True)
'''
source += tail
ast.parse(source)
(root / 'publish_completed_preflight_authorized.py').write_text(source, encoding='utf-8')
print('ACTUAL_M0_COMPLETE_PUBLICATION_PREPARED_NOT_EXECUTED')
