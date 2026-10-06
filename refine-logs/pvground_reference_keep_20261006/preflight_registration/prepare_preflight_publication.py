"""Append the actual new loss source and M0 launch to the existing Doc88."""
from pathlib import Path

root=Path(__file__).resolve().parent
previous=root.parent/'pvground_referit_mask_reference_20261006'
source=(previous/'publish_preflight_authorized.py').read_text(encoding='utf-8')
source=source.replace('registered ReferIt preflight preparation and actual launch to Doc86',
    'reference-preservation source and actual ScanRefer M0 launch to Doc88')
source=source.replace('PUBLISH_PREFLIGHT_SOURCE_REVIEW','PUBLISH_PREFLIGHT_SOURCE_REVIEW')
begin=source.index("launch=json.loads((root/'preflight_launch.json').read_bytes())")
end=source.index("workspace=Path('C:/Users/gb')",begin)
source=source[:begin]+'''launch=json.loads((root/'preflight_launch.json').read_bytes())
assert launch['accuracy_result'] is False and launch['optimizer_steps_planned_per_arm']==2
assert launch['root']=='/root/autodl-tmp/pvground_reference_keep_20261006'
previous_path=root.parent/'pvground_referit_mask_reference_20261006/comparison_publication.json'
previous=json.loads(previous_path.read_bytes());assert previous['section']=='20.376.88'
state_path=root.parent/'pvground_query_supported_geometry_20261005/active_continuation_state.json'
state=json.loads(state_path.read_bytes());assert Path(state['latest_publication']).resolve()==previous_path.resolve()
''' + source[end:]
source=source.replace('pvground_referit_mask_reference_20261006/preflight_registration/',
    'pvground_reference_keep_20261006/preflight_registration/')
begin=source.index("section=f'''")
end=source.index('client=paramiko.SSHClient()',begin)
source=source[:begin]+'''section=f\'\'\'

## 20.376.89 ScanRefer参考保持训练约束：源码通过与真实两组预检启动（{stamp}）

§88新用户目标不变：同一完整9508模型至少5620/4764，三个经直接消融支持的有效模块后再固定结构、从对应作者预训练权重独立训练Sr/Nr；single seed2027。当前保护5598/4848零更新Mask参考模型，不把预检或零输出分支算有效学习贡献。Nr/Sr正式训练未启动。

当前唯一新增方向是训练期参考相对退化约束，不增加推理参数、排序源或Mask/文本更新。两组严格从保留initial.pth完整10项几何状态＋官方PV＋原G重建，复用实际1304状态CPU重建证明，无需已清理的4511权重。相同456102几何参数、融合Mask空间参考、全局109维/局部7×16支撑、六面33节点分布，父模型/Mask/语言/原生语义/全零R冻结；全部256候选，一套last/bbs、同QueryBoxMask。原native/G/匹配DFL及既有额外几何项不改。

控制λ=0、策略λ=1；可信集合内计算squared ReLU(stopgrad(IoU(reference,GT))-IoU(final,GT))，先表达内平均再实际batch平均。原匹配查询对应其原生过滤后真实GT；额外查询须自身Query和融合Mask对root均IoU>.5，排除全部原匹配，不以Box是否刚过.5截断保持责任。资格只用于训练，不输入模型/推理。该项允许有益修正，惩罚相对参考的几何退化；目前只是待验证训练机制，不保证精度或参考一定不被破坏。

fresh SOURCE_ONLY审查已完成、0剩余阻断，same-family/provisional、backend与runtime未认证。发现并最小修正真实CPU/CUDA接口错误：原HungarianMatcher返回CPU int64索引，新增Mask资格索引在CUDA，直接cat会失败；只在新增函数内显式将matched_queries/targets移至boxes.device，原匹配/旧loss不变。原finding和修正保留。

真实M0两组已于{launch['time_cst']}由唯一控制器{launch['controller_pid']}启动，每组真实增强batch8两步更新、seed2027/lr1e-5/WD.0005/clip.1，更新不承接正式训练，不保存预检权重。待核对初始中性解码/保持损失严格0、实际新项梯度资格范围、nativebbs/Mask与父状态、几何梯度和Adam内存CPU重载。当前仅launch证据，不写预检通过或新精度。

启动资源：GPU实际空闲、保护4848权重SHA核对；数据盘{launch['resource']['data_free_bytes']}字节，系统盘{launch['resource']['system_free_bytes']}字节。复用既有A10040GB/Torch1.10.2环境，不安装新包。上一同构M0单组470秒，本轮两组约16～20分钟；首次启动720秒后查看、后续240秒或已完成阶段实测估计，不提前查询或另起观察器。后续完整pair约7小时，正式前另核对真实保存/评估空间。

通过M0才拟进行两组同起点/同预算29778条fit一次、3723更新（B8/累积1）与初始/终点6887模块留出及完整9508原生验证。模块留出场景被作者预训练见过；只超过历史4832终点不足以证明新增学习，要同时比本轮控制与4848起点看修复/破坏，最终核对5620/4764。当前0新正式结果、三模块证据未成立，不调seed、不同时加边界新采样/attention/教师/质量头。源码和方案见{prefix}；权重/原始NPZ/凭据/.aris不发布。
\'\'\'
new=old+section.encode('utf-8');assert new.startswith(old) and new.count(b'## 20.376.89 ')==1
names=['EXPERIMENT_PLAN.md','EXPERIMENT_TRACKER.md','research_contract.md','control_spec.json','keep_spec.json',
    'preflight_launch.json','resource_check.json','SOURCE_REVIEW.json','SOURCE_REVIEW.md',
    'reference_keep.py','run_reference_keep_fit.py','selected_mask_reference_factory.py',
    'query_supported_geometry.py','mask_reference.py','check_invalid_reference.py','invalid_reference_fixture.json',
    'controller.py','deploy_preflight_authorized.py','observe_preflight_authorized.py','prepare_reference_keep.py',
    'prepare_preflight_publication.py','publish_preflight_authorized.py',
    'PUBLISH_PREFLIGHT_SOURCE_REVIEW.json','PUBLISH_PREFLIGHT_SOURCE_REVIEW.md']
payloads={prefix+name:(root/name).read_bytes() for name in names};payloads[prefix+'.gitattributes']=b'** -text\\n'
assert all(not name.endswith(('.pt','.pth','.npz')) and '.aris' not in name for name in payloads)
''' + source[end:]
source=source.replace("spec=json.loads((root/'preflight_spec.json').read_bytes())", "spec=json.loads((root/'control_spec.json').read_bytes())")
source=source.replace('pvground_referit_mask_reference_20261006/preflight_registration',
    'pvground_reference_keep_20261006/preflight_registration')
source=source.replace('Corresponding Nr/Sr initialization, native training roles and four real mixed-row preflights launched; no new formal metrics.',
    'Reference-preservation-only ScanRefer loss contrast actual M0 launched; no new formal metrics.')
source=source.replace('Register corresponding Nr3D Sr3D Mask-reference real preflight',
    'Register ScanRefer reference-preservation loss and real M0 launch')
source=source.replace("section='20.376.87'", "section='20.376.89'")
source=source.replace("execution_scope='REAL_PREFLIGHT_LAUNCH_REGISTERED_NOT_TERMINAL'", "execution_scope='REFERENCE_KEEP_M0_LAUNCH_NOT_PASSED'")
source=source.replace("referit_preflight_launch=str(root/'preflight_launch.json'),referit_preflight_controller_pid=launch['controller_pid'],", "reference_keep_preflight_launch=str(root/'preflight_launch.json'),reference_keep_preflight_controller_pid=launch['controller_pid'],")
source=source.replace("next_action='Wait until scheduled ReferIt preflight observation; no formal training launched'", "next_action='Wait until00:16:49.814 CST original32309 first M0 observation; Scan first, Nr/Sr deferred'")
begin=source.index("stream.write('\\nPV-Ground '+record['time_cst']+")
end=source.index("\nprint(json.dumps(record)",begin)
source=source[:begin]+'''stream.write('\\nPV-Ground '+record['time_cst']+': Doc89 reference-preservation λ0/1 SOURCEPASS and actualM0 controller'+str(launch['controller_pid'])+'; current5598/4848 protected, no newformal result. Main '+heads[0]+'. Single2027; Nr/Srdeferred untilScan5620/4764+threeeffective modules. First sole32309 check00:16:49.814, noearlyNNquery.\\n')'''+source[end:]
compile(source,'publish_preflight_authorized.py','exec')
(root/'publish_preflight_authorized.py').write_text(source,encoding='utf-8')
print('DOC89_M0_PUBLICATION_PREPARED_NOT_EXECUTED')
