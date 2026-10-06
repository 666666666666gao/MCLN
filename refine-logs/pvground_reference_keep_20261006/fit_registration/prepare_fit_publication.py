"""Prepare an additive Doc90 recording actual M0 closure and the formal launch."""
from pathlib import Path

root = Path(__file__).resolve().parent
source = (root / 'publish_preflight_authorized.py').read_text(encoding='utf-8')
source = source.replace('Append only reference-preservation source and actual ScanRefer M0 launch to Doc88.', 'Append actual M0 closure and formal ScanRefer reference-keep launch to Doc89.')
source = source.replace('preflight_publication.json', 'fit_publication.json')
source = source.replace('PUBLISH_PREFLIGHT_SOURCE_REVIEW', 'PUBLISH_FIT_SOURCE_REVIEW')
source = source.replace("root/'SOURCE_REVIEW.json'", "root/'FIT_SOURCE_REVIEW.json'")
begin = source.index("launch=json.loads((root/'preflight_launch.json').read_bytes())")
end = source.index("state_path=root.parent", begin)
source = source[:begin] + '''launch=json.loads((root/'fit_launch.json').read_bytes())
assert launch['accuracy_result'] is False and launch['updates_per_arm']==3723
assert launch['protected_best_hits']==[5598,4848] and launch['candidate_gate_hits']==[5620,4764]
assert launch['root']=='/root/autodl-tmp/pvground_reference_keep_20261006'
m0=json.loads((root/'preflight_wait.json').read_bytes())
assert m0['observer_closed'] and m0['exitcode']==0 and m0['status']['status']=='complete'
observer=json.loads((root/'fit_observer_started.json').read_bytes())
assert observer['first_check_seconds']==24300 and observer['poll_seconds']==240
previous_path=root/'preflight_publication.json'
previous=json.loads(previous_path.read_bytes());assert previous['section']=='20.376.89'
''' + source[end:]
source = source.replace('preflight_registration', 'fit_registration')
begin = source.index("section=f'''")
end = source.index('client=paramiko.SSHClient()', begin)
source = source[:begin] + '''section=f\'\'\'

## 20.376.90 参考保持约束：真实两步预检通过，正式同预算对照启动（{stamp}）

用户目标保持：同一完整模型9508条ScanRefer Acc@0.25严格>59.1%、Acc@0.5严格>50.1%，至少5620/4764命中；三个模块分别有直接消融和有效贡献之后，固定完整方法，Sr3D/Nr3D分别从作者对应预训练权重公平初始化并独立训练。固定seed2027，不做多seed。当前受保护最好模型5598/4848；这次工程预检不是新的精度结果，也不是第三项贡献已经成立。

上一节M0启动记录已获得真实终态：控制器776037于{m0['status']['finished_cst']}结束，exit0，两组各执行两次优化器更新，没有产生预检权重文件。两组仅更新456102参数/10状态的已有几何头，父模型、全零R、原生bbs和Mask保持；中性初始解码严格等于预测Mask参考，保持项初始严格为0。输出层更新后，新约束有非零直接输出梯度，且只作用可信对应集合；几何头及Adam的键、组、步数和moments实际内存CPU保存恢复一致。39条真实空支撑样本保持既有原框规则，全部256候选的原始点成员极值见证通过。

控制λ=0，策略λ=1，均使用同一融合Mask参考、同一保留initial.pth重建和fresh optimizer；不承接M0两步状态。训练内参考退化项为表达内均值再实际batch均值的squared ReLU(stopgrad(IoU(reference,GT))-IoU(final,GT))。原匹配候选对应自己的过滤后真实GT；额外候选要求自身Query及融合Mask均得到root支撑确认，排除全部原匹配，没有Box>0.5截断。GT资格不进入推理；原native/G/匹配DFL及已有额外定位目标不变，一套last/bbs、同Query框与Mask、256候选保留。两组跨进程M0有极小浮点差异，应将后续称为同起点同预算经验对照，不宣称逐位配对。

正式控制器{launch['controller_pid']}实际启动于{launch['time_cst']}，screen {launch['screen']}，共享GPU锁串行运行。每组fit29778条一次、3723更新、batch8/累积1、lr1e-5/WD5e-4/clip0.1；初始及终点各评估6887模块留出和9508正式开发验证。6887场景已被作者预训练见过，不作为正式泛化证据。当前仅启动，没有发布新命中数。保留隐藏状态此前11169次更新，终点14892；本轮重置输出层终点3723，原G历史另计。

实际启动资源：数据盘空闲{launch['resources']['data_free_bytes']}字节，系统盘{launch['resources']['system_free_bytes']}字节，保存预留{launch['resources']['required_reserve_bytes']}字节；官方PV、原G和当前4848权重保持。沿用已有A100/Torch1.10.2环境，不安装新包。上一同预算完整pair耗时24792秒，当前估计约7小时；唯一观察器PID {observer['observer_local_pid']}，首次远端查看{observer['first_observation_cst']}，此前不轮询，临近结束按240秒复查。旧M0观察器32309及Doc89发布80406已消费exit0，不能重复运行。

本轮需同时超过同预算控制与自身4848起点，并核对5620/4764双门槛；保留修复/破坏和完整候选覆盖。若只追回控制退化，不认定新增模块有效。实验闭合后按已授权范围清理自身非最佳权重，保留日志、指标、必要依赖与最好权重；不自动扩大原始数组或数据集删除权限。Nr/Sr正式训练继续等待ScanRefer与三模块证据，不因这次M0通过而提前启动。
\'\'\'
new=old+section.encode('utf-8');assert new.startswith(old) and new.count(b'## 20.376.90 ')==1
names=['EXPERIMENT_PLAN.md','EXPERIMENT_TRACKER.md','research_contract.md','control_spec.json','keep_spec.json',
    'reference_keep.py','run_reference_keep_fit.py','selected_mask_reference_factory.py','controller.py',
    'preflight_wait.json','preflight_complete/control/preflight.json','preflight_complete/keep/preflight.json',
    'FIT_SOURCE_REVIEW.json','FIT_SOURCE_REVIEW.md','fit_launch.json','fit_resource_check.json','fit_observer_started.json',
    'prepare_fit_launch.py','launch_fit_authorized.py','observe_fit_authorized.py','collect_closed_fit_authorized.py',
    'prepare_fit_publication.py','publish_fit_authorized.py','PUBLISH_FIT_SOURCE_REVIEW.json','PUBLISH_FIT_SOURCE_REVIEW.md']
payloads={prefix+name:(root/name).read_bytes() for name in names};payloads[prefix+'.gitattributes']=b'** -text\\n'
assert all(not name.endswith(('.pt','.pth','.npz')) and '.aris' not in name for name in payloads)
''' + source[end:]
source = source.replace('Reference-preservation-only ScanRefer loss contrast actual M0 launched; no new formal metrics.', 'Reference-preservation real M0 PASS and same-budget formal fit launched; no new formal metrics.')
source = source.replace('Register ScanRefer reference-preservation loss and real M0 launch', 'Record reference-preservation M0 PASS and formal ScanRefer contrast launch')
source = source.replace("section='20.376.89'", "section='20.376.90'")
source = source.replace("execution_scope='REFERENCE_KEEP_M0_LAUNCH_NOT_PASSED'", "execution_scope='REFERENCE_KEEP_M0_PASSED_FORMAL_FIT_LAUNCHED_NOT_COMPLETE'")
source = source.replace("reference_keep_preflight_launch=str(root/'preflight_launch.json'),reference_keep_preflight_controller_pid=launch['controller_pid']", "reference_keep_fit_launch=str(root/'fit_launch.json'),reference_keep_fit_controller_pid=launch['controller_pid'],reference_keep_observer_pid=observer['observer_local_pid'],reference_keep_first_observation_cst=observer['first_observation_cst']")
source = source.replace("next_action='Wait until00:16:49.814 CST original32309 first M0 observation; Scan first, Nr/Sr deferred'", "next_action='Wait until '+observer['first_observation_cst']+' sole formal-fit observer; Scan first, Sr/Nr author-init deferred'")
source = source.replace("workspace/'memory/2026-10-06.md'", "workspace/'memory/2026-10-07.md'")
begin = source.index("    stream.write('\\nPV-Ground '+record['time_cst']+")
end = source.index('\nprint(json.dumps(record)', begin)
source = source[:begin] + '''    stream.write('\\nPV-Ground '+record['time_cst']+': Doc90 actualM0bothPASS and formal controller'+str(launch['controller_pid'])+' launched, MAIN'+heads[0]+' ONEdocSHA'+digest+' exactfourlocal+remote+Git; no newformal result. Current5598/4848 protected. Single2027/no multiseed; Sr/Nr author-init AFTERScan5620/4764+threeeffective. Sole observer'+str(observer['observer_local_pid'])+' first'+observer['first_observation_cst']+', noearlyNNquery.\\n')''' + source[end:]
compile(source, 'publish_fit_authorized.py', 'exec')
(root / 'publish_fit_authorized.py').write_text(source, encoding='utf-8')
print('DOC90_PREPARED_NOT_EXECUTED')
