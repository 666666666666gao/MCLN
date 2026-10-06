"""Persist the user's updated research objective and append it to the pending Doc88."""
import datetime
import json
from pathlib import Path

root = Path(__file__).resolve().parent
stamp = datetime.datetime.now().astimezone().isoformat()
goal = dict(updated_cst=stamp, status='ACTIVE_UNMET', baseline='PV-Ground', seed=2027,
    multiseed=False, scanrefer=dict(rows=9508, acc025_strictly_greater_than=59.1,
    acc050_strictly_greater_than=50.1, minimum_hits025=5620, minimum_hits050=4764,
    current_best_hits025=5598, current_best_hits050=4848, same_complete_model_required=True),
    paper=dict(effective_modules_required=3, direct_ablations_required=True,
        zero_output_or_unvalidated_branches_do_not_count=True),
    execution_order=['ScanRefer thresholds and three effective modules',
        'Freeze the complete final model and inference', 'Independently train Sr3D and Nr3D from fair corresponding initializations'],
    constraints=['one native last/bbs scoring path', 'all256 candidates',
        'same Query Box and Mask', 'no deployed V99 dual ranking or side chain',
        'single A100 serial execution', 'keep best weights and required official/PV/G/V99 dependencies',
        'retire only authorized closed nonbest generated weights', 'scheduled monitoring, no early polling'],
    referit_preflight_r2='SOURCE_PREPARATION_ONLY_DEFERRED_BY_USER_SCAN_FIRST_PRIORITY',
    new_nr_sr_formal_results=0,
    user_request='更新一下目标吧，我希望我们的ScanRefer的Acc0.25和0.5分别超过59.1和50.1，然后有有效的三模块能支撑起论文，再拿我们完整的模型去训练Sr3D和Nr3D')
(root / 'current_research_goals.json').write_text(json.dumps(goal, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
text = f'''# PV-Ground当前研究目标（{stamp}）

用户本次目标替代“接近V99即可进入Nr/Sr”的旧准入条件，当前状态未完成。

1. ScanRefer同一完整模型、完整9508条原生验证：Acc@0.25严格超过59.1%，Acc@0.5严格超过50.1%，即至少5620/4764个命中。不得拼接不同检查点的两列指标。当前保留5598/4848，宽松项还差22条，严格项已达本项门槛。
2. 三项有效模块支撑论文：每项职责明确，有匹配的直接消融和实际输出/学习贡献；不将零输出、不使用或未验证分支写成有效创新。候选方向为预测支撑空间参考、真实边界观测精修、参考约束的跨任务几何学习，后两项仍待验证，不能预先计为三项已成立。原PV和已有G机制如实披露，已有前作不重新包装。
3. 前两项满足后固定完整模型，再从对应公平初始化独立训练Sr3D和Nr3D。三个基准报告同一最终方法版本；不把Scan权重直接测试结果称为Nr/Sr训练结果。既有Nr/Sr接口及R2预检源码作为准备保留，正式训练现在不启动。

PV-Ground为baseline；单一路径last/bbs、保留全部256候选，同一Query输出Box和Mask，不恢复V99双源/侧链。固定seed2027，不做多seed搜索、挑选或集成。A100任务串行，按实测估算时间，到了约定时间再查，接近结束复查间隔180～300秒。

保留当前Scan最佳及必要官方PV/原G/V99依赖，及时清理用户已授权范围内的自生成、闭合、非最佳权重。历史负实验的文本和指标仍保留。当前最优50.9886%来自本轮零更新Mask空间参考，不称学习式精修/蒸馏增益。
'''
(root / 'CURRENT_RESEARCH_GOALS.md').write_text(text, encoding='utf-8')
old_contract = Path('C:/Users/gb/.codex_pvground_cs_20261002/idea-stage/docs/research_contract.md').read_bytes()
assert old_contract == Path('C:/Users/gb/.codex_mcln_g0_20260905/idea-stage/docs/research_contract.md').read_bytes()
(root / 'research_contract_before_goal_update.md').write_bytes(old_contract)
current = text.encode('utf-8') + b'\n## Historical contract through2026-10-02 (superseded; evidence preserved)\n\n' + old_contract
(root / 'current_research_contract.md').write_bytes(current)
publisher = root / 'publish_comparison_authorized.py'
source = publisher.read_text(encoding='utf-8')
(root / 'publish_comparison_authorized_before_goal_update.py').write_text(source, encoding='utf-8')
source = source.replace("diag=root/'initial_comparison'", "goal=json.loads((root/'current_research_goals.json').read_bytes())\nassert goal['scanrefer']['minimum_hits025']==5620 and goal['scanrefer']['minimum_hits050']==4764 and not goal['multiseed']\ndiag=root/'initial_comparison'")
source = source.replace("stamp=datetime.datetime.now().astimezone().isoformat()", "contract='idea-stage/docs/research_contract.md'\nold_contract=(root/'research_contract_before_goal_update.md').read_bytes()\nnew_contract=(root/'current_research_contract.md').read_bytes()\nassert all((repo/contract).read_bytes()==old_contract for repo in repos[:2])\nstamp=datetime.datetime.now().astimezone().isoformat()", 1)
old_next = '下一步根据实际数值采用最小预检修正，再完成各对应初始化的真实两步更新和恢复检查；本节不将诊断替代M0或Nr/Sr完整精度。'
new_next = '依据实际数值的最小R2源码修正仅作为准备保留，本节不将诊断替代M0或Nr/Sr完整精度。用户在本次明确调整研究顺序，R2 GPU及Nr/Sr正式训练不启动。'
assert old_next in source; source = source.replace(old_next, new_next)
sentence = 'Scan best5598/4848及原PV/G/V99保持；'
new_sentence = '''用户最新目标已更新：ScanRefer同一完整模型Acc@0.25严格超过59.1%、Acc@0.5严格超过50.1%，9508条至少5620/4764命中。当前5598/4848，宽松项还差22条，严格项已满足这项新门槛；还需三项有直接消融和实际作用的有效模块，零输出或未验证分支不计作贡献。先在ScanRefer满足两阈值与三模块证据，再固定完整模型独立训练Sr3D/Nr3D；不能混合不同最终方法版本或用Scan终点零更新冒充独立训练。候选方向为支撑空间参考、边界观测精修、参考约束几何学习，后两项仍待验证。旧“接近V99即可转Nr/Sr”准入被本次要求替代，当前无活动GPU任务。

Scan best5598/4848及原PV/G/V99保持；'''
assert sentence in source; source = source.replace(sentence, new_sentence)
source = source.replace("base_names=['RUNTIME_FAILURE_SOURCE_DIAGNOSIS.md'", "base_names=['CURRENT_RESEARCH_GOALS.md','current_research_goals.json','current_research_contract.md',\n    'research_contract_before_goal_update.md','prepare_goal_publication.py',\n    'RUNTIME_FAILURE_SOURCE_DIAGNOSIS.md'")
source = source.replace("evidence=(project/b['prefix']).resolve()", "contract=project/b['contract']\nassert contract.read_bytes()==base64.b64decode(b['old_contract'])\nevidence=(project/b['prefix']).resolve()")
source = source.replace("assert doc.read_bytes()==new\nprint", "assert doc.read_bytes()==new\nnew_contract=base64.b64decode(b['new_contract']);contract.write_bytes(new_contract)\nassert contract.read_bytes()==new_contract\nprint")
source = source.replace("bundle=dict(doc=doc,prefix=prefix,", "bundle=dict(doc=doc,prefix=prefix,contract=contract,old_contract=base64.b64encode(old_contract).decode(),\n    new_contract=base64.b64encode(new_contract).decode(),")
source = source.replace('for path in copies:path.write_bytes(new)', 'for path in copies:path.write_bytes(new)\nfor repo in repos[:2]:(repo/contract).write_bytes(new_contract)')
source = source.replace("stage+=['MANIFEST.md',*payloads]", "stage+=['MANIFEST.md',contract,*payloads]")
source = source.replace("next_action='Use actual A/B/C deltas for smallest initialization-check repair; formal Nr/Sr not launched'", "next_action='ScanRefer >59.1/>50.1 and three effective modules first; R2 preparation retained, Nr/Sr formal deferred',\n    current_research_goals=str(root/'current_research_goals.json'),scanrefer_minimum_hits=[5620,4764],\n    effective_modules_required=3,single_seed=2027,no_multiseed=True")
compile(source, 'publish_comparison_authorized.py', 'exec')
publisher.write_text(source, encoding='utf-8')
state_path = root.parent / 'pvground_query_supported_geometry_20261005/active_continuation_state.json'
state = json.loads(state_path.read_bytes())
state.update(time_cst=stamp, current_research_goals=str(root/'current_research_goals.json'),
    scanrefer_minimum_hits=[5620,4764], effective_modules_required=3, single_seed=2027, no_multiseed=True,
    owned_gpu_job_active=False, referit_preflight_r2_deferred_by_user=True,
    referit_formal_training_launched=False, overall_goal_complete=False,
    next_action='ScanRefer thresholds >59.1/>50.1 and three effective modules before full Sr3D/Nr3D training.')
state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
with Path('C:/Users/gb/memory/2026-10-06.md').open('a', encoding='utf-8') as stream:
    stream.write('\nPV-Ground '+stamp+': USER_UPDATED_GOAL: same completeScan model strictlyAcc25>59.1/Acc50>50.1, at least5620/4764 of9508; current5598/4848 @25need22; three ACTUALLY effective modules with directablations, THEN freezefullmodel and independently trainSr/Nr. Prior Scan58.3/50 gate and immediateNr/Sr priority superseded. Fixed2027/no multiseed. R2sourceprep retained, no R2GPUlaunch/formalNrSr; no ownedGPUjob. Goal ACTIVE_UNMET, zerooutputbranches notcontributions. PendingDoc88 publisher revised toincludeusergoal andcurrentresearchcontract; exactSOURCErefresh required, notexecuted.\n')
print('USER_GOAL_PERSISTED_AND_PUBLISHER_UPDATED_NOT_EXECUTED')
