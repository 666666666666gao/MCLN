"""Prepare the one-variable normalization control using the completed runner."""
import hashlib
import json
from pathlib import Path

local = Path(__file__).parent
previous = local.parent / 'pvground_candidate_consistency_20261003'
remote = '/root/autodl-tmp/pvground_candidate_normalization_20261003'
assert not (local / 'run.py').exists()
shared = ['run.py', 'pvground_semantic_assignment.py', 'pvground_task_observation_query.py',
          'pvground_observation_query.py', 'pvground_source_query.py']
for name in shared:
    (local / name).write_bytes((previous / name).read_bytes())

source = (previous / 'pvground_candidate_consistency.py').read_text(encoding='utf-8')
source = source.replace('Only the semantic correspondence is expanded. Native Hungarian regression and\n'
                        'Mask responsibilities, the CE replacement and inference remain unchanged.',
                        'The same semantic correspondence is expanded and the entire expanded final\n'
                        'contrastive term uses its expanded correspondence count. Native Hungarian\n'
                        'regression, Mask responsibilities, CE replacement and inference are unchanged.')
old = "    new = set_criterion.loss_sem_align(outputs, targets, expanded, num_boxes, None)['loss_sem_align']"
new = ("    expanded_count = sum(len(queries) for queries, _ in expanded)\n"
       "    new = set_criterion.loss_sem_align(outputs, targets, expanded, expanded_count, None)['loss_sem_align']")
assert source.count(old) == 1
source = source.replace(old, new)
source = source.replace('                           semantic_target_only=True), selected',
                        '                           contrastive_native_denominator=num_boxes,\n'
                        '                           contrastive_expanded_denominator=expanded_count,\n'
                        '                           semantic_target_only=False,\n'
                        '                           expanded_count_normalization=True), selected')
(local / 'pvground_candidate_consistency.py').write_bytes(source.encode('utf-8'))

test = (previous / 'cpu_test.py').read_text(encoding='utf-8')
old = "expected = native.loss_sem_align(outputs, targets, expanded, 3, None)['loss_sem_align'] * (.5 / 7)"
assert test.count(old) == 1
test = test.replace(old, "expected = native.loss_sem_align(outputs, targets, expanded, 7, None)['loss_sem_align'] * (.5 / 7)")
test = test.replace('    corrected = old * (.5 / 7) + correction',
                    "    assert statistics['contrastive_native_denominator'] == 3\n"
                    "    assert statistics['contrastive_expanded_denominator'] == 3 + int(selected.sum())\n"
                    "    corrected = old * (.5 / 7) + correction")
test = test.replace("CPU_only=True, model_forward=False, fixtures=records)",
                    "CPU_only=True, model_forward=False, expanded_count_normalization=True, fixtures=records)")
(local / 'cpu_test.py').write_bytes(test.encode('utf-8'))

module_sha = hashlib.sha256((local / 'pvground_candidate_consistency.py').read_bytes()).hexdigest()
for arm in ('preflight', 'normalized'):
    spec = json.loads((previous / 'g_consistent_spec.json').read_bytes())
    spec.update(root=remote + '/' + arm, consistency_module_sha256=module_sha,
                comparison='same G qualification expansion and native formula; expanded final contrastive denominator N+A',
                contrastive_normalization='expanded_correspondence_count',
                completed_comparators=str(previous))
    (local / (arm + '_spec.json')).write_text(json.dumps(spec, indent=2) + '\n', encoding='utf-8')

plan = (previous / 'NORMALIZATION_CONTROL_PLAN.md').read_text(encoding='utf-8')
plan = plan.replace('# 同对应集合下的归一化控制（待实施）', '# 同对应集合下的归一化控制')
plan = plan.replace('状态：仅设计，没有源码改动、GPU前向、优化步或精度结果。',
                    '状态：实现准备完成；尚未代码终审、真实CPU/GPU预检或正式训练。')
plan += ('\n## 实现与执行\n\n复用上一轮run.py，文件字节不变；G资格和CE模块亦字节不变。'
         '唯一损失变量是expanded_count替代new对比项的原num_boxes分母。记录N、N+A及新增对应数量。'
         '只新增一个normalized训练组，预检从原G重新加载并做2步真实更新，随后正式训练另从原G与fresh AdamW开始，预检步不计入3723步。'
         '控制器串行执行normalized/train与normalized/formal，复用既有单GPU锁；不重新训练旧控制。'
         '正式使用同一原生评估和GT数据接口；训练与推理职责不混用。无新增模型参数，全部256候选保留。\n\n'
         '预期单组train约2小时54分，formal约17分，总约3小时13分；依据上一轮实际耗时估计，非完成承诺。'
         '按预计结束前几分钟开始180～300秒轮询，活动训练不改代码。只保留一个活动恢复点；非最佳终点完整归档并核对后按既有授权清理。\n')
(local / 'EXPERIMENT_PLAN.md').write_bytes(plan.encode('utf-8'))
(local / 'EXPERIMENT_TRACKER.md').write_text(
    '# 归一化控制执行表\n\n| 阶段 | 状态 | 证据 |\n|---|---|---|\n'
    '| 代码实现 | READY_FOR_REVIEW | 复用runner；仅扩展最后层对比项分母改变 |\n'
    '| CPU原生公式核验 | PENDING | A=0损失/梯度；A>0直接调用；其他GT保护 |\n'
    '| GPU batch8预检 | PENDING | 2步、有限梯度、恢复与显存 |\n'
    '| normalized训练 | NOT_STARTED | 原G/fresh AdamW；3723步；29778条一次 |\n'
    '| 正式9508验证 | NOT_STARTED | 原生last/bbs；所有候选与GT离线诊断 |\n', encoding='utf-8')
(local / 'research_contract.md').write_text(
    '# 研究合同：扩展对应后的归一化\n\n假设：在相同G资格和对应扩展下，扩展匹配计数归一化可能改善有效监督规模。'
    '只检验这一训练策略变量，不声称纯标签作用、已达到50%或已跨数据集有效。'
    '必须报告相对本轮历史控制、未归一化一致性组及原G的Acc@0.50，附Acc@0.25、Mask和修复/破坏。'
    '跨进程起点数值限制实际核验并披露；原G5615/4495未被超越时不晋级。'
    '目标为保住5615宽松命中并达到4754严格命中。GT资格仅训练/离线诊断，推理保留全部256候选和原生唯一评分。\n', encoding='utf-8')
manifest = {name: hashlib.sha256((local / name).read_bytes()).hexdigest()
            for name in shared + ['pvground_candidate_consistency.py', 'cpu_test.py']}
assert manifest['run.py'] == '919fc36d61b8f21a7285f55c153717f05baa4e969ca43ccbc25171e165bc7392'
(local / 'source_preparation.json').write_text(json.dumps(dict(status='ready_for_review',
    source_shas=manifest, runner_byte_identical=True, official_environment_reused=True,
    new_model_parameters=0, new_gpu_experiment_started=False), indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(status='ready_for_review', module_sha256=module_sha, runner_byte_identical=True)))
