"""Record the user's new research objective without changing an active run."""
import datetime
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
out = root / 'research_target_update_20261011'
assert not out.exists()
out.mkdir()
now = datetime.datetime.now().astimezone().isoformat()
request = ('继续优化我们的创新点，然后看看能不能在Scanrefer上把0.25和0.5分别推进到60.0以及52.0，'
           '根据找到的问题进行优化或改进我们的模块，可以借鉴其他领域已经开源的代码模块，'
           '可以加入诊断指标进行判断为什么会失败，也可以根据实际问题来尝试加入新的模块或者改进我们的模块，'
           '另外注意一下我们的模块与原来baseline的是否有冲突可以修复，在Scanrefer上如果很接近目标了'
           '但怎么改进也没有办法的话则可以试一下调参，达到目标后可以加载baseline的nr3d和sr3d的权重'
           '先做一次验证，看看预训练权重指标能有多少，然后再训练我们的模块，因为数据集个性不同，'
           '可能需要在保证我们的网络模型以及模块不变的情况下，针对不同的数据集有些模块的参数可能需要调整'
           '才能适配，这两个数据集的指标也要超过我们baseline论文里汇报的值')
paper = Path('C:/Users/gb/.codex/tmp/pvground_novelty_20261009/PV-Ground.pdf')
assert paper.is_file()
targets = dict(updated_cst=now, status='ACTIVE_UNMET', baseline='PV-Ground',
    user_request=request, seed=2027, multiseed=False,
    scanrefer=dict(rows=9508, acc025_at_least_percent=60.0, acc050_at_least_percent=52.0,
        minimum_hits025=5705, minimum_hits050=4945, same_complete_model_required=True,
        current_best_hits025=5677, current_best_hits050=4920,
        remaining_hits025=28, remaining_hits050=25, current_best_joint_target_passed=False,
        current_best_scope='retained E0, not a new normal-joint-training improvement'),
    paper=dict(effective_modules_required=3, direct_ablations_required=True,
        zero_output_or_unvalidated_branches_do_not_count=True,
        reuse_requires_problem_specific_adaptation=True, check_conflicts_with_baseline=True),
    optimization=dict(priority='diagnosis-grounded module improvement in normal network training',
        diagnostics_allowed=True, new_or_revised_modules_allowed=True,
        open_source_cross_domain_reuse_allowed=True,
        hyperparameter_tuning_allowed_after_near_target_structure_improvement_stalls=True,
        no_unbounded_same_configuration_extension=True),
    cross_benchmark=dict(order=['Reach ScanRefer target with the complete method',
        'Evaluate each corresponding author Nr3D/Sr3D checkpoint before adaptation',
        'Train the unchanged complete network/modules independently on Nr3D and Sr3D'],
        author_pretrained_initialization_allowed=True, unchanged_network_and_module_structure=True,
        dataset_specific_module_parameter_adjustment_allowed=True,
        parameter_changes_must_be_disclosed=True,
        exceed_baseline_paper_on_corresponding_protocol=True,
        author_paper_two_stage_accuracy_percent=dict(nr3d=62.1, sr3d=68.9),
        author_paper_single_stage_accuracy_percent=dict(nr3d=51.3, sr3d=56.5),
        paper_metric='ReferIt3D object-candidate selection accuracy, not a reported Acc@0.5 column',
        native_box_acc025_acc050_also_recorded=True,
        paper_to_native_evaluation_protocol_equivalence_still_requires_verification=True),
    constraints=['one native scoring path', 'all 256 candidates', 'same Query box and mask',
        'normal forward and criterion integration', 'no V99 deployed dual ranking or side chain',
        'single A100 serial execution', 'seed2027 only',
        'retire proven unused generated weights; protect best and required dependencies',
        'scheduled observations; 180-300 seconds only as needed near completion'],
    active_experiment_selection_rule_unchanged=True,
    future_experiment_selection_rule_must_use_updated_targets=True,
    active_training_source_changed=False, new_neural_calls=0, training_restarted=False,
    full_goal_complete=False)
targets['baseline_paper_source'] = dict(
    url='https://openaccess.thecvf.com/content/CVPR2026/papers/Shang_PV-Ground_Text-Guided_Point-Voxel_Interaction_for_3D_Visual_Grounding_CVPR_2026_paper.pdf',
    local_path=str(paper), sha256=hashlib.sha256(paper.read_bytes()).hexdigest(),
    table=2, pdf_page=7, verified_by_actual_pdf_text_extraction=True)
(out / 'CURRENT_RESEARCH_TARGETS.json').write_text(json.dumps(targets, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
paths = [Path('C:/Users/gb/.codex/tmp/pvground_referit_mask_reference_20261006/current_research_goals.json'),
         Path('C:/Users/gb/.codex/tmp/pvground_query_supported_geometry_20261005/active_continuation_state.json'),
         root / 'NORMAL_CONTINUATION_STATE.json']
for index, path in enumerate(paths):
    value = json.loads(path.read_text(encoding='utf-8-sig'))
    (out / ('prior_active_state_' + str(index) + '.json')).write_bytes(path.read_bytes())
    value.update(updated_cst=now, research_goal_update=targets, status='ACTIVE_UNMET', full_goal_complete=False)
    if index == 0:
        old = value['scanrefer']
        value['previous_scanrefer_targets_at_20261011_update'] = {key:old[key] for key in
            ('acc025_strictly_greater_than','acc050_strictly_greater_than','minimum_hits025','minimum_hits050')}
        del old['acc025_strictly_greater_than'], old['acc050_strictly_greater_than']
        old.update(targets['scanrefer'])
        old['current_best_metric_scope'] = targets['scanrefer']['current_best_scope']
        value['user_request'] = request
        value['execution_order'] = targets['cross_benchmark']['order']
        value['cross_benchmark_initialization'].update(
            corresponding_author_checkpoint_evaluation_before_module_training=True,
            dataset_specific_module_parameter_adjustment_allowed=True,
            same_final_architecture_across_benchmarks=True)
    if index == 1:
        value['target_hits'] = [5705,4945]
        value['strict_target_gap'] = 25
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
notes = '''# PV-Ground当前研究目标（2026-10-11更新）

完整目标保持ACTIVE_UNMET。ScanRefer同一完整模型、完整9508条、原生last/bbs：Acc@0.25至少60.0%，Acc@0.5至少52.0%，即5705／4945命中。现有保留E0为5677／4920，还差28／25；已有模型不是正常联合训练的新增收益。不能拼接不同检查点的两列。

继续以实际问题优化创新机制，在正常网络前向和criterion中实现。三项有效机制仍需明确职责、匹配直接消融和实际贡献；不计零输出、未使用或未验证分支。检查新增机制与PV／原G是否冲突；允许借鉴开源跨领域代码，但需要针对已发现问题适配。允许有用的诊断、新模块和模块修订；结构改进接近目标后仍停滞，可进行有依据的调参，不无限延长同一失败配置。

达ScanRefer目标后，先分别完整验证作者对应Nr3D／Sr3D预训练权重，再训练完整方法。网络与模块结构保持一致；允许按数据集调整并披露模块参数。最终结果同时比较实际同协议作者权重和baseline论文。PV-Ground论文Table2（PDF第7页）两阶段对象选择Accuracy为Nr3D62.1%、Sr3D68.9%，单阶段为51.3%、56.5%；该表不报告Acc@0.5列，不能把预训练日志IoU指标冒称论文严格阈值。对应评估协议等价性仍须核对，原生框双阈值同时保留。

来源：https://openaccess.thecvf.com/content/CVPR2026/papers/Shang_PV-Ground_Text-Guided_Point-Voxel_Interaction_for_3D_Visual_Grounding_CVPR_2026_paper.pdf

固定seed2027、不做多seed；单A100串行；保留全部256候选和同Query框／Mask；不恢复V99部署双源或侧链。及时清理已闭合且证明无用的自生成权重，保护最佳和必要父状态。主任务按实测速率估计，在约定时间观察，必要补查间隔180～300秒。

正在运行的C关闭三轮对照保持原配置和事前选模规则，不因新目标中途改源码或追溯改选模。Query自身Mask匹配源码已部署，尚未运行真实GPU预检和正式训练；未来新实验的选模门槛要更新为5705／4945。当前唯一主观察器PID28804仍按09:07:13执行，未增加服务器轮询或重启训练。
'''
(out / 'CURRENT_RESEARCH_TARGETS.md').write_text(notes, encoding='utf-8')
memory_note = ('\nPV-Ground '+now+': USER GOAL UPDATED to same9508 ScanRefer >=60.0/>=52.0 '
    '(5705/4945; retained E0 5677/4920 gaps28/25). Continue evidence-grounded normal-network module improvements, '
    'diagnostics, baseline conflict checks and problem-specific open-source adaptation; hyperparameter tuning allowed '
    'after near-target module improvements stall. Three effective directly validated mechanisms remain required. '
    'After Scan target evaluate author Nr/Sr parents first, then train same architecture/modules; disclosed dataset-specific '
    'module parameters allowed; exceed corresponding baseline-paper accuracy. Actual PDF Table2 page7 primary source '
    'two-stage Nr62.1/Sr68.9 selection accuracy (not an Acc@0.5 table); native protocol equivalence pending. '
    'Current C-off run/selection unchanged; original observer28804 due09:07; no new GPU call/restart. ACTIVE_UNMET.\n')
for path in (Path('C:/Users/gb/memory/2026-10-11.md'), Path('C:/Users/gb/MEMORY.md')):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(memory_note)
print(json.dumps(dict(status='NEW_FULL_RESEARCH_TARGET_RECORDED', minimum_hits=[5705,4945],
    current_gap=[28,25], active_run_unchanged=True, full_goal_complete=False)))
