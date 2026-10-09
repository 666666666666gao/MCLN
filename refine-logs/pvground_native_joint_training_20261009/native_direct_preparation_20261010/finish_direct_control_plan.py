import datetime
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
controls = root / 'native_direct_controls_20261010'
now = datetime.datetime.now().astimezone()
stage = controls / 'refine-logs'
assert not stage.exists()
stage.mkdir()
claim_map = '''# 正常训练直接对照执行计划

主要论点C1：候选支撑修正与边界来源信息能在原生正常训练中提供真实增量。
主要论点C2：部署选中但未匹配的候选获得明确Mask目标具有独立价值。

五个核心实验块继续沿用已发布的正常训练计划：B1全模型正常训练；B2三机制直接控制；B3更简单固定融合；B4同一完整模型Nr3D/Sr3D；B5必要的修复/破坏分析。这里补齐B2/B3的可执行源码准备，不替换当前活动训练或其配方。完整共同协议见同级各版本的NORMAL_NATIVE_RUN_PROTOCOL.json，以及上一阶段normal_controls_20261010/refine-logs/EXPERIMENT_PLAN.md。

顺序：先等活动正常训练的E0与3轮终态，完成正式结果与恢复检查；再做已准备的C-off最小对照；然后按A/B所缺的贡献证据逐个选择本目录控制。common接口仅做实际行为等价检查，不默认另跑3轮；A内容输入对照是信息隔离的后续，固定B是简单性直接对照。Nr/Sr在完整最终版本固定后各自加载公平的作者父权重并训练。

同一seed2027、全ScanRefer36665表达、有效batch8、3轮、每轮预期4583更新及完整9508验证；实际loader数量须从运行日志确认。核心/骨干LR1e-6、新增模块1e-5、WD5e-4、clip0.1、优化器重新初始化。每个3轮ScanRefer控制预计约15–19 GPU小时，这是工程吞吐粗估，须由活动实际轮次修正；没有承诺或排队全部控制，Nr/Sr成本尚未实测。每组保留best/latest和必要依赖，闭合后清理无用生成权重。

成功标准：同一正常模型达到至少5658/4850，即严格大于59.5%/51%；三机制分别超出直接控制，并报告真实新增能力而非仅追回退化控制。统一版本在Nr/Sr按各自原生协议超过对应baseline；原MCLN论文数值只作外部参照。失败时依据E0、Mask/Box变化与Query选择判断，不无限延长训练、不用新后处理掩盖负结果。

本方法没有LLM/VLM/扩散/RL额外模块，不开展无关frontier实验。必需证据为完整正常主结果、三机制最小控制和跨基准；等有效参数量的额外容量扫描不是当前默认任务。所有模块历史、输入置零导致的梯度容量变化、冻结分支、硬Mask参考均披露。

下面记录本次源码准备的具体机制、接口差异及准入检查。

'''
plan = claim_map.encode('utf-8') + (controls / 'DIRECT_CONTROLS.md').read_bytes()
tracker = (controls / 'EXPERIMENT_TRACKER.md').read_bytes()
entries = []
for name, raw in (('EXPERIMENT_PLAN', plan), ('EXPERIMENT_TRACKER', tracker)):
    version = stage / (name + '_' + now.strftime('%Y%m%d_%H%M%S') + '.md')
    latest = stage / (name + '.md')
    version.write_bytes(raw)
    latest.write_bytes(raw)
    for path in (version, latest):
        entries.append((path.relative_to(controls).as_posix(), hashlib.sha256(raw).hexdigest()))
for path in sorted(controls.rglob('*')):
    if path.is_file() and path.parent != stage and 'actual_source_review' not in path.parts:
        entries.append((path.relative_to(controls).as_posix(), hashlib.sha256(path.read_bytes()).hexdigest()))
text = '# Research Output Manifest\n\n| Timestamp | Skill | File | Stage | Description |\n|---|---|---|---|---|\n'
for name, digest in entries:
    text += '| ' + now.strftime('%Y-%m-%d %H:%M:%S') + ' | /experiment-plan | ' + name + ' | source preparation | SHA256 ' + digest + ' |\n'
(controls / 'MANIFEST.md').write_text(text, encoding='utf-8')
record = dict(status='NATIVE_A_B_DIRECT_CONTROLS_AND_INTERFACE_NOTES_PREPARED',
    generated_cst=now.isoformat(), files=[dict(path=name, sha256=digest) for name,digest in entries],
    live_normal_source_unchanged=True, current_training_status_queries=0,
    three_effective_contributions_proven=False, normal_controls_launched=False,
    full_goal_complete=False)
(controls / 'PLAN_PREPARATION.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
for path in (root.parent.parent / 'pvground_referit_mask_reference_20261006/current_research_goals.json',
             root.parent.parent / 'pvground_query_supported_geometry_20261005/active_continuation_state.json'):
    value = json.loads(path.read_bytes())
    value['normal_training_requirement'].update(
        native_A_B_control_preparation=str(controls / 'DIRECT_CONTROL_PREPARATION.json'),
        native_A_B_control_plan=str(stage / 'EXPERIMENT_PLAN.md'),
        native_A_B_control_tracker=str(stage / 'EXPERIMENT_TRACKER.md'),
        referit_native_interface_notes=str(controls / 'REFERIT_NATIVE_INTERFACE_NOTES.md'))
    value.update(current_turn_classification='ACTUAL_PROGRESS_NATIVE_A_B_DIRECT_CONTROL_SOURCE_PREPARED',
                 full_goal_complete=False)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
note = ('\nPV-Ground ' + now.isoformat() + ': isolated normal A bypass/content-only and B fixed-half '
    'controls prepared. Only3 of14 source files changed in isolated copy; active14 source SHAs unchanged. '
    'Original matching/score/criterion/normal recipe unchanged; same trained parent history disclosed. '
    'Bypassed A27841 / fixed B29793 registered but frozen; full1295 state/factory/recovery still unverified. '
    'Fresh source audit pending. CPU bundle/test only prepared, no execution yet, no GPU/current remote job query. '
    'Nr/Sr interfaces checked in actual SHA-bound dataset snapshot: language versus annotation source differs, '
    'Sr token weights0.625/0.125/0.125/0.125 versusScan/Nr0.6/0.2/0.2/0.1; language coefficient1 versusScan0.5; '
    'G and C currentlyScan asserted. Author weightsallowed, do not overwriteNr/Sr core withfullScanGdelta unnoticed. '
    'First normal observer unchanged08:19. Full goalACTIVE_UNMET.\n')
for path in (Path('C:/Users/gb/memory/2026-10-10.md'), Path('C:/Users/gb/MEMORY.md')):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(note)
print(json.dumps(dict(status=record['status'], files=len(entries), neural_calls=0,
    active_source_unchanged=True, full_goal_complete=False)))
