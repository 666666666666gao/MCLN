"""Describe closed saved outputs; no NN, remote query, or metric promotion."""
import csv
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

root = Path('\\\\?\\' + str(Path(__file__).resolve().parent))
captured = root / 'OBS_selected_geometry_rows_20261014_files'
recount_path = root / 'selected_geometry_cpu_recount_20261014' / 'RESULT.json'
recount = json.loads(recount_path.read_bytes())
assert recount['neural_forwards'] == recount['optimizer_updates'] == 0
output = root / 'selected_endpoint_changes_20261014'
assert not output.exists()
output.mkdir()
sources = {str(recount_path): hashlib.sha256(recount_path.read_bytes()).hexdigest()}
rows = {}
values = {}
for epoch in (0, 1, 2):
    path, = captured.rglob('native_rows_epoch%03d.jsonl' % epoch)
    raw = path.read_bytes()
    sources[str(path)] = hashlib.sha256(raw).hexdigest()
    assert sources[str(path)] == recount['source_hashes'][str(path)]
    rows[epoch] = [json.loads(line) for line in raw.decode('utf-8').splitlines()]
    assert len(rows[epoch]) == 9508
    cpu_path = root / 'selected_geometry_cpu_recount_20261014' / ('epoch%03d_selected_rows.csv' % epoch)
    sources[str(cpu_path)] = hashlib.sha256(cpu_path.read_bytes()).hexdigest()
    with cpu_path.open(newline='', encoding='utf-8') as stream:
        cpu_rows = list(csv.DictReader(stream))
    assert len(cpu_rows) == 9508
    values[epoch] = {key: np.array([float(r[column]) for r in cpu_rows]) for key, column in
        (('final', 'final_iou_cpu64'), ('native', 'native_iou_cpu64'), ('mask', 'mask_endpoint_iou_cpu64'))}
    for a, b in zip(rows[0], rows[epoch]):
        assert all(a[key] == b[key] for key in ('row_index', 'scan_id', 'target_id',
            'utterance_sha256', 'root_gt_box', 'point_cloud_sha256', 'gt_mask_sha256'))
volumes = np.array([np.prod(r['root_gt_box'][3:]) for r in rows[0]], dtype=np.float64)
cuts = np.quantile(volumes, [.25, .5, .75])
groups = np.searchsorted(cuts, volumes, side='right')
epoch_reports = []
for epoch in (0, 1, 2):
    v = values[epoch]
    support = np.array([2 * r['own_mask_intersection'] > r['own_mask_union'] and
        2 * r['fused_mask_intersection'] > r['fused_mask_union'] for r in rows[epoch]])
    report = {'epoch': epoch}
    for suffix, threshold in (('025', .25), ('050', .5)):
        initial = values[0]['final'] > threshold
        hit, native, mask = (v[key] > threshold for key in ('final', 'native', 'mask'))
        coverage = np.array([r['best256_iou'] > threshold for r in rows[epoch]])
        broken = initial & ~hit
        repaired = ~initial & hit
        endpoint_available = native | mask
        failed = ~hit
        categories = {
            'final_fails_but_at_least_one_selected_endpoint_passes': failed & endpoint_available,
            'both_selected_endpoints_fail_but_logged_full256_has_qualified_box': failed & ~endpoint_available & coverage,
            'both_selected_endpoints_fail_and_logged_full256_has_no_qualified_box': failed & ~endpoint_available & ~coverage,
        }
        assert sum(int(category.sum()) for category in categories.values()) == int(failed.sum())
        assert int(hit.sum()) == recount['epochs'][epoch][suffix]['hits']
        assert int(broken.sum()) == recount['epochs'][epoch][suffix]['versus_E0']['breaks']
        report[suffix] = {
            'hits': int(hit.sum()), 'native_hits_same_selected_query': int(native.sum()),
            'mask_endpoint_hits_same_selected_query': int(mask.sum()),
            'repairs_vs_E0': int(repaired.sum()), 'breaks_vs_E0': int(broken.sum()),
            'final_vs_mask_repairs': int((hit & ~mask).sum()),
            'final_vs_mask_breaks': int((~hit & mask).sum()),
            'all_errors_categories': {name: int(category.sum()) for name, category in categories.items()},
            'breaks_vs_E0_categories': {name: int((broken & category).sum()) for name, category in categories.items()},
            'breaks_vs_E0_own_and_fused_masks_still_qualified': int((broken & support).sum()),
            'expression_weighted_GT_volume_quartiles': [
                {'quartile': g + 1, 'expressions': int((groups == g).sum()),
                 'E0_hits': int((initial & (groups == g)).sum()),
                 'current_hits': int((hit & (groups == g)).sum()),
                 'repairs': int((repaired & (groups == g)).sum()),
                 'breaks': int((broken & (groups == g)).sum()),
                 'final_fails_but_selected_mask_endpoint_passes': int((~hit & mask & (groups == g)).sum())}
                for g in range(4)],
        }
    epoch_reports.append(report)
result = {
    'scope': 'CLOSED_C_SAVED_SELECTED_ENDPOINT_TRANSITIONS',
    'completed_cst': datetime.datetime.now().astimezone().isoformat(),
    'epochs': epoch_reports, 'expression_weighted_GT_volume_cuts_m3': cuts.tolist(),
    'source_hashes': sources,
    'neural_forwards': 0, 'optimizer_updates': 0, 'remote_queries': 0,
    'full256_coverage_is_logged_scalar_not_independently_recounted': True,
    'endpoint_change_does_not_isolate_candidate_ranking_from_geometry': True,
    'slot_changes_are_not_physical_instance_identity_changes': True,
    'no_raw_gate_or_clamp_causal_claim': True,
    'no_new_metric_or_best_promoted': True, 'full_goal_complete': False,
}
(output / 'RESULT.json').write_bytes((json.dumps(result, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
lines = [
    '# 已结束C实验的已选候选端点变化', '',
    '仅分析保存的9508条逐行输出和已有CPU重算IoU；没有运行模型、更新参数或查询服务器。', '',
    '|轮次|严格最终命中|同Query原回归框|同QueryMask范围框|最终框对Mask修复/破坏|对E0修复/破坏|',
    '|---|---:|---:|---:|---:|---:|',
]
for report in epoch_reports:
    r = report['050']
    lines.append('|E%d|%d|%d|%d|%d/%d|%d/%d|' % (report['epoch'], r['hits'],
        r['native_hits_same_selected_query'], r['mask_endpoint_hits_same_selected_query'],
        r['final_vs_mask_repairs'], r['final_vs_mask_breaks'], r['repairs_vs_E0'], r['breaks_vs_E0']))
lines += ['', '同一行比较的端点属于该轮同一已选Query；跨轮Query和端点均可能变化。',
    '某轮最终框失败而同Query的端点已合格，说明当轮几何融合未保住可用端点；不能据此将全部跨轮损失归因于B、学习率或裁剪。',
    '两个已选端点都不合格且日志显示其他候选合格，也不能单独证明选错物理实例：同实例边界竞争仍可能存在。',
    '四分位以表达行的GT体积定义，重复目标和场景相关性未消除；不是独立样本显著性检验。',
    '新低B学习率正常训练配置保持不变；此分析不用于事后选择推理规则。']
(output / 'REPORT.md').write_bytes(('\n'.join(lines) + '\n').encode('utf-8'))
print(json.dumps({'scope': result['scope'], 'epochs': [{'epoch': r['epoch'], **r['050']} for r in epoch_reports]}, ensure_ascii=False))
