"""Summarize the actual closed native-bbs pair without inference or weight loading."""
import csv
import datetime
import hashlib
import json
from pathlib import Path

local = Path(__file__).resolve().parent
complete = local / 'complete'
intake = json.loads((complete / 'INTAKE.json').read_bytes())
terminal = intake['remote_terminal']
assert terminal['status']['status'] == 'complete' and terminal['exitcode'] == 0
assert not terminal['controller_alive'] and intake['downloaded_weights'] == 0
assert not intake['inference_or_optimizer_replayed']
analysis = local / 'analysis'
assert not analysis.exists()


def read_json(relative):
    path = complete / relative
    raw = path.read_bytes()
    identity = intake['files'][relative]
    assert len(raw) == identity['bytes'] and hashlib.sha256(raw).hexdigest() == identity['sha256']
    return json.loads(raw)


def read_rows(relative):
    path = complete / relative
    raw = path.read_bytes()
    identity = intake['files'][relative]
    assert len(raw) == identity['bytes'] and hashlib.sha256(raw).hexdigest() == identity['sha256']
    return [json.loads(line) for line in raw.decode('utf-8').splitlines()]


def paired(before, after, threshold):
    assert len(before) == len(after)
    fixes = damages = changed = 0
    coverable = missing = 0
    for old, new in zip(before, after):
        for key in ('row_id', 'scan_id', 'target_id', 'root_box', 'point_sha256'):
            assert old[key] == new[key]
        prior = old['bbs']['iou'] > threshold
        current = new['bbs']['iou'] > threshold
        changed += old['bbs']['query'] != new['bbs']['query']
        fixes += not prior and current
        damages += prior and not current
        if not prior and current:
            available = old['bbs']['oracle25' if threshold == .25 else 'oracle50'][-1]
            coverable += bool(available)
            missing += not available
    return dict(repairs=fixes, damages=damages, net=fixes-damages,
                selected_query_changes=changed, repairs_from_coverable_errors=coverable,
                repairs_from_missing_candidate_errors=missing)


def summarize(rows, receipt):
    metrics = receipt['metrics']['bbs']
    hits25 = sum(row['bbs']['iou'] > .25 for row in rows)
    hits50 = sum(row['bbs']['iou'] > .5 for row in rows)
    assert hits25 == metrics['rec_hits25'] and hits50 == metrics['rec_hits50']
    assert all(row['same_forward_geometry_exact'] and row['native_head_calls'] == 1
               and row['diagnostic_native_head_replay_calls'] == 1 for row in rows)
    direct = {}
    coverage = {}
    for threshold in (.25, .5):
        key = str(threshold)
        before = [dict(row, bbs=dict(row['bbs'], query=row['bypass_fixed_frame']['query'],
                  iou=row['bypass_fixed_frame']['iou'])) for row in rows]
        direct[key] = paired(before, rows, threshold)
        for field, count in (('fixes', direct[key]['repairs']), ('damages', direct[key]['damages']),
                             ('net', direct[key]['net'])):
            assert count == receipt['fixed_frame_readback_effect'][key][field]
        oracle_key = 'oracle25' if threshold == .25 else 'oracle50'
        oracles = [sum(row['bbs'][oracle_key][index] for row in rows) for index in range(4)]
        coverage[key] = dict(topk=[16,32,64,256], oracle_hits=oracles,
            errors_with_good_full256=sum(row['bbs']['iou'] <= threshold and row['bbs'][oracle_key][-1] for row in rows),
            errors_without_good_full256=sum(row['bbs']['iou'] <= threshold and not row['bbs'][oracle_key][-1] for row in rows))
    return dict(rows=len(rows), rec_hits25=hits25, rec_hits50=hits50,
        rec_acc25=100*hits25/len(rows), rec_acc50=100*hits50/len(rows),
        mask_hits25=metrics['mask_hits25'], mask_hits50=metrics['mask_hits50'], mask_miou=metrics['mask_miou'],
        fixed_frame_readback_effect=direct, candidate_availability=coverage)


arms = {}
rows_by_arm = {}
stages = {}
training_orders = {}
for arm in ('evidence_hidden','evidence_visible'):
    fit = read_json(arm + '/receipt.json')
    restore = read_json(arm + '/formal_restore.json')
    retention = read_json(arm + '/weight_retention.json')
    spec = read_json(arm + '/spec.json')
    assert fit['status'] == 'complete' and fit['training_steps'] == 3723 and fit['fit_rows'] == 29778
    assert fit['frozen_parent_states_exact'] and fit['fit_seen_exactly_once']
    assert restore['status'] == 'pass' and restore['strict_model_restore'] and restore['restored_steps'] == 3723
    assert retention['required_parent_chain_preserved'] and retention['cpu_box_threshold_changes'] == 0
    assert not retention['local_weight_archive_created']
    train = read_rows(arm + '/train.jsonl')
    assert len(train) == 3723 and [row['step'] for row in train] == list(range(1,3724))
    assert all(len(row['rows']) == (2 if row['step'] == 3723 else 8) for row in train)
    training_orders[arm] = [sample for row in train for sample in row['rows']]
    assert len(training_orders[arm]) == len(set(training_orders[arm])) == 29778
    stages[arm] = {}
    for stage, expected in (('initial',6887),('terminal',6887),('formal',9508)):
        rows = read_rows(arm + '/' + stage + '/rows.jsonl')
        receipt = read_json(arm + '/' + stage + '/receipt.json')
        assert receipt['status'] == 'pass' and receipt['rows'] == expected == len(rows)
        stages[arm][stage] = summarize(rows, receipt)
        rows_by_arm[arm + '/' + stage] = rows
    assert [row['row_id'] for row in rows_by_arm[arm + '/formal']] == list(range(9508))
    arms[arm] = dict(fit=fit, restore=restore, retention=retention,
        use_geometry_evidence=spec['use_geometry_evidence'])
assert training_orders['evidence_hidden'] == training_orders['evidence_visible']
comparison = {}
for stage in ('initial','terminal','formal'):
    before = rows_by_arm['evidence_hidden/' + stage]
    after = rows_by_arm['evidence_visible/' + stage]
    comparison[stage] = {str(threshold):paired(before,after,threshold) for threshold in (.25,.5)}
    comparison[stage]['full256_oracle_row_disagreements'] = {
        str(threshold):sum(a['bbs'][key][-1] != b['bbs'][key][-1] for a,b in zip(before,after))
        for threshold,key in ((.25,'oracle25'),(.5,'oracle50'))}
parent_dir = local.parent.parent / 'pvground_boundary_distribution_20261004/complete/distribution/formal'
parent_raw = (parent_dir / 'rows.jsonl').read_bytes()
parent = [json.loads(line) for line in parent_raw.decode('utf-8').splitlines()]
assert len(parent) == 9508
assert sum(row['bbs']['iou'] > .25 for row in parent) == 5616
assert sum(row['bbs']['iou'] > .5 for row in parent) == 4506
parent_comparisons = {
    arm:{str(threshold):paired(parent,rows_by_arm[arm+'/formal'],threshold) for threshold in (.25,.5)}
    for arm in arms}
formal_hidden = rows_by_arm['evidence_hidden/formal']
formal_visible = rows_by_arm['evidence_visible/formal']
volume_order = sorted(range(9508), key=lambda i:(parent[i]['root_box'][3]*parent[i]['root_box'][4]*parent[i]['root_box'][5],i))
groups = []
for quartile in range(4):
    indices = volume_order[quartile*2377:(quartile+1)*2377]
    groups.append(dict(volume_quartile=quartile+1, rows=len(indices),
        hidden_hits50=sum(formal_hidden[i]['bbs']['iou'] > .5 for i in indices),
        visible_hits50=sum(formal_visible[i]['bbs']['iou'] > .5 for i in indices),
        visible_vs_hidden=paired([formal_hidden[i] for i in indices],[formal_visible[i] for i in indices],.5)))
table = [dict(system='frozen4506_geometry_parent',rec_hits25=5616,rec_hits50=4506,
              rec_acc25=100*5616/9508,rec_acc50=100*4506/9508)]
for arm in arms:
    metric = stages[arm]['formal']
    table.append(dict(system=arm, **{key:metric[key] for key in ('rec_hits25','rec_hits50','rec_acc25','rec_acc50')}))
summary = dict(time_cst=datetime.datetime.now().astimezone().isoformat(), primary='native last/bbs Acc@0.50',
    table=table, stages=stages, visible_vs_hidden=comparison, versus_protected_parent=parent_comparisons,
    formal_gt_volume_groups=groups, training_order_exact=True, updates_per_arm=3723, effective_batch=8,
    samples_per_arm=29778, fit_seen_once=True, retained_best=terminal['status']['retained_best'],
    actual_finished_cst=terminal['status']['finished_cst'],
    parent_rows_sha256=hashlib.sha256(parent_raw).hexdigest(),
    target_pass=terminal['status']['scanrefer_target_pass'], integrity_review_pending=True,
    scope='Single seed2027; actual native9508 formal;6887 pretrained-seen holdout separately. Fixed-frame replay is a forward diagnostic, not an independently trained ablation. Frozen parent states are exact; independent complete forward floating outputs are not claimed bitwise identical. GT coverage and volume groups are offline diagnosis only. No Nr/Sr results.',
    downloaded_weights=0, inference_or_optimizer_replayed=False)
analysis.mkdir()
(analysis/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
with (analysis/'FORMAL_METRICS.csv').open('w',newline='',encoding='utf-8') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(table[0]))
    writer.writeheader()
    writer.writerows(table)
lines=['# 固定4506几何的回读对照：实际终态','',
       f"实际结束：{summary['actual_finished_cst']}；主指标为原生last/bbs Acc@0.50。",'',
       '| 模型 | @0.25命中 | @0.50命中 | @0.25% | @0.50% |',
       '|---|---:|---:|---:|---:|']
for row in table:
    lines.append(f"| {row['system']} | {row['rec_hits25']} | {row['rec_hits50']} | {row['rec_acc25']:.4f} | {row['rec_acc50']:.4f} |")
lines+=['','## 修复与破坏','']
for arm in arms:
    direct=stages[arm]['formal']['fixed_frame_readback_effect']['0.5']
    parent_effect=parent_comparisons[arm]['0.5']
    lines.append(f"- {arm}：同帧R前/后修复{direct['repairs']}、破坏{direct['damages']}、净{direct['net']:+d}；相对历史保护父模型修复{parent_effect['repairs']}、破坏{parent_effect['damages']}、净{parent_effect['net']:+d}。两种比较分别解释。")
effect=comparison['formal']['0.5']
lines.append(f"- 可见对隐藏：修复{effect['repairs']}、破坏{effect['damages']}、净{effect['net']:+d}。")
lines+=['','两组各29778条fit一次、有效batch8、3723更新，实际训练行顺序相同；父权重状态保持冻结。',
        '6887条模块留出不是未见场景正式验证。GT候选上界及体积分组仅用于离线诊断。',
        '同帧语义头回放不等于独立训练消融；不把独立完整前向浮点输出描述为逐位相同。',
        '终态完整性审查尚待实际新鲜审查，本分析不是独立审查PASS。',
        '只有ScanRefer、单seed；未证明Nr3D/Sr3D收益或三个论文创新点。','']
(analysis/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps(dict(status='ACTUAL_CLOSED_ROWS_ANALYSED',table=table,target_pass=summary['target_pass'],
    integrity_review_pending=True,downloaded_weights=0)),flush=True)
