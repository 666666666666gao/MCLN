"""Summarize the archived CS validation replay without changing formal results."""

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def transition(rows, before, after, threshold):
    old = [row[before] > threshold for row in rows]
    new = [row[after] > threshold for row in rows]
    result = {'before': sum(old), 'after': sum(new),
              'repairs': sum(not a and b for a, b in zip(old, new)),
              'damages': sum(a and not b for a, b in zip(old, new))}
    result['net'] = result['after'] - result['before']
    assert result['net'] == result['repairs'] - result['damages']
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    opt = parser.parse_args()
    receipt = json.loads((opt.source / 'provenance.json').read_text())
    packed = (opt.source / receipt['archive']).read_bytes()
    assert hashlib.sha256(packed).hexdigest() == receipt['archive_sha256']
    data = gzip.decompress(packed)
    assert hashlib.sha256(data).hexdigest() == receipt['source_sha256']
    result = json.loads(data)
    assert result['schema'] == 'cs-mcln-m3-full-scanrefer-validation-v5'
    assert result['sample_count'] == 9508 and result['optimizer_steps'] == 0
    assert result['checkpoint_epoch'] == 15
    assert result['checkpoint_sha256'] == receipt['checkpoint_sha256']
    assert result['batch_size'] == result['checkpoint_formal_batch_size'] == 12
    assert receipt['exit_code'] == 1 and not receipt['accepted_as_exact_historical_reproduction']
    assert result['formal_result_match'] == {'hits025': False, 'hits050': True}
    rows = result['rows']
    assert [row['row_id'] for row in rows] == list(range(9508))
    valid = [row for row in rows if row['fused_exact_valid']]
    assert len(valid) == result['summary']['fused_exact_valid_rows'] == 9477
    values = {}
    for threshold, suffix in ((0.25, '025'), (0.5, '050')):
        m3 = transition(rows, 'pre_selected_iou', 'post_selected_iou', threshold)
        fused = transition(valid, 'post_selected_iou', 'fused_exact_selected_iou', threshold)
        for name, expected in (
            ('selected_pre_hits', m3['before']), ('selected_post_hits', m3['after']),
            ('same_query_repairs', m3['repairs']), ('same_query_damages', m3['damages']),
            ('post_hits_on_fused_valid', fused['before']), ('fused_exact_hits', fused['after']),
            ('post_to_fused_exact_repairs', fused['repairs']),
            ('post_to_fused_exact_damages', fused['damages']),
        ):
            assert result['summary'][name + suffix] == expected
        misses = [row for row in rows if row['post_selected_iou'] <= threshold]
        ranks = [row['post_first_qualified_rank' + suffix] for row in misses]
        partition = {'rank2to16': sum(rank is not None and 2 <= rank <= 16 for rank in ranks),
                     'rank17to64': sum(rank is not None and 17 <= rank <= 64 for rank in ranks),
                     'rank65to256': sum(rank is not None and 65 <= rank <= 256 for rank in ranks),
                     'no_qualified': sum(rank is None for rank in ranks)}
        assert sum(partition.values()) == len(misses)
        oracle = sum(row['post_raw256_oracle_iou'] > threshold for row in rows)
        assert oracle == result['summary']['raw256_post_oracle_hits' + suffix]
        assert oracle - m3['after'] == len(misses) - partition['no_qualified']
        values[suffix] = {'m3': m3, 'fused_valid': fused, 'errors': len(misses),
                          'error_partition': partition, 'oracle_hits': oracle,
                          'qualified_not_selected_fraction': (oracle - m3['after']) / len(misses)}
    volumes = sorted(row['root_box_volume_m3'] for row in rows)
    edges = [volumes[len(volumes) * index // 4] for index in (1, 2, 3)]
    quartiles = []
    for quartile in range(4):
        group = [row for row in rows if sum(row['root_box_volume_m3'] >= edge for edge in edges) == quartile]
        group_valid = [row for row in group if row['fused_exact_valid']]
        item = {'quartile': quartile + 1, 'all_rows': len(group), 'fused_valid_rows': len(group_valid)}
        for threshold, suffix in ((0.25, '025'), (0.5, '050')):
            item['fused' + suffix] = transition(group_valid, 'post_selected_iou', 'fused_exact_selected_iou', threshold)
        quartiles.append(item)
    assert sum(item['all_rows'] for item in quartiles) == 9508
    for suffix in ('025', '050'):
        assert sum(item['fused' + suffix]['net'] for item in quartiles) == values[suffix]['fused_valid']['net']
    audited = {'status': 'independent_replay_with_unresolved_historical_parity',
               'accepted_as_exact_historical_reproduction': False,
               'formal_hits': result['checkpoint_formal_metrics'], 'thresholds': values,
               'volume_quartile_edges_m3': edges, 'volume_quartiles': quartiles,
               'source_sha256': receipt['source_sha256'],
               'result_written_at_cst': receipt['result_written_at_cst'],
               'exit_written_at_cst': receipt['exit_written_at_cst']}
    (opt.source / 'audited_summary.json').write_text(json.dumps(audited, indent=2) + '\n')
    with (opt.source / 'fused_geometry_by_volume.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('quartile', 'all_rows', 'valid_rows', 'repairs025', 'damages025', 'net025', 'repairs050', 'damages050', 'net050'))
        for item in quartiles:
            writer.writerow((item['quartile'], item['all_rows'], item['fused_valid_rows'],
                             *[item['fused' + suffix][name] for suffix in ('025', '050') for name in ('repairs', 'damages', 'net')]))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    xs = [0, 1]
    axes[0].bar([x - .18 for x in xs], [values[s]['m3']['before'] for s in ('025', '050')], .36, label='Before final M3', color='#8c9bad')
    axes[0].bar([x + .18 for x in xs], [values[s]['m3']['after'] for s in ('025', '050')], .36, label='After final M3', color='#2864aa')
    axes[0].set_title('Same selected query, last M3 only')
    axes[0].set_ylabel('Hits / 9508 replay rows')
    axes[0].legend(fontsize=8)
    bottom = [0, 0]
    for name, label, color in (
        ('rank2to16', 'Good candidate rank 2-16', '#386cb0'),
        ('rank17to64', 'Good candidate rank 17-64', '#7fc97f'),
        ('rank65to256', 'Good candidate rank 65-256', '#fdc086'),
        ('no_qualified', 'No qualified candidate', '#be4c4c'),
    ):
        heights = [values[s]['error_partition'][name] for s in ('025', '050')]
        axes[1].bar(xs, heights, .6, bottom=bottom, label=label, color=color)
        bottom = [a + b for a, b in zip(bottom, heights)]
    axes[1].set_title('Final selected-query failures')
    axes[1].set_ylabel('Number of errors; oracle needs GT')
    axes[1].legend(fontsize=7)
    axes[2].bar([x - .18 for x in xs], [values[s]['fused_valid']['repairs'] for s in ('025', '050')], .36, color='#3b9162', label='Repairs')
    axes[2].bar([x + .18 for x in xs], [-values[s]['fused_valid']['damages'] for s in ('025', '050')], .36, color='#be4c4c', label='Damages')
    axes[2].axhline(0, color='#333333', linewidth=.7)
    axes[2].set_title('Hard fused-Mask box counterfactual')
    axes[2].set_ylabel('Same query; 9477 valid rows only')
    axes[2].legend(fontsize=8)
    for axis in axes:
        axis.set_xticks(xs, ('IoU > 0.25', 'IoU > 0.50'))
        axis.grid(axis='y', alpha=.15)
        axis.set_axisbelow(True)
    fig.suptitle('Independent CS e15 replay: historical @0.25 differs by +1 hit; formal result unchanged', fontsize=10)
    fig.tight_layout()
    fig.savefig(opt.source / 'validation_replay.png', dpi=180)
    svg = opt.source / 'validation_replay.svg'
    fig.savefig(svg)
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines()) + '\n', encoding='utf-8')
    plt.close(fig)
    table = '\n'.join(f"| Q{item['quartile']} | {item['all_rows']} | {item['fused_valid_rows']} | {item['fused025']['repairs']}/{item['fused025']['damages']} | {item['fused025']['net']:+d} | {item['fused050']['repairs']}/{item['fused050']['damages']} | {item['fused050']['net']:+d} |" for item in quartiles)
    report = f'''# CS第15轮best：完整验证独立复核诊断

**正式复现验收尚未通过。** batch12复核的9508条结果为5467/4409，checkpoint存档为5466/4409；@0.25多1个、@0.50一致。原断言未放宽，退出码1。正式best仍为57.4884%/46.3715%，本次差异不计作收益。结果写于{receipt['result_written_at_cst']}，退出标记写于{receipt['exit_written_at_cst']}。

本目录压缩归档保留全部原始逐行数据；解压字节SHA **{receipt['source_sha256']}**。本分析重算逐行命中、修复/破坏、错误分组及几何反事实，均与GPU输出摘要一致。checkpoint SHA **{result['checkpoint_sha256']}**，无优化更新。历史逐行输出未保存，所以不能仅凭总命中差识别是哪条表达改变，更不能将最接近0.25的当前样本直接认定为历史差异样本。

| 最后一次M3，同一已选Query，9508条 | 精修前 | 精修后 | 修复/破坏 | 净变化 |
|---|---:|---:|---:|---:|
| @0.25 | 4623 | 5467 | 885/41 | +844 |
| @0.50 | 1283 | 4409 | 3147/21 | +3126 |

这是共同训练后的内部中间框对最终框的比较；前框已可能受到前一层M3影响，不是native模型，也不是整个M3的独立训练消融。不能将3126写成相对baseline新增命中。它反驳了“当前验证中最后M3主要修坏框”的假设。

| 精修后候选与实际选择 | @0.25 | @0.50 |
|---|---:|---:|
| 实际已选Query | 5467 | 4409 |
| 全256框GT几何上界 | 8995 | 7864 |
| 有合格框但没选中的错误 | 3528/4041＝87.30% | 3455/5099＝67.76% |
| 全256仍缺合格框的错误 | 513 | 1644 |

候选上界只用于离线GT诊断，不可部署；它还不能完全区分选错物体与同实例边界较差。当前结构中的几何证据回读与原生候选判别值得优先验证。

| 同一Query，融合Mask零logit真实点AABB反事实，仅9477条有效 | 原生最终框命中 | 硬融合Mask框命中 | 修复/破坏 | 净变化 |
|---|---:|---:|---:|---:|
| @0.25 | 5463 | 5410 | 141/194 | −53 |
| @0.50 | 4406 | 4403 | 490/493 | −3 |

31条没有有效硬Mask框，不用原框补齐。上表两边使用同一有效子集，不能与9508条准确率直接相减。旧V99固定所选Query后的+379没有在当前CS直接换框反事实中复现；这不能证明可学习的融合支撑没有价值，但不足以优先将所有最终框换成硬融合Mask框。

| GT体积组，仅离线分组 | 全组条数 | 有效融合框条数 | @0.25修复/破坏 | 净变化 | @0.50修复/破坏 | 净变化 |
|---|---:|---:|---:|---:|---:|---:|
{table}

体积分组使用本次GT体积四分位边界，相同体积保持同组，组大小不必相等；不是推理规则。超点中心上的M3概率质量位于GT框外的中位数为13.9473%，融合Mask为13.6543%；框外质量不等于真实实例背景率，且中位数不代表所有样本。最终Query与内部M3 logit的最大差中位数为0.007288，不能视为完全相同张量。

![完整验证独立复核诊断](validation_replay.png)

固定原batch重复前向已于2026-09-30 12:19:10 CST结束，见[数值稳定性探针](stability_analysis.md)。原M1成员聚合下相同输入的输出会变化；仅在探针中改用现有确定性聚合后，该批次三次输出逐元素完全一致。这不证明全部历史差异的唯一来源，也不改变正式best。结构方面优先真实GPU预检已有R回读草稿，保留现有几何计算和单一语义输出；暂不同时加入质量目标、G、教师或融合尾部。后续训练启动仍要满足实际容量、梯度及磁盘保存条件。当前数据不是未训练R或未来融合尾部的成绩。
'''
    (opt.source / 'analysis.md').write_text(report, encoding='utf-8')
    print(json.dumps(audited, indent=2))


if __name__ == '__main__':
    main()
