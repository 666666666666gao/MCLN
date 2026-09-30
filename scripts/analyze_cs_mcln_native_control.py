"""Compare the completed, single-seed 21-epoch CS and native controls."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_run(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    source = directory / 'epoch_metrics.csv'
    assert manifest['completed_through_epoch'] == 21
    assert digest(source) == manifest['csv_sha256']
    with source.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row['epoch']) for row in rows] == list(range(22))
    assert len(manifest['sources']) == len(rows)
    for row, entry in zip(rows, manifest['sources']):
        epoch = int(row['epoch'])
        raw_path = directory / entry['file']
        assert digest(raw_path) == entry['sha256'] == row['source_sha256']
        raw = json.loads(raw_path.read_text())
        metrics = raw if epoch == 0 else raw['validation']
        assert int(row['samples']) == metrics['samples'] == 9508
        for suffix in ('025', '050'):
            row['hits' + suffix] = int(row['hits' + suffix])
            assert row['hits' + suffix] == metrics['hits' + suffix]
            value = 100 * row['hits' + suffix] / 9508
            assert abs(value - float(row['acc' + suffix + '_percent'])) < 1e-10
            assert abs(value - 100 * metrics['acc' + suffix]) < 1e-10
            row['acc' + suffix + '_percent'] = value
        row['epoch'] = epoch
    best = max(rows[1:], key=lambda row: (
        min(row['hits025'] / 5572, row['hits050'] / 4797),
        row['hits025'] + row['hits050'],
    ))
    assert best['epoch'] == manifest['best_training_epoch']
    return rows, best, manifest['csv_sha256']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cs', type=Path, required=True)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cs, cs_best, cs_sha = load_run(args.cs)
    native, native_best, native_sha = load_run(args.native)
    assert (cs[0]['hits025'], cs[0]['hits050']) == (5514, 4408)
    assert cs[0]['source_sha256'] == native[0]['source_sha256']
    assert (native[0]['hits025'], native[0]['hits050']) == (5514, 4408)
    args.output.mkdir(parents=True, exist_ok=True)

    paired = []
    for left, right in zip(cs, native):
        paired.append({
            'epoch': left['epoch'],
            'samples': 9508,
            'cs_hits025': left['hits025'], 'cs_hits050': left['hits050'],
            'native_hits025': right['hits025'], 'native_hits050': right['hits050'],
            'cs_minus_native_hits025': left['hits025'] - right['hits025'],
            'cs_minus_native_hits050': left['hits050'] - right['hits050'],
        })
    paired_path = args.output / 'paired_epoch_metrics.csv'
    with paired_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)

    def metrics(row):
        return {key: row[key] for key in (
            'epoch', 'hits025', 'hits050', 'acc025_percent', 'acc050_percent'
        )}

    summary = {
        'protocol': 'ScanRefer full 9508; native last/bbs; E71; seed2027; batch12; 21 epochs',
        'cs_csv_sha256': cs_sha, 'native_csv_sha256': native_sha,
        'paired_csv_sha256': digest(paired_path),
        'analysis_script_sha256': digest(Path(__file__)),
        'initialization': metrics(cs[0]),
        'cs_training_best': metrics(cs_best),
        'native_training_best': metrics(native_best),
        'cs_fixed_endpoint': metrics(cs[21]),
        'native_fixed_endpoint': metrics(native[21]),
        'endpoint_cs_minus_native_hits025': paired[21]['cs_minus_native_hits025'],
        'endpoint_cs_minus_native_hits050': paired[21]['cs_minus_native_hits050'],
        'best_cs_minus_native_hits025': cs_best['hits025'] - native_best['hits025'],
        'best_cs_minus_native_hits050': cs_best['hits050'] - native_best['hits050'],
        'cs_gate_shortfall_hits025': 5544 - cs_best['hits025'],
        'cs_gate_shortfall_hits050': 4754 - cs_best['hits050'],
        'cs_above_native_epochs025': [row['epoch'] for row in paired[1:]
                                     if row['cs_minus_native_hits025'] > 0],
        'cs_above_native_epochs050': [row['epoch'] for row in paired[1:]
                                     if row['cs_minus_native_hits050'] > 0],
        'note': 'Epochs are dependent snapshots of two single-seed runs, not independent replicates. No seed-variance or module-causal claim.',
    }
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')

    colors = {'CS M1/M2/M3': '#CC6E27', 'Native MCLN': '#2564A4'}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'svg.fonttype': 'none'})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True,
                             gridspec_kw={'height_ratios': [1.35, 1]})
    epochs = list(range(22))
    for col, suffix in enumerate(('025', '050')):
        top, bottom = axes[0, col], axes[1, col]
        for label, rows, best in (
            ('Native MCLN', native, native_best),
            ('CS M1/M2/M3', cs, cs_best),
        ):
            values = [row['acc' + suffix + '_percent'] for row in rows]
            top.plot(epochs, values, color=colors[label], label=label,
                     linewidth=1.7, marker='.', markersize=5)
            top.scatter(best['epoch'], best['acc' + suffix + '_percent'],
                        color=colors[label], s=80, facecolors='none', linewidths=1.8)
            top.scatter(21, values[21], color=colors[label], s=36, marker='s')
        initial = cs[0]['acc' + suffix + '_percent']
        top.axhline(initial, color='#3B3B3B', linestyle='--', linewidth=1,
                    label='E71 zero-update')
        top.set_title('REC Acc@' + ('0.25' if suffix == '025' else '0.50'))
        top.set_ylabel('Accuracy (%)')
        deltas = [row['cs_minus_native_hits' + suffix] for row in paired]
        bottom.axhline(0, color='#3B3B3B', linewidth=1)
        bottom.plot(epochs, deltas, color='#7755A4', linewidth=1.7,
                    marker='.', markersize=5)
        bottom.set_ylabel('CS minus native (hits)')
        bottom.set_xlabel('Completed epoch')
        for ax in (top, bottom):
            ax.grid(axis='y', alpha=0.2)
            ax.spines[['top', 'right']].set_visible(False)
            ax.set_xlim(-0.3, 21.5)
            ax.set_xticks([0, 3, 6, 9, 12, 15, 18, 21])
    axes[0, 0].legend(loc='lower right', fontsize=8)
    fig.suptitle('Completed CS and native ScanRefer controls: 21 epochs, one seed', fontsize=14)
    fig.text(0.5, 0.015,
             '9508 validation expressions; circles = predefined training-best, squares = epoch 21. '
             'Development gate: 58.3% / 50.0% (outside the upper plots).',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, 0.045, 1, 0.96))
    for extension in ('png', 'svg'):
        fig.savefig(args.output / ('training_curves.' + extension), dpi=180)
    plt.close(fig)
    svg = args.output / 'training_curves.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines()) + '\n',
                   encoding='utf-8')

    report = [
        '# Completed CS and native ScanRefer controls', '',
        'Both runs use E71 initialization, seed2027, batch12, 21 epochs and full9508 native `last/bbs` validation. Epoch0 is excluded from saved training-best ranking.', '',
        '| Epoch | Native hits @.25 / @.50 | CS hits @.25 / @.50 | CS minus native |',
        '|---:|---:|---:|---:|',
    ]
    for row in paired:
        report.append(f"| {row['epoch']} | {row['native_hits025']} / {row['native_hits050']} | "
                      f"{row['cs_hits025']} / {row['cs_hits050']} | "
                      f"{row['cs_minus_native_hits025']:+d} / {row['cs_minus_native_hits050']:+d} |")
    report += [
        '', '1. At the fixed epoch21 endpoint, CS gains23/40 hits over native; CS itself remains49/4 hits below E71. This supports an endpoint advantage over this continuation control, without proving any individual module contribution.',
        '2. The predefined CS best is epoch15 (5466/4409); native best is epoch8 (5441/4410). Comparing these saved artifacts gives+25/-1 hits. Neither jointly surpasses E71 (5514/4408).',
        '3. CS trails native at @.50 in epochs1-8 and17-18, and leads in epochs9-16 and19-21. These dependent checkpoints are not independent trials; do not sum their differences or infer cross-seed significance.',
        '4. CS best remains78/345 hits below the fixed ScanRefer gate5544/4754. Complete the running full-validation support diagnostic before choosing the next structural experiment.',
        '', '![Training curves](training_curves.png)', '',
        f'Input CSV SHA256: CS `{cs_sha}`; native `{native_sha}`. Every raw epoch JSON was rechecked against its manifest and CSV before analysis.', '',
        'No R/readback or fused-Mask-tail training result is included. V99 system metrics are a separate output-path reference, not a result of these runs.',
    ]
    (args.output / 'comparison.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
