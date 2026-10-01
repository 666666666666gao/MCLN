import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
plt.rcParams.update({
    'font.family': 'DejaVu Serif',
    'font.size': 10,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'svg.fonttype': 'none',
    'svg.hashsalt': 'mcln-readback-epoch8-20261001',
    'pdf.fonttype': 42,
})


def draw_curve(hit_column, ylabel, name):
    with (HERE / 'epoch0_8_counts.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    series = {
        method: [row for row in rows if row['method'] == method]
        for method in ('CS+R', 'CS historical', 'native historical')
    }
    for points in series.values():
        assert [int(row['epoch']) for row in points] == list(range(9))
        assert all(int(row['samples']) == 9508 for row in points)
    starts = [int(points[0][hit_column]) for points in series.values()]
    assert starts[0] == starts[1] == starts[2]

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.17, top=0.77)
    styles = (
        ('CS+R', '#0072B2', '-', 'o', 3),
        ('CS historical', '#D55E00', '--', '^', 2),
        ('native historical', '#4C4C4C', ':', 's', 2),
    )
    all_values = []
    for method, color, line, marker, order in styles:
        points = series[method]
        values = [100 * int(row[hit_column]) / int(row['samples']) for row in points]
        all_values.extend(values)
        ax.plot(range(9), values, label=method, color=color, linestyle=line,
                marker=marker, markersize=4, linewidth=1.5, zorder=order)
    ax.axhline(100 * starts[0] / int(series['CS+R'][0]['samples']),
               label='E71 start', color='#777777',
               linestyle=(0, (5, 3)), linewidth=1, zorder=1)
    ax.set_xlim(-0.15, 8.15)
    ax.set_ylim(math.floor(min(all_values) * 10) / 10 - 0.2,
                math.ceil(max(all_values) * 10) / 10 + 0.2)
    ax.set_xticks(range(9))
    ax.set_xlabel('Completed training epoch')
    ax.set_ylabel(ylabel)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.29), ncol=2,
              frameon=False, columnspacing=1.6, handlelength=2.3)
    fig.savefig(HERE / (name + '.pdf'),
                metadata={'CreationDate': None, 'ModDate': None})
    fig.savefig(HERE / (name + '.svg'), metadata={'Date': None})
    fig.savefig(HERE / (name + '.png'))
    plt.close(fig)
