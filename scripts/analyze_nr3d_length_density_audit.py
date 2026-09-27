"""Cross-tabulate the sealed Nr3D candidate audit by expression length and target points."""

import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path


AUDIT = (Path(__file__).resolve().parents[1]
         / 'refine-logs/official_candidate_audit_20260905_v1')
EXPECTED_ROWS_SHA256 = 'd99a504828876eb18d8fe77de3b901dc3ce374e58363ba662e317871fe6d5767'


def length_group(count):
    if count <= 8:
        return '2-8'
    if count <= 12:
        return '9-12'
    return '13+'


def point_group(count):
    if count <= 32:
        return '<=32'
    if count <= 227:
        return '33-227'
    return '>227'


def main():
    metadata = json.loads((AUDIT / 'enrichment.json').read_text(
        encoding='utf-8'))['metadata']
    with gzip.open(AUDIT / 'rows.jsonl.gz', 'rb') as stream:
        raw = stream.read()
    assert hashlib.sha256(raw).hexdigest() == EXPECTED_ROWS_SHA256
    rows = [json.loads(line) for line in raw.splitlines()]
    assert len(rows) == len(metadata) == 7899

    groups = defaultdict(lambda: defaultdict(int))
    overall = defaultdict(int)
    for row, meta in zip(rows, metadata):
        assert row['id'] == meta['id']
        key = (length_group(meta['csv_token_count']),
               point_group(row['root_target_input_points']))
        selected = (row['rec_selection']['box_iou']
                    if row['rec_selection'] is not None else 0.0)
        oracle = row['score_profiles']['default']['after_filter']['top_256']
        flags = {
            'n': 1,
            'selected025': selected > 0.25,
            'selected050': selected > 0.5,
            'oracle025': oracle['hit025'],
            'oracle050': oracle['hit050'],
            'qualified_not_selected025': selected <= 0.25 and oracle['hit025'],
            'no_qualified_candidate025': not oracle['hit025'],
        }
        for name, value in flags.items():
            groups[key][name] += int(value)
            overall[name] += int(value)

    assert dict(overall) == {
        'n': 7899,
        'selected025': 4478,
        'selected050': 3763,
        'oracle025': 7473,
        'oracle050': 6823,
        'qualified_not_selected025': 2995,
        'no_qualified_candidate025': 426,
    }
    cells = []
    for length in ('2-8', '9-12', '13+'):
        for points in ('<=32', '33-227', '>227'):
            counts = dict(groups[(length, points)])
            n = counts['n']
            cells.append({
                'csv_token_group': length,
                'target_input_point_group': points,
                **counts,
                'selected025_percent': 100 * counts['selected025'] / n,
                'selected050_percent': 100 * counts['selected050'] / n,
                'oracle025_percent': 100 * counts['oracle025'] / n,
                'oracle050_percent': 100 * counts['oracle050'] / n,
            })
    print(json.dumps({
        'schema': 'nr3d-length-target-points-cross-tab-v1',
        'source_rows_sha256': EXPECTED_ROWS_SHA256,
        'source': 'sealed current-source native audit; no new model forward',
        'overall': dict(overall),
        'cells': cells,
    }, indent=2))


if __name__ == '__main__':
    main()
