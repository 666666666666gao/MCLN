"""Break down protected V99's selected-query box change by geometry variant."""

import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / 'refine-logs/scanrefer_stage_diagnostic_20260907_v1'
          / 'diagnostic_result')


def box_iou(first, second):
    first_lo = [first[d] - max(first[d + 3], 1e-6) / 2 for d in range(3)]
    first_hi = [first[d] + max(first[d + 3], 1e-6) / 2 for d in range(3)]
    second_lo = [second[d] - max(second[d + 3], 1e-6) / 2 for d in range(3)]
    second_hi = [second[d] + max(second[d + 3], 1e-6) / 2 for d in range(3)]
    overlap = 1.0
    for d in range(3):
        overlap *= max(0.0, min(first_hi[d], second_hi[d])
                       - max(first_lo[d], second_lo[d]))
    first_volume = first[3] * first[4] * first[5]
    second_volume = second[3] * second[4] * second[5]
    return overlap / (first_volume + second_volume - overlap)


def main():
    manifest = json.loads((SOURCE / 'stage_rows_archive.json').read_text())
    compressed = (SOURCE / manifest['archive']).read_bytes()
    assert hashlib.sha256(compressed).hexdigest() == manifest['archive_sha256']
    decoded = gzip.decompress(compressed)
    assert hashlib.sha256(decoded).hexdigest() == manifest['source_sha256']
    rows = json.loads(decoded)['protected_v99']
    assert len(rows) == 9508
    summary = json.loads((SOURCE / 'evidence_breakdown.json').read_text())[
        'arms']['protected_v99']['v99_final']

    result = {str(variant): {
        'count': 0,
        '025': {'raw_hits': 0, 'variant_hits': 0, 'repairs': 0, 'damages': 0},
        '050': {'raw_hits': 0, 'variant_hits': 0, 'repairs': 0, 'damages': 0},
    } for variant in range(7)}
    for row in rows:
        selected = row['stages']['v99_final']
        variant = str(selected['variant_index'])
        position = row['top16_query_indices'].index(selected['query_index'])
        raw_iou = box_iou(row['top16_boxes'][position], row['root_box'])
        refined_iou = selected['rec_iou']
        group = result[variant]
        group['count'] += 1
        for key, threshold in (('025', 0.25), ('050', 0.5)):
            raw_hit = raw_iou >= threshold
            variant_hit = refined_iou >= threshold
            item = group[key]
            item['raw_hits'] += raw_hit
            item['variant_hits'] += variant_hit
            item['repairs'] += variant_hit and not raw_hit
            item['damages'] += raw_hit and not variant_hit

    assert sum(group['count'] for group in result.values()) == 9508
    for key in ('025', '050'):
        assert sum(group[key]['raw_hits'] for group in result.values()) == (
            summary['raw_box_of_selected_query_hits']['hits' + key]
        )
        assert sum(group[key]['variant_hits'] for group in result.values()) == (
            summary['actual_selected_variant_hits']['hits' + key]
        )
        assert sum(group[key]['repairs'] for group in result.values()) == (
            summary['same_selected_query_raw_to_variant_box'][key]['repairs']
        )
        assert sum(group[key]['damages'] for group in result.values()) == (
            summary['same_selected_query_raw_to_variant_box'][key]['damage']
        )
    output = SOURCE / 'protected_v99_variant_gain.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
