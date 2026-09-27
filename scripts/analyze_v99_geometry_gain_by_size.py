"""Audit protected V99's selected-query geometry gains by GT box volume."""

import gzip
import hashlib
import json

from analyze_v99_geometry_variant_gain import SOURCE, box_iou


def main():
    manifest = json.loads((SOURCE / 'stage_rows_archive.json').read_text())
    compressed = (SOURCE / manifest['archive']).read_bytes()
    assert hashlib.sha256(compressed).hexdigest() == manifest['archive_sha256']
    decoded = gzip.decompress(compressed)
    assert hashlib.sha256(decoded).hexdigest() == manifest['source_sha256']
    rows = json.loads(decoded)['protected_v99']
    assert len(rows) == 9508

    volumes = sorted(
        row['root_box'][3] * row['root_box'][4] * row['root_box'][5]
        for row in rows
    )
    edges = [volumes[len(volumes) * index // 4] for index in (1, 2, 3)]
    groups = {str(index): {
        'count': 0,
        'fused_t0_selected': 0,
        'all': {'raw050': 0, 'final050': 0, 'repairs050': 0, 'damages050': 0},
        'fused_t0': {'raw050': 0, 'final050': 0, 'repairs050': 0, 'damages050': 0},
    } for index in (1, 2, 3, 4)}
    for row in rows:
        volume = row['root_box'][3] * row['root_box'][4] * row['root_box'][5]
        quartile = str(1 + sum(volume >= edge for edge in edges))
        chosen = row['stages']['v99_final']
        position = row['top16_query_indices'].index(chosen['query_index'])
        raw_iou = box_iou(row['top16_boxes'][position], row['root_box'])
        refined_iou = chosen['rec_iou']
        group = groups[quartile]
        group['count'] += 1
        group['fused_t0_selected'] += chosen['variant_index'] == 1
        raw_hit = raw_iou >= 0.5
        final_hit = refined_iou >= 0.5
        for name in ('all', 'fused_t0'):
            if name == 'fused_t0' and chosen['variant_index'] != 1:
                continue
            item = group[name]
            item['raw050'] += raw_hit
            item['final050'] += final_hit
            item['repairs050'] += final_hit and not raw_hit
            item['damages050'] += raw_hit and not final_hit

    assert sum(group['count'] for group in groups.values()) == 9508
    assert sum(group['all']['raw050'] for group in groups.values()) == 4388
    assert sum(group['all']['final050'] for group in groups.values()) == 4795
    assert sum(group['all']['repairs050'] for group in groups.values()) == 463
    assert sum(group['all']['damages050'] for group in groups.values()) == 56
    assert sum(group['fused_t0_selected'] for group in groups.values()) == 6909
    assert sum(group['fused_t0']['repairs050'] for group in groups.values()) == 427
    assert sum(group['fused_t0']['damages050'] for group in groups.values()) == 48
    result = {'volume_quartile_edges_m3': edges, 'groups': groups}
    destination = SOURCE / 'protected_v99_geometry_gain_by_size.json'
    destination.write_bytes((json.dumps(result, indent=2) + '\n').encode())
    print(json.dumps(result))


if __name__ == '__main__':
    main()
