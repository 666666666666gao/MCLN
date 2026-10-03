"""Count GT box/mask agreement for the same selected query in the fixed cache."""
import hashlib
import json
from pathlib import Path


def main():
    source = Path(r'C:\Users\gb\.codex\tmp\pvground_fused_support_20261002\candidate_audit\full_gt_scope_v2')
    output = Path(__file__).parent/'cached_p3_selected_box_mask.json'
    assert not output.exists()
    receipt = json.loads((source/'receipt.json').read_bytes())
    raw = (source/'rows.jsonl').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == receipt['rows_sha256']
    rows = [json.loads(line) for line in raw.splitlines()]
    assert len(rows) == receipt['rows'] == 9508
    labels = ('root_nearest_with_nonzero_overlap', 'another_scene_GT_nearer', 'no_annotated_overlap')
    groups = {label: dict(strict_box_errors=0, selected_mask_hit50=0,
                         zero_box_overlap_mask_hit50=0, selected_mask_hit50_with_qualified_box=0)
              for label in labels}
    joint = dict(box_hit_mask_hit=0, box_hit_mask_miss=0, box_miss_mask_hit=0, box_miss_mask_miss=0)
    for row in rows:
        box_hit = row['selected_iou'] > .5
        mask_hit = row['selected_mask_iou'] > .5
        key = ('box_hit_' if box_hit else 'box_miss_') + ('mask_hit' if mask_hit else 'mask_miss')
        joint[key] += 1
        if not box_hit:
            if row['selected_iou'] > 0 and row['selected_root_joint_best_overlap']:
                label = labels[0]
            elif not row['selected_root_joint_best_overlap']:
                label = labels[1]
            else:
                assert row['selected_iou'] == 0
                label = labels[2]
            group = groups[label]
            group['strict_box_errors'] += 1
            group['selected_mask_hit50'] += int(mask_hit)
            group['zero_box_overlap_mask_hit50'] += int(mask_hit and row['selected_iou'] == 0)
            group['selected_mask_hit50_with_qualified_box'] += int(mask_hit and row['first_qualified_rank']['50'] is not None)
    assert sum(joint.values()) == 9508
    assert joint['box_hit_mask_hit'] + joint['box_hit_mask_miss'] == receipt['rec_hits']['50'] == 4406
    assert [groups[label]['strict_box_errors'] for label in labels] == [1333, 3750, 19]
    result = dict(status='complete_cached_CPU_analysis', model='completed_tail_fused_negative_result',
                  rows=9508, threshold=0.5, same_selected_query=True, counts=joint,
                  geometry_proxy_groups=groups, optimizer_updates=0, GPU_forward_executed=False,
                  rows_sha256=receipt['rows_sha256'],
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  interpretation_limits=[
                      'A GT mask hit and GT box miss are evaluation facts; they do not identify the training cause.',
                      'These are outputs from the old fused P3 checkpoint, not the active candidate-consistency pair.',
                      'Scene-box nearest-overlap groups remain geometry proxies rather than physical instance labels.',
                  ])
    output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
