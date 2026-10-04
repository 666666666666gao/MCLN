"""Count saved diagnostic match roles; these are validation root-only matches."""
import collections
import datetime
import hashlib
import json
from pathlib import Path


local = Path(__file__).resolve().parent
source = local.parent/'pvground_fused_support_20261002/candidate_audit/full_gt_scope_v2'
receipt = json.loads((source/'receipt.json').read_bytes())
raw = (source/'rows.jsonl').read_bytes()
assert hashlib.sha256(raw).hexdigest()==receipt['rows_sha256']
rows = [json.loads(line) for line in raw.decode('utf-8').splitlines()]
assert len(rows)==9508 and [row['row_id'] for row in rows]==list(range(9508))
assert all(row['valid_native_GT_slots']==[0] and row['candidate_count']==256 for row in rows)
subset = [row for row in rows if row['selected_mask_iou']>.5 and row['selected_iou']<=.5]
assert len(subset)==1008
roles = collections.Counter('matched_root' if row['selected_matched_slot']==0 else
    'unmatched' if row['selected_matched_slot']==-1 else 'matched_other' for row in subset)
result = dict(time_cst=datetime.datetime.now().astimezone().isoformat(),source=str(source),
    source_rows_sha256=receipt['rows_sha256'],model='legacy failed tail_fused',
    source_formal_hits=[5566,4406],selected_mask_good_box_bad=len(subset),
    selected_match_roles=dict(roles),all_formal_native_gt_slots_root_only=True,
    row_ids_by_role={role:[row['row_id'] for row in subset if
        ('matched_root' if row['selected_matched_slot']==0 else 'unmatched' if row['selected_matched_slot']==-1 else 'matched_other')==role]
        for role in roles},inference_or_optimizer_replayed=False,new_accuracy_result=False,
    scope='Saved validation-input Hungarian matches with only root GT slot, not actual training assignments or physical identity. Legacy failure model, not protected4506 parent. No label/matching change or deployment gate. Counts motivate checking actual training geometry responsibility; they do not prove its cause or predict auxiliary-loss efficacy.')
destination=local/'LEGACY_MASK_QUALIFIED_MATCH_ROLE.json'
assert not destination.exists()
destination.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps({key:value for key,value in result.items() if key!='row_ids_by_role'}),flush=True)
