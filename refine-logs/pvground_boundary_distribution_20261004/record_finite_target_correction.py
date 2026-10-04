"""Close the plan's explicit pre-clamp target witness gap, before deployment."""
import datetime
import hashlib
import json
from pathlib import Path

local = Path(__file__).resolve().parent
module = local/'pvground_boundary_box_refiner.py'
digest = hashlib.sha256(module.read_bytes()).hexdigest()
assert not (local/'finite_target_correction.json').exists()
before = {}
for arm in ('residual','distribution'):
    for suffix in ('_preflight_spec.json','_spec.json'):
        path = local/(arm+suffix)
        item = json.loads(path.read_bytes())
        before[path.name] = item['whole_range_files'][module.name]
        item['whole_range_files'][module.name] = digest
        path.write_text(json.dumps(item,indent=2)+'\n',encoding='utf-8')
preparation_path = local/'source_preparation.json'
preparation = json.loads(preparation_path.read_bytes())
preparation['source_files'][module.name] = digest
preparation['post_prepare_source_correction'] = 'explicit planned pre-clamp finite-target assertion and logged witness'
preparation_path.write_text(json.dumps(preparation,indent=2)+'\n',encoding='utf-8')
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(),
    reason='actual reviewer found missing explicit finite-target witness required by existing plan',
    original_module_hashes=before,current_module_sha256=digest,
    change='assert original continuous targets finite before clamp; log boundary_targets_finite=True',
    native_model_forwards=0,optimizer_steps=0,remote_jobs=0,source_review_recheck_required=True)
(local/'finite_target_correction.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record))
