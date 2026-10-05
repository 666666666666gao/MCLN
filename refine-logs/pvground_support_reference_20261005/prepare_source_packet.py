"""Collect actual source paths for the required fresh SOURCE review."""
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parent
assert not (root/'SOURCE_PACKET.json').exists()
paths=list(root.glob('*.py'))+list(root.glob('*spec.json'))+[root/'EXPERIMENT_PLAN.md']
bundle=root.parent/'pvground_final_quality_20261005/runtime_bundle'
spec=json.loads((root/'control_spec.json').read_bytes())
paths += [bundle/name for name in spec['runner_files']]
model=root.parent/'pvground_geometry_readback_20261004/revision2/source_preview/PV-Ground'
actual_imports=json.loads((root.parent/'pvground_auxiliary_target_20261005/complete/control/imports.json').read_bytes())
copies={
    'models.pv_ground':model/'models/pv_ground.py',
    'models.losses':root.parent/'pvground_g_p2_20261002/complete/source/ported/models/losses.py',
    'main_utils':root.parent/'pvground_fused_support_20261002/complete_tail_raw/source/imported/main_utils.py',
    'prepare_data':root.parent/'pvground_fused_support_20261002/complete_tail_raw/source/imported/prepare_data.py',
    'src.joint_det_dataset':root.parent/'pvground_g_p2_20261002/complete/source/joint_det_dataset.py'}
for name,path in copies.items():
    assert hashlib.sha256(path.read_bytes()).hexdigest()==actual_imports['sha256'][name],str(path)
paths += list(copies.values())
paths += [root.parent/'pvground_g_p2_20261002/complete/source/joint_det_dataset.py',
          root.parent/'pv_ground_source_20260905/src/visual_data_handlers.py',
          root.parent/'pvground_runtime_bundle_20260908_v1/PV-Ground/src/grounding_evaluator.py',
          root.parent/'pvground_auxiliary_target_20261005/weight_retention.json',
          root.parent/'pvground_auxiliary_target_20261005/terminal_publication.json',
          root.parent/'pvground_auxiliary_target_20261005/complete/control/receipt.json']
entries=[]
for path in sorted(set(paths)):
    assert path.is_file(),str(path)
    raw=path.read_bytes()
    entries.append({'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
record={'execution_scope':'SOURCE_ONLY','status':'READY_NOT_REVIEWED','files':entries,
        'weights_or_auth_wrapper_included':False,'GPU_or_remote_calls':0}
(root/'SOURCE_PACKET.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'source_files':len(entries),'scope':'SOURCE_ONLY','GPU_runs':0}))
