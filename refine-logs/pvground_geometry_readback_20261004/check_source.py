"""Source syntax/identity only; no torch import or native execution."""
import ast
import datetime
import hashlib
import json
from pathlib import Path
import shutil

local = Path(__file__).resolve().parent
assert not (local/'LOCAL_SOURCE_CHECK.json').exists()
prior = local.parent/'pvground_boundary_distribution_20261004/pvground_boundary_box_refiner.py'
shutil.copyfile(prior,local/prior.name)
assert hashlib.sha256(prior.read_bytes()).hexdigest() == '88debfef64f4295f87d697b7b9d4cd1eab17b39f4041f6d9ae773fc65dda9461'
model = local/'pvground_boundary_evidence_readback.py'
ast.parse(model.read_text(encoding='utf-8'),feature_version=(3,7))
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(),
    status='LOCAL_AST_ONLY_UNINTEGRATED', files={name:dict(bytes=(local/name).stat().st_size,
        sha256=hashlib.sha256((local/name).read_bytes()).hexdigest())
        for name in ('pvground_boundary_evidence_readback.py','pvground_boundary_box_refiner.py','SOURCE_SCOPE.md')},
    native_factory_integrated=False, gpu_forwards=0, optimizer_updates=0,
    measured_parameters=None, calculated_parameters=96672, calculated_trainable_tensors=23,
    active_face_fit_changed=False, accuracy_result=False)
(local/'LOCAL_SOURCE_CHECK.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status=record['status'],model_source_bytes=model.stat().st_size,
    native_forwards=0,optimizer_updates=0,accuracy=False)))
