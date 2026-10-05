import ast
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parent
prior=root.parent/'pvground_mask_branch_responsibility_20261005'
source=prior/'run_mask_branch_probe.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='84c0858166781214cf241a04d5ee42243b92246fe7c4705579568f58fc6602c7'
setup=source.read_text(encoding='utf-8').split("    assert spec['probe_batches']==8 and spec['probe_rows']==64")[0]
setup=setup.replace('Read-only candidate Query/Text/fused Mask qualification on augmentedfit64.',
    'Read-only paired geometry heads on the existing augmentedfit64 support cohort.')
setup=setup.replace(" and spec['quality_weight'] == 1.0","")
setup=setup.replace('from pvground_boundary_box_refiner import distribution_loss',
    'from pvground_boundary_box_refiner import face_targets, KNOTS')
runner=setup+(root/'probe_body.py').read_text(encoding='utf-8')+"\n\nif __name__ == '__main__':\n    main()\n"
ast.parse(runner,feature_version=(3,7))
assert all(token not in runner for token in ('torch.optim','torch.save','torch.autograd.grad','.backward('))
(root/'run_cohort_probe.py').write_text(runner,encoding='utf-8')
for name in ('controller.py','launch_probe_authorized.py','observe_probe_authorized.py','collect_probe_authorized.py'):
    text=(prior/name).read_text(encoding='utf-8')
    text=text.replace('pvground_mask_branch_responsibility_20261005','pvground_query_geometry_cohort_20261005')
    text=text.replace('run_mask_branch_probe.py','run_cohort_probe.py')
    text=text.replace('pvg_mask_branch_probe_20261005','pvg_query_geometry_cohort_20261005')
    text=text.replace('MASK_BRANCH_OBSERVATION','GEOMETRY_COHORT_OBSERVATION')
    if name=='controller.py':
        text=text.replace("receipt['model_state_unchanged']","receipt['model_state_restored']")
    if name=='launch_probe_authorized.py':
        text=text.replace("closed=Path('/root/autodl-tmp/pvground_mask_geometry_responsibility_20261005')",
            "closed=Path('/root/autodl-tmp/pvground_query_supported_geometry_20261005')")
        text=text.replace("(closed/'status.json')","(closed/'fit_status.json')")
        text=text.replace("(closed/'controller.exit')","(closed/'fit_controller.exit')")
        text=text.replace("estimated_seconds=480","estimated_seconds=600")
        text=text.replace("Warm parent reconstruction plus8 actual fit-batch forwards; no validation or optimizer.",
            "Prior actual augmentedfit64 probe428.20s plus24 cached geometry-head replays; no validation or optimizer.")
    ast.parse(text,feature_version=(3,7))
    (root/name).write_text(text,encoding='utf-8')
(root/'GENERATION.json').write_text(json.dumps(dict(source=str(source),
    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    runner_sha256=hashlib.sha256((root/'run_cohort_probe.py').read_bytes()).hexdigest(),
    python37_parsed=True,weight_files_created=0,optimizer_constructed=False),indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(runner=str(root/'run_cohort_probe.py'),lines=len(runner.splitlines()))))
