import ast
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parent
prior=root.parent/'pvground_mask_geometry_responsibility_20261005'
source=prior/'run_mask_geometry_probe.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='ff7a5ec93a0021c53fd08ed362d92c7e2f7cff6b99736452ad5237bbf47c73ff'
setup=source.read_text(encoding='utf-8').split('    # Fixed first eight fit batches')[0]
setup=setup.replace('"""Frozen4506 R fit with training-only native final-IoU differences.\n\nImplementation draft; source review and two-step preflight precede launch.\n"""','"""Read-only candidate Query/Text/fused Mask qualification on augmentedfit64."""')
runner=setup+(root/'probe_body.py').read_text(encoding='utf-8')+"\n\nif __name__ == '__main__':\n    main()\n"
ast.parse(runner,feature_version=(3,7))
assert 'torch.optim' not in runner and 'torch.save' not in runner and 'torch.autograd.grad' not in runner
(root/'run_mask_branch_probe.py').write_text(runner,encoding='utf-8')
spec=json.loads((prior/'spec.json').read_bytes())
spec.update(root='/root/autodl-tmp/pvground_mask_branch_responsibility_20261005',
    reference_rows='/root/autodl-tmp/pvground_mask_geometry_responsibility_20261005/rows.jsonl',
    purpose='Candidate-specific Query support versus shared Text support; no output gradient replay')
(root/'spec.json').write_text(json.dumps(spec,indent=2)+'\n',encoding='utf-8')
for name in ('controller.py','launch_probe_authorized.py','observe_probe_authorized.py','collect_probe_authorized.py'):
    text=(prior/name).read_text(encoding='utf-8')
    text=text.replace('pvground_mask_geometry_responsibility_20261005','pvground_mask_branch_responsibility_20261005')
    text=text.replace('run_mask_geometry_probe.py','run_mask_branch_probe.py')
    text=text.replace('pvg_mask_geometry_probe_20261005','pvg_mask_branch_probe_20261005')
    text=text.replace('FIT_ROLE_OBSERVATION','MASK_BRANCH_OBSERVATION')
    if name=='launch_probe_authorized.py':
        text=text.replace("closed=Path('/root/autodl-tmp/pvground_final_quality_20261005')", "closed=Path('/root/autodl-tmp/pvground_mask_geometry_responsibility_20261005')")
        text=text.replace("(closed/'fit_status.json')", "(closed/'status.json')")
        text=text.replace("(closed/'fit_controller.exit')", "(closed/'controller.exit')")
    ast.parse(text,feature_version=(3,7))
    (root/name).write_text(text,encoding='utf-8')
(root/'GENERATION.json').write_text(json.dumps(dict(source=str(source),
    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    generated_sha256=hashlib.sha256((root/'run_mask_branch_probe.py').read_bytes()).hexdigest(),
    python37_parsed=True,new_model_weights=0,optimizer_constructed=False,
    reused_setup_loader_controller=True),indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(runner=str(root/'run_mask_branch_probe.py'),lines=len(runner.splitlines()))))
