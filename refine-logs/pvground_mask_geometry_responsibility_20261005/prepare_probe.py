"""Generate the bounded diagnostic from the executed, reviewed setup code."""
import ast
import hashlib
import json
from pathlib import Path

here = Path(__file__).resolve().parent
prior = here.parent / 'pvground_final_quality_20261005'
source_path = prior / 'runtime_bundle/run_final_quality_fit.py'
source = source_path.read_text(encoding='utf-8')
assert hashlib.sha256(source_path.read_bytes()).hexdigest() == '47457bc0a95161287e633aa423274de4f701a7cc482bfc95f9a890d9207a5379'
setup = source[:source.index('    evaluator_path = runtime')]
setup = setup.replace("    parser.add_argument('--mode', choices=['preflight', 'train', 'formal'], required=True)\n", '')
setup = setup.replace("        assert sha(output / name) == digest, name", "        assert sha(Path(spec['helper_root']) / name) == digest, name")
setup = setup.replace('    sys.path.insert(0, str(model_source))', "    sys.path.insert(0, spec['helper_root'])\n    sys.path.insert(0, str(model_source))")
setup = setup.replace('    from readback_preflight_checks import (observed_readback_forward, repeated_forward_differences,\n        zero_readback_cached_native_head, native_bbs_witness, readback_semantic_route)', '    from readback_preflight_checks import observed_readback_forward')
setup = setup.replace('    from pvground_semantic_assignment import semantic_assignment_correction\n    from native_final_quality import native_final_quality_loss, verify_quality_loss\n    from whole_model_preflight_checks import optimizer_restore_exact', '    from native_root_bbs import native_root_bbs\n    from pvground_boundary_box_refiner import distribution_loss')
start = setup.index('    trainable = {name: parameter')
stop = setup.index('    class FitDataset', start)
setup = setup[:start] + "    for parameter in model.parameters():\n        parameter.requires_grad_(False)\n    model.eval()\n    verify_scanrefer_superpoints(manifest['data_root'], 'train', manifest['superpoint_files']['train'])\n\n" + setup[stop:]
start = setup.index('    if formal:\n')
stop = setup.index("        dataset = FitDataset", start)
setup = setup[:start] + ''.join(line[4:] if line.startswith('    ') else line for line in setup[stop:].splitlines(keepends=True))
helpers = source[source.index('    def loader(part, shuffle):'):source.index('    @torch.no_grad()\n    def evaluate(stage):')]
body = (here / 'probe_body.py').read_text(encoding='utf-8')
runner = setup + helpers + body + "\n\nif __name__ == '__main__':\n    main()\n"
assert 'torch.optim' not in runner and 'optimizer.step' not in runner and 'torch.save' not in runner
ast.parse(runner, feature_version=(3, 7))
(here / 'run_mask_geometry_probe.py').write_text(runner, encoding='utf-8')
spec = json.loads((prior / 'quality_fit_spec.json').read_bytes())
spec.update(root='/root/autodl-tmp/pvground_mask_geometry_responsibility_20261005',
    helper_root='/root/autodl-tmp/pvground_final_quality_20261005/quality',
    probe_batches=8, probe_rows=64,
    diagnostic_only=True, optimizer_steps=0, replay_model='protected4506 + fresh zero-output R',
    reference_fit_log='/root/autodl-tmp/pvground_final_quality_20261005/quality/train.jsonl')
(here / 'spec.json').write_text(json.dumps(spec, indent=2)+'\n', encoding='utf-8')
(here / 'GENERATION.json').write_text(json.dumps(dict(source=str(source_path),
    source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
    generated_sha256=hashlib.sha256((here/'run_mask_geometry_probe.py').read_bytes()).hexdigest(),
    parsed_python37=True, new_model_weights=0, optimizer_constructed=False,
    note='Setup and loader copied from executed runner, optimizer/training/evaluation loops excluded.'), indent=2)+'\n', encoding='utf-8')
print(json.dumps(dict(lines=len(runner.splitlines()), runner=str(here/'run_mask_geometry_probe.py'))))
