"""Prepare the formal loop using the actual successful factory and executed evaluator."""
import ast
import hashlib
import json
from pathlib import Path
import shutil

root = Path(__file__).parent
engineering = root.parent / 'pvground_whole_mask_integration_20261003'
prior = root.parent / 'pvground_fused_support_20261002'
assert not (root / 'source_preparation.json').exists()
core_path = engineering / 'run_whole_mask_preflight.py'
tail_path = prior / 'run_pvground_tail_support_retry.py'
core = core_path.read_text().replace('\r\n', '\n')
core = core[:core.rindex("if __name__ == '__main__':")]
tail = tail_path.read_text().replace('\r\n', '\n')
tail = tail[tail.index('    @torch.no_grad()\n    def evaluate(stage):'):]

def replace(text, before, after):
    assert text.count(before) == 1, before
    return text.replace(before, after)

core = replace(core, '"""Actual PV factory and two-step whole-Mask range preflight; no formal fit."""',
               '"""Same original G: local fused support with whole-Mask range information on/off."""')
core = replace(core, "choices=['cpu', 'preflight']", "choices=['cpu', 'preflight', 'train', 'formal']")
core = replace(core, "        assert terminal['tail_module_sha256'] == spec['tail_module_sha256']\n",
               "        assert terminal['tail_module_sha256'] == spec['tail_module_sha256']\n"
               "        assert terminal['whole_range_files'] == spec['whole_range_files']\n"
               "        assert terminal['use_whole_range'] == spec['use_whole_range']\n"
               "        assert terminal['spec_sha256'] == sha(args.spec)\n")
before = tail[tail.index("    reference_root=Path(spec['reused_control_root'])"):tail.index("    assert all(torch.equal(v.detach().cpu(),initial[k])")]
tail = replace(tail, before,
    "    initial_comparison = None\n"
    "    reference_training = None\n"
    "    if spec['use_whole_range']:\n"
    "        reference_root = Path(spec['paired_control_root'])\n"
    "        reference_initial = [json.loads(line) for line in (reference_root/'initial/rows.jsonl').read_text().splitlines()]\n"
    "        from initial_range_comparison import compare_initial_rows\n"
    "        initial_comparison = compare_initial_rows(initial_rows, reference_initial)\n"
    "        write_json(output/'initial_control_comparison.json', initial_comparison)\n"
    "        reference_training = [json.loads(line) for line in (reference_root/'train.jsonl').read_text().splitlines()]\n"
    "        assert len(reference_training) == 3723\n"
    "        print('WHOLE_RANGE_INITIAL_PAIR_COMPARISON '+json.dumps(initial_comparison), flush=True)\n")
tail = replace(tail, "            assert batch['local_training_id'].tolist()==reference_training[index-1]['rows']\n",
    "            if reference_training is not None:\n"
    "                assert batch['local_training_id'].tolist() == reference_training[index-1]['rows']\n")
tail = replace(tail, "        data.update(semantic_assignment=True,assignment_module_sha256=spec['assignment_module_sha256'],\n",
    "        data.update(whole_range_files=spec['whole_range_files'], use_whole_range=spec['use_whole_range'],\n"
    "                    whole_range_model=True, head_parameters=400614)\n"
    "        data.update(semantic_assignment=True,assignment_module_sha256=spec['assignment_module_sha256'],\n")
start = tail.index("    receipt.update(p3=True,p3_module_sha256=spec['p3_module_sha256'],initial_rec_inputs_match_control=")
end = tail.index('    receipt.update(semantic_assignment=', start)
tail = tail[:start] + (
    "    receipt.update(p3=True, p3_module_sha256=spec['p3_module_sha256'],\n"
    "                   initial_pair_comparison=initial_comparison,\n"
    "                   fit_batch_row_order_matches_control=reference_training is not None,\n"
    "                   paired_control_root=spec['paired_control_root'],\n"
    "                   whole_range_files=spec['whole_range_files'], use_whole_range=spec['use_whole_range'],\n"
    "                   whole_range_model=True, head_parameters=400614)\n"
) + tail[end:]
text = core + tail
ast.parse(text, feature_version=(3, 7))
(root/'run_whole_mask_fit.py').write_text(text, encoding='utf-8', newline='\n')
modules = ['whole_model_preflight_checks.py', 'pvground_whole_mask_box_refiner.py', 'whole_mask_range.py',
    'pvground_candidate_box_refiner.py', 'pvground_tail_support_box_refiner.py', 'pvground_tail_preflight.py',
    'pvground_semantic_assignment.py', 'pvground_source_query.py', 'pvground_observation_query.py',
    'pvground_task_observation_query.py']
for name in modules:
    shutil.copyfile(engineering/name, root/name)
for arm in ('local_range', 'whole_range'):
    spec = json.loads((engineering/(arm+'_spec.json')).read_bytes())
    spec.update(root='/root/autodl-tmp/pvground_whole_mask_fit_20261003/'+arm,
        comparison='same original G / fresh optimizer / paired range-information source control',
        formal_rows=9508, preflight_only_at_install=False, formal_training_supported=True,
        paired_control_root='/root/autodl-tmp/pvground_whole_mask_fit_20261003/local_range',
        primary_threshold=.5, preflight_root='/root/autodl-tmp/pvground_whole_mask_preflight_20261003/'+arm)
    (root/(arm+'_spec.json')).write_text(json.dumps(spec, indent=2)+'\n', encoding='utf-8', newline='\n')
shutil.copyfile(engineering/'FORMAL_RANGE_CONTROL_PLAN.md', root/'FORMAL_RANGE_CONTROL_PLAN.md')
record = dict(status='PREPARED_NOT_LAUNCHED',
    factory_source=str(core_path), factory_sha256=hashlib.sha256(core_path.read_bytes()).hexdigest(),
    loop_source=str(tail_path), loop_sha256=hashlib.sha256(tail_path.read_bytes()).hexdigest(),
    model_modules_copied_without_changes=modules,
    model_updates=0, formal_training_started=False,
    changes=['restore train/formal entry points', 'reuse actual executed evaluation and fit loop',
             'compare actual pair starts with discrepancies recorded', 'add current range metadata to strict restore'])
(root/'source_preparation.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
print(json.dumps(record))
