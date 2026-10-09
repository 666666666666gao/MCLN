"""Prepare an isolated native entry from the already executed dataset contract."""
import ast
import hashlib
import json
from pathlib import Path


root = Path(__file__).resolve().parent
output = root / 'runner_v1'
output.mkdir(exist_ok=True)
prior = root.parent / 'pvground_selected_mask_training_20261009'
old = (prior / 'run_mask_support_pair.py').read_text(encoding='utf-8')
assert hashlib.sha256((prior / 'run_mask_support_pair.py').read_bytes()).hexdigest() == '51413967adca1a644731040565f312856f5854ca85f881c23791bca20b1ab69a'
replacements = [
    ('Native Mask support correction, matched parent responsibility, one frozen PV.',
     'Native geometry span control, original matched responsibility, one frozen PV.'),
    ("choices=['preflight', 'train', 'initial_formal', 'formal']", "choices=['preflight', 'train', 'formal']"),
    ("assert spec['support_modes'] == ['content', 'selected_query']",
     "assert spec['support_modes'] == ['whole_support', 'extremal_support']\n    assert spec['parent_selection_status'] == 'FINALIZED_AFTER_PRIOR_PAIR_CLOSED'"),
    ('from paired_support_loop import PairedSupportRun', 'from paired_span_loop import PairedSpanRun'),
    ("assert spec['starting_hits']==[5599,4859] and spec['head_only']",
     "assert spec['candidate_gate_hits']==[5658,4850] and spec['head_only']"),
    ("model,config,load=build_support_model(cfg,official_payload,g_payload,selected_payload,manifest['data_root'])\n    assert load['full_state_tensors']==1304 and not load['old_geometry_checkpoint_required']",
     "assert sha(spec['parent_support_terminal']) == spec['parent_support_terminal_sha256']\n    parent_support_payload = torch.load(spec['parent_support_terminal'], map_location='cpu')\n    model,config,load=build_support_model(cfg,official_payload,g_payload,selected_payload,manifest['data_root'],parent_support_payload)\n    assert load['full_state_tensors']==1314 and not load['old_geometry_checkpoint_required']\n    for parameter in model.parameters():\n        parameter.requires_grad_(False)"),
    ("added_support_parameters_per_arm=27841, initial_support_output_zero=False, warm_support_restore_required=True,\n        only_mask_correction_trainable=True, fresh_optimizers_required=True",
     "added_span_parameters_per_arm=29793, initial_span_output_zero=True, parent_support_restore_required=True,\n        only_span_mixer_trainable=True, fresh_optimizers_required=True"),
    ("formal = args.mode in ('initial_formal','formal')", "formal = args.mode == 'formal'"),
    ('runner=PairedSupportRun(model,config,spec,output,args,initial,prepare,loader,dataset,',
     'runner=PairedSpanRun(model,config,spec,output,args,initial,prepare,loader,dataset,'),
    ("g_payload,selected_payload,manifest['data_root'])\n    runner.run",
     "g_payload,selected_payload,manifest['data_root'],parent_support_payload)\n    runner.run"),
]
text = old
for before, after in replacements:
    assert text.count(before) == 1, before
    text = text.replace(before, after)
ast.parse(text, feature_version=(3, 7))
target = output / 'run_span_pair.py'
assert not target.exists()
target.write_text(text, encoding='utf-8')
for name in ('mask_reference.py', 'mask_support_corrector.py', 'mask_support_model_factory.py',
             'selected_mask_reference_factory.py', 'matched_mask_objective.py'):
    destination = output / name
    assert not destination.exists()
    destination.write_bytes((prior / name).read_bytes())
for name in ('extremal_span_mixer.py', 'span_refinement_model.py', 'matched_span_objective.py'):
    destination = output / name
    assert not destination.exists()
    destination.write_bytes((root / name).read_bytes())
(output / 'ENTRY_PROVENANCE.json').write_text(json.dumps(dict(
    source=str(prior / 'run_mask_support_pair.py'),
    source_sha256=hashlib.sha256((prior / 'run_mask_support_pair.py').read_bytes()).hexdigest(),
    prepared_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
    replacement_count=len(replacements), input_protocol_copied=True,
    active_source_changed=False, neural_forwards=0, deployed=False), indent=2) + '\n')
print(json.dumps(dict(status='NATIVE_ENTRY_PREPARED_NOT_EXECUTED', bytes=target.stat().st_size,
                     py37_ast=True, active_source_changed=False)), flush=True)
