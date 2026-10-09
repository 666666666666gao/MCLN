"""Prepare only Nr/Sr source adapters; never deploy or inspect active training."""
import ast
import datetime
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
destination = root / 'referit_native_preparation_20261010'
assert not destination.exists()
destination.mkdir()
source = root / 'source'
port = json.loads((root / 'NATIVE_SOURCE_PORT.json').read_bytes())
dataset = root.parent.parent / 'pvground_extremal_span_evidence_20261009/runner_v1/source_review/snapshots/038_joint_det_dataset.py'
assert hashlib.sha256(dataset.read_bytes()).hexdigest() == '3afebfc69232f9547e6810563ba6dd4f34a0f770f929df2f4203b21493de0e3d'
original = {p.relative_to(source).as_posix(): p.read_bytes() for p in source.rglob('*.py')}
assert len(original) == 14
prepared = dict(original)
changes = []


def replace(name, before, after):
    raw = prepared[name]
    # Observed source: initializer is LF, the other four edited files are CRLF.
    ending = '\n' if name == 'native_model_initialization.py' else '\r\n'
    old = before.replace('\n', ending).encode('utf-8')
    new = after.replace('\n', ending).encode('utf-8')
    assert raw.count(old) == 1, name
    prepared[name] = raw.replace(old, new)
    changes.append(dict(file=name, reason=before.splitlines()[0]))


replace('native_model_initialization.py',
    "    assert spec['dataset'] == 'scanrefer' and spec['seed'] == 2027",
    "    assert spec['dataset'] in ('nr3d', 'sr3d') and spec['seed'] == 2027\n"
    "    assert spec['new_module_initialization'] == 'fresh'\n"
    "    assert spec['g_checkpoint'] is None and spec['support_checkpoint'] is None and spec['span_checkpoint'] is None")
start = "    if initialize_weights:\n        parent_g = checked_payload(spec['g_checkpoint'], spec['g_checkpoint_sha256'])"
end = '    model.candidate_support_corrector = CandidateMaskSupportCorrector(False)'
text = prepared['native_model_initialization.py'].decode('utf-8').replace('\r\n', '\n')
assert text.count(start) == text.count(end) == 1
before = text[text.index(start):text.index(end)]
replace('native_model_initialization.py', before, '')
start = "    if initialize_weights:\n        support = checked_payload(spec['support_checkpoint'], spec['support_checkpoint_sha256'])"
end = "    assert len(model.state_dict()) == 1295"
text = prepared['native_model_initialization.py'].decode('utf-8').replace('\r\n', '\n')
assert text.count(start) == text.count(end) == 1
before = text[text.index(start):text.index(end)]
replace('native_model_initialization.py', before,
    "    model.candidate_span_mixer = ExtremalSpanMixer(spec['span_source_mode'])\n")
replace('native_model_initialization.py', "        dataset='scanrefer', seed=2027)",
    "        dataset=spec['dataset'], seed=2027,\n"
    "        pretrained_core='corresponding_author_checkpoint', new_module_initialization='fresh',\n"
    "        scanrefer_core_or_module_state_loaded=False)")
replace('train_dist_mod.py',
    "    assert opt.model == 'PVGround' and opt.dataset == ['scanrefer'] and opt.test_dataset == 'scanrefer'",
    "    assert opt.model == 'PVGround' and opt.test_dataset in ('nr3d', 'sr3d')\n"
    "    assert opt.dataset == [opt.test_dataset] and opt.num_decoder_layers == 6")

replace('pvground_semantic_assignment.py',
    "    assert bool(batch['box_label_mask'][:, 0].all()) and bool(torch.isfinite(iou).all())",
    "    assert all(name in ('scanrefer', 'nr3d', 'sr3d', 'scannet') for name in batch['sample_dataset'])\n"
    "    expression = torch.tensor([name != 'scannet' for name in batch['sample_dataset']],\n"
    "                              device=iou.device, dtype=torch.bool)\n"
    "    assert bool(batch['box_label_mask'][expression, 0].all()) and bool(torch.isfinite(iou).all())")
replace('pvground_semantic_assignment.py',
    "        assert int((targets == 0).sum()) == 1\n        matched[bid, queries] = True\n    return (iou > .5) & ~matched",
    "        if bool(expression[bid]):\n            assert int((targets == 0).sum()) == 1\n        matched[bid, queries] = True\n    return (iou > .5) & ~matched & expression[:, None]")
replace('pvground_semantic_assignment.py',
    "    # Exact ScanRefer native weighting, including its unnormalized target mass.",
    "    assert len(set(batch['language_dataset'])) == 1\n"
    "    assert batch['language_dataset'][0] in ('scanrefer', 'nr3d', 'sr3d')\n"
    "    if batch['language_dataset'][0] == 'sr3d':\n"
    "        return (batch['positive_map'][:, 0] * .625\n"
    "                + batch['modify_positive_map'][:, 0] * .125\n"
    "                + batch['pron_positive_map'][:, 0] * .125\n"
    "                + batch['rel_positive_map'][:, 0] * .125).detach()\n"
    "    # Exact ScanRefer/Nr3D native weighting, including unnormalized mass.")
replace('pvground_semantic_assignment.py',
    "def semantic_assignment_correction(predictions, batch, indices, eos_coef):\n"
    "    assert eos_coef == .1 and all(x == 'scanrefer' for x in batch['language_dataset'])",
    "def semantic_assignment_correction(predictions, batch, indices, eos_coef, num_decoder_layers):\n"
    "    assert eos_coef == .1 and num_decoder_layers == 6")
replace('pvground_semantic_assignment.py',
    "    assert bool((target[:, -1] == 0).all()) and bool((target.sum(-1) > 0).all())",
    "    active = selected.any(-1)\n"
    "    assert bool((target[active, -1] == 0).all()) and bool((target[active].sum(-1) > 0).all())")
replace('pvground_semantic_assignment.py',
    "    correction = (new_ce - old_ce) * (.5 / 7)",
    "    coefficient = .5 if batch['language_dataset'][0] == 'scanrefer' else 1.0\n"
    "    correction = (new_ce - old_ce) * (coefficient / (num_decoder_layers + 1))")
replace('pvground_semantic_assignment.py',
    "def verify_native_replacement(predictions, batch, indices, eos_coef, correction):",
    "def verify_native_replacement(predictions, batch, indices, eos_coef, correction, num_decoder_layers):")
replace('pvground_semantic_assignment.py',
    "        target[bid, queries] = maps[0] * .6 + maps[1] * .2 + maps[2] * .2 + maps[3] * .1",
    "        if batch['language_dataset'][0] == 'sr3d':\n"
    "            target[bid, queries] = maps[0] * .625 + maps[1] * .125 + maps[2] * .125 + maps[3] * .125\n"
    "        else:\n"
    "            target[bid, queries] = maps[0] * .6 + maps[1] * .2 + maps[2] * .2 + maps[3] * .1")
replace('pvground_semantic_assignment.py',
    "    expected = ce(changed) * (.5 / 7)\n    corrected = actual * (.5 / 7) + correction",
    "    coefficient = .5 if batch['language_dataset'][0] == 'scanrefer' else 1.0\n"
    "    expected = ce(changed) * (coefficient / (num_decoder_layers + 1))\n"
    "    corrected = actual * (coefficient / (num_decoder_layers + 1)) + correction")
replace('models/losses.py',
    "            correction, record = semantic_assignment_correction(end_points, end_points, last_indices, set_criterion.eos_coef)",
    "            correction, record = semantic_assignment_correction(end_points, end_points, last_indices, set_criterion.eos_coef, num_decoder_layers)")
replace('selected_query_mask_objective.py',
    "        assert batch['language_dataset'][bid] == 'scanrefer'",
    "        assert batch['language_dataset'][bid] in ('scanrefer', 'nr3d', 'sr3d')\n"
    "        assert batch['sample_dataset'][bid] in ('scanrefer', 'nr3d', 'sr3d', 'scannet')\n"
    "        if batch['sample_dataset'][bid] == 'scannet':\n"
    "            records.append(dict(query=int(selected[bid]), supervised=False, role='detection',\n"
    "                target_gt_slot=None, reason='joint_detection_row_preserved'))\n"
    "            continue")
replace('selected_query_mask_objective.py',
    "                        for role in ('matched_root','matched_other','unmatched')},",
    "                        for role in ('matched_root','matched_other','unmatched','detection')},")

changed = []
for name, raw in prepared.items():
    ast.parse(raw, filename=name)
    target = destination / 'source' / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
    if raw != original[name]:
        changed.append(name)
assert sorted(changed) == sorted(['native_model_initialization.py', 'pvground_semantic_assignment.py',
    'selected_query_mask_objective.py', 'models/losses.py', 'train_dist_mod.py'])
old_referit = root.parent.parent / 'pvground_referit_mask_reference_20261006/complete_preflight'
for dataset_name in ('nr3d', 'sr3d'):
    old_spec = json.loads((old_referit / (dataset_name + '_native/spec.json')).read_bytes())
    spec = dict(dataset=dataset_name, seed=2027, official_checkpoint=old_spec['checkpoint'],
        official_checkpoint_sha256=old_spec['checkpoint_sha256'], new_module_initialization='fresh',
        g_checkpoint=None, support_checkpoint=None, span_checkpoint=None,
        span_source_mode='extremal_support', use_g_supervision=True, use_selected_mask_supervision=True)
    arm = destination / dataset_name
    arm.mkdir()
    (arm / 'init.json').write_text(json.dumps(spec, indent=2) + '\n', encoding='utf-8')
report = dict(status='REFERIT_NATIVE_SOURCE_ADAPTERS_PREPARED_NOT_EXECUTED',
    time_cst=datetime.datetime.now().astimezone().isoformat(), changed_files=sorted(changed),
    source_file_count=14, AST_files_parsed=14, model_architecture='same_current_G_A_B_C_native_1295_state_layout',
    parent_core_initialization='corresponding_author_checkpoint', scanrefer_state_transfer=False,
    new_modules='fresh G/A/B; C no parameters', dataset_snapshot=str(dataset),
    dataset_snapshot_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
    source_port_sha256=hashlib.sha256((root / 'NATIVE_SOURCE_PORT.json').read_bytes()).hexdigest(),
    original_source_sha256={name:hashlib.sha256(raw).hexdigest() for name,raw in original.items()},
    prepared_source_sha256={name:hashlib.sha256(raw).hexdigest() for name,raw in prepared.items()},
    exact_changes=changes, active_scanrefer_source_changed=False, GPU_calls=0, SSH_calls=0,
    full_model_checked=False, real_dataset_rows_checked=0, source_review_pending=True,
    any_Nr_Sr_training_launched=False, full_goal_complete=False)
(destination / 'SOURCE_ADAPTER_PREPARATION.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({k:report[k] for k in ('status','changed_files','AST_files_parsed','scanrefer_state_transfer',
    'active_scanrefer_source_changed','any_Nr_Sr_training_launched')}))
