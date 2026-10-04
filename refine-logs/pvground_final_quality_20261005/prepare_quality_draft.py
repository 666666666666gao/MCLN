"""Prepare an isolated training-loss change from the actually executed control."""
import ast
import hashlib
import json
from pathlib import Path


root = Path(__file__).resolve().parent
control = root.parent / 'pvground_geometry_readback_20261004/formal_draft'
bundle = root / 'runtime_bundle'
assert not bundle.exists()
bundle.mkdir()
spec = json.loads((control / 'evidence_visible_fit_spec.json').read_bytes())
for name, digest in spec['runner_files'].items():
    raw = (control / 'runtime_bundle' / name).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == digest
    if name != 'run_readback_fit.py':
        (bundle / name).write_bytes(raw)
(bundle / 'native_final_quality.py').write_bytes((root / 'native_final_quality.py').read_bytes())
source = (control / 'runtime_bundle/run_readback_fit.py').read_text(encoding='utf-8')


def replace_once(old, new):
    global source
    assert source.count(old) == 1, old
    source = source.replace(old, new)


replace_once(source[:source.index('import argparse')],
    '"""Frozen4506 R fit with training-only native final-IoU differences.\n\n'
    'Implementation draft; source review and two-step preflight precede launch.\n"""\n')
replace_once("choices=['train', 'formal']", "choices=['preflight', 'train', 'formal']")
replace_once("    assert isinstance(spec['use_geometry_evidence'], bool)",
    "    assert spec['use_geometry_evidence'] is True and spec['quality_weight'] == 1.0")
replace_once('    from pvground_semantic_assignment import semantic_assignment_correction',
    '    from pvground_semantic_assignment import semantic_assignment_correction\n'
    '    from native_final_quality import native_final_quality_loss, verify_quality_loss')
replace_once("        assert terminal['use_geometry_evidence'] == spec['use_geometry_evidence']",
    "        assert terminal['use_geometry_evidence'] == spec['use_geometry_evidence']\n"
    "        assert terminal['quality_weight'] == spec['quality_weight']")
preflight = '''    if args.mode == 'preflight':
        dataset.augment = True
        dataset.augment_det = True
        reset_rng()
        batch_cpu = next(iter(loader('fit', True)))
        assert len(batch_cpu['utterances']) == 8
        model.eval()
        readback.train()
        witnesses = []
        for previous_updates in (0, 1):
            inputs, batch = prepare(batch_cpu, 'train')
            predictions, call = observed_readback_forward(model, inputs)
            zero = zero_readback_cached_native_head(model, predictions) if previous_updates == 0 else None
            matching = []
            hook = set_criterion.matcher.register_forward_hook(
                lambda module, arguments, result: matching.append([(q.clone(), t.clone()) for q, t in result]))
            native, predictions = native_loss(predictions, batch)
            hook.remove()
            assert len(matching) == 7
            correction, assignment = semantic_assignment_correction(predictions, batch, matching[1], set_criterion.eos_coef)
            quality, counts = native_final_quality_loss(predictions, batch, matching[1])
            route = verify_quality_loss(predictions, batch, matching[1], readback, previous_updates)
            semantic = readback_semantic_route(predictions, correction, readback, previous_updates)
            scoring = native_bbs_witness(predictions['last_sem_cls_scores'], batch)
            loss = native + correction + spec['quality_weight'] * quality
            optimizer.zero_grad()
            loss.backward()
            assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in trainable.values())
            assert all(parameter.grad is None for name, parameter in model.named_parameters() if name not in selected)
            norm = torch.nn.utils.clip_grad_norm_(tuple(trainable.values()), spec['clip_norm'])
            optimizer.step()
            witnesses.append(dict(previous_updates=previous_updates, loss=float(loss), quality=float(quality),
                counts=counts, quality_route=route, semantic_route=semantic, native_score=scoring,
                same_frame_call=call, zero_output=zero, gradient_norm=float(norm)))
            del inputs, predictions, batch
        assert all(torch.equal(model.state_dict()[name].detach().cpu(), initial[name]) for name in core_names)
        delta = {name: value.detach().cpu().clone() for name, value in model.state_dict().items() if name in selected}
        memory = io.BytesIO()
        torch.save(dict(state_delta=delta, optimizer=optimizer.state_dict()), memory)
        serialization_bytes = memory.tell()
        memory.seek(0)
        restored = torch.load(memory, map_location='cpu')
        with torch.no_grad():
            next(iter(trainable.values())).add_(1)
        model.load_state_dict(dict(initial, **restored['state_delta']), strict=True)
        optimizer.load_state_dict(restored['optimizer'])
        optimizer_check = optimizer_restore_exact(optimizer, restored['optimizer'])
        assert all(int(state['step']) == 2 for state in optimizer.state.values())
        assert all(torch.equal(model.state_dict()[name].detach().cpu(), value) for name, value in restored['state_delta'].items())
        assert all(torch.equal(model.state_dict()[name].detach().cpu(), initial[name]) for name in core_names)
        receipt = dict(status='pass', time_cst=datetime.datetime.now().astimezone().isoformat(),
            optimizer_steps=2, batch_size=8, accuracy_result=False, weight_files_created=0,
            readback_parameters=load['readback_parameters'], readback_state_tensors=23,
            geometry_provider_and_g_states_exact=True, quality_weight=spec['quality_weight'],
            initial_zero_output_native_exact=True, isolated_quality_route_verified=True,
            optimizer_exact_check=optimizer_check, serialization_bytes=serialization_bytes,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(), witnesses=witnesses,
            spec_sha256=sha(args.spec), runner_sha256=sha(__file__))
        write_json(output / 'preflight.json', receipt)
        print('FINAL_QUALITY_PREFLIGHT_COMPLETE ' + json.dumps(receipt), flush=True)
        return

'''
replace_once("    initial_rows, initial_receipt = evaluate('initial')", preflight + "    initial_rows, initial_receipt = evaluate('initial')")
replace_once("            loss = native + correction\n", "            quality, quality_counts = native_final_quality_loss(predictions, batch, matching[1])\n"
    "            loss = native + correction + spec['quality_weight'] * quality\n")
replace_once("                assignment_correction=float(correction), assignment_counts=counts, gradient_norm=float(norm),",
    "                assignment_correction=float(correction), assignment_counts=counts, gradient_norm=float(norm),\n"
    "                quality_loss=float(quality), quality_weight=spec['quality_weight'], quality_counts=quality_counts,")
replace_once("            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),",
    "            quality_weight=spec['quality_weight'],\n"
    "            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),")
replace_once("        fresh_optimizer=True, use_geometry_evidence=spec['use_geometry_evidence'],",
    "        fresh_optimizer=True, use_geometry_evidence=spec['use_geometry_evidence'], quality_weight=spec['quality_weight'],\n"
    "        control_reused=spec['control_root'], quality_target='detached final IoU differences inside original G root pool',")
ast.parse(source, feature_version=(3, 7))
(bundle / 'run_final_quality_fit.py').write_bytes(source.encode('utf-8'))
for path in bundle.glob('*.py'):
    ast.parse(path.read_text(encoding='utf-8'), feature_version=(3, 7))
spec.update(root='/root/autodl-tmp/pvground_final_quality_20261005/quality', quality_weight=1.0,
    control_root='/root/autodl-tmp/pvground_readback_fit_20261004/evidence_visible')
spec['runner_files'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(bundle.glob('*.py'))}
(root / 'quality_fit_spec.json').write_text(json.dumps(spec, indent=2) + '\n', encoding='utf-8')
pre = dict(spec, root='/root/autodl-tmp/pvground_final_quality_20261005/preflight')
(root / 'quality_preflight_spec.json').write_text(json.dumps(pre, indent=2) + '\n', encoding='utf-8')
(root / 'IMPLEMENTATION_STATUS.json').write_text(json.dumps(dict(status='draft_ast37_pass',
    new_training_term=True, added_model_parameters=0, gpu_preflight_executed=False,
    formal_training_started=False, accuracy_result=False,
    control_runner_sha256=hashlib.sha256((control/'runtime_bundle/run_readback_fit.py').read_bytes()).hexdigest()), indent=2)+'\n', encoding='utf-8')
print(json.dumps(dict(status='draft_ast37_pass', files=len(spec['runner_files']), quality_weight=1.0,
    optimizer_steps_per_formal_arm=3723, new_gpu_updates=0)))
