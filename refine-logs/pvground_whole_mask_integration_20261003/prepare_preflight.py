"""Minimally adapt the executed tail preflight; prepare only, never launch it."""
import ast
import hashlib
import json
from pathlib import Path

local = Path(__file__).parent
assert not (local / 'source_preparation.json').exists()
prior = local.parent / 'pvground_fused_support_20261002'
head = local.parent / 'pvground_whole_mask_refiner_20261003'
source = prior / 'run_pvground_tail_support_retry.py'
original = source.read_bytes()
text = original.decode().replace('\r\n', '\n')
text = text[:text.index('    @torch.no_grad()\n    def evaluate(stage):')]
text += "\n\nif __name__ == '__main__':\n    main()\n"


def replace(before, after):
    global text
    assert text.count(before) == 1, before
    text = text.replace(before, after)


replace('"""Paired same-tail raw versus native predicted fused-Mask support."""',
        '"""Actual PV factory and two-step whole-Mask range preflight; no formal fit."""')
replace("choices=['cpu', 'preflight', 'train', 'formal']", "choices=['cpu', 'preflight']")
replace("    assert spec['support_arm'] in ('tail_raw','tail_fused')\n"
        "    assert spec['fused_support'] == (spec['support_arm'] == 'tail_fused')\n",
        "    assert spec['support_arm'] in ('local_range', 'whole_range')\n"
        "    assert spec['use_whole_range'] == (spec['support_arm'] == 'whole_range')\n"
        "    assert spec['fused_support']\n"
        "    for name, digest in spec['whole_range_files'].items():\n"
        "        assert sha(output / name) == digest, name\n")
replace("    from pvground_tail_support_box_refiner import install_tail_support_refinement\n"
        "    install_tail_support_refinement(model, spec['fused_support'])\n",
        "    from pvground_whole_mask_box_refiner import install_whole_mask_refinement\n"
        "    install_whole_mask_refinement(model, spec['use_whole_range'])\n"
        "    assert sum(p.numel() for p in model.candidate_box_refiner.parameters()) == 400614\n")
replace("    from pvground_tail_preflight import observed_forward, native_mask_loss_routes\n",
        "    from pvground_tail_preflight import observed_forward, native_mask_loss_routes\n"
        "    from whole_model_preflight_checks import whole_range_loss_routes, optimizer_restore_exact\n")
replace("                        fused_support=spec['fused_support'], time_cst=now())\n",
        "                        fused_support=spec['fused_support'], use_whole_range=spec['use_whole_range'],\n"
        "                        head_parameters=400614, time_cst=now())\n")
replace("            if not spec['fused_support']:\n"
        "                assert assignment_counts['geometry_to_mask_output_gradient'] == 0\n",
        "            assignment_counts['whole_range_loss_route'] = whole_range_loss_routes(\n"
        "                predictions, spec['use_whole_range'])\n")
replace("        with torch.no_grad():\n"
        "            with_p3,call_witness = observed_forward(model,inputs)\n",
        "        captured = []\n"
        "        handle = model.candidate_box_refiner.register_forward_pre_hook(\n"
        "            lambda module, arguments: captured.append(arguments))\n"
        "        with torch.no_grad():\n"
        "            with_p3,call_witness = observed_forward(model,inputs)\n"
        "            handle.remove()\n"
        "            assert len(captured) == 1\n"
        "            read = model.candidate_box_refiner\n"
        "            pair_outputs = []\n"
        "            for condition in (False, True):\n"
        "                read.use_whole_range = condition\n"
        "                pair_outputs.append(read(*captured[0]))\n"
        "            read.use_whole_range = spec['use_whole_range']\n"
        "            assert all(torch.equal(a, b) for a, b in zip(*pair_outputs))\n"
        "            assert torch.equal(pair_outputs[0][0], captured[0][2])\n"
        "            assert torch.equal(pair_outputs[0][1], captured[0][3])\n"
        "            del captured, pair_outputs\n")
replace("                if spec['fused_support']:\n"
        "                    assert record['geometry_to_mask_output_gradient'] > 0\n",
        "                assert record['geometry_to_mask_output_gradient'] > 0\n"
        "                if spec['use_whole_range']:\n"
        "                    route = record['whole_range_loss_route']\n"
        "                    assert route['range_gradient'] > 0\n"
        "                    assert all(value > 0 for value in route['global_only_mask_gradients'].values())\n")
replace("        assert all(int(state['step'])==2 for state in optimizer.state.values())\n",
        "        assert all(int(state['step'])==2 for state in optimizer.state.values())\n"
        "        restored_optimizer_check = optimizer_restore_exact(optimizer, reloaded['optimizer'])\n")
replace("            direct_native_mask_loss_to_refiner_gradients_zero=True)\n",
        "            direct_native_mask_loss_to_refiner_gradients_zero=True,\n"
        "            whole_range_model=True, use_whole_range=spec['use_whole_range'],\n"
        "            same_cached_inputs_zero_head_pair_exact=True, head_parameters=400614,\n"
        "            optimizer_exact_check=restored_optimizer_check, weight_files_created=0,\n"
        "            peak_reserved_bytes=torch.cuda.max_memory_reserved())\n")
replace('def main():\n    parser = argparse.ArgumentParser()\n', 'def main():\n    preflight_begin = time.perf_counter()\n    parser = argparse.ArgumentParser()\n')
replace('    import torch\n    from torch.utils.data import DataLoader, Subset\n', "    import torch\n    from torch.utils.data import DataLoader, Subset\n    if args.mode == 'preflight':\n        torch.cuda.reset_peak_memory_stats()\n")
replace('        torch.cuda.reset_peak_memory_stats()\n        steps=[]\n', '        steps=[]\n')
replace("        if args.mode == 'preflight':\n            predictions,call_witness = observed_forward(model,inputs)\n", "        if args.mode == 'preflight':\n            torch.cuda.synchronize(); forward_begin = time.perf_counter()\n            predictions,call_witness = observed_forward(model,inputs)\n            torch.cuda.synchronize(); forward_seconds = time.perf_counter() - forward_begin\n")
replace("            assignment_counts['whole_range_loss_route'] = whole_range_loss_routes(\n                predictions, spec['use_whole_range'])\n", "            assignment_counts['whole_range_loss_route'] = whole_range_loss_routes(\n                predictions, spec['use_whole_range'])\n            assignment_counts['full_model_forward_seconds'] = forward_seconds\n")
replace('        with torch.no_grad():\n            with_p3,call_witness = observed_forward(model,inputs)\n', '        with torch.no_grad():\n            torch.cuda.synchronize(); forward_begin = time.perf_counter()\n            with_p3,call_witness = observed_forward(model,inputs)\n            torch.cuda.synchronize(); initial_forward_seconds = time.perf_counter() - forward_begin\n')
replace('            pair_outputs = []\n', '            torch.cuda.synchronize(); replay_begin = time.perf_counter()\n            pair_outputs = []\n')
replace("            read.use_whole_range = spec['use_whole_range']\n", "            read.use_whole_range = spec['use_whole_range']\n            torch.cuda.synchronize(); pair_replay_seconds = time.perf_counter() - replay_begin\n")
replace('            without_p3,call_witness = observed_forward(model,inputs)\n', '            torch.cuda.synchronize(); forward_begin = time.perf_counter()\n            without_p3,call_witness = observed_forward(model,inputs)\n            torch.cuda.synchronize(); native_forward_seconds = time.perf_counter() - forward_begin\n')
replace('            peak_reserved_bytes=torch.cuda.max_memory_reserved())\n', '            peak_reserved_bytes=torch.cuda.max_memory_reserved(),\n            memory_peak_scope="CUDA allocator from before native factory through restore",\n            initial_full_forward_seconds=initial_forward_seconds,\n            cached_information_pair_replay_seconds=pair_replay_seconds,\n            native_without_refiner_forward_seconds=native_forward_seconds,\n            runner_wall_seconds_through_restore=time.perf_counter() - preflight_begin)\n')
runner = text.encode()
ast.parse(runner)
(local / 'run_whole_mask_preflight.py').write_bytes(runner)
copied = {}
for name in ('pvground_whole_mask_box_refiner.py', 'whole_mask_range.py',
             'pvground_candidate_box_refiner.py', 'pvground_tail_support_box_refiner.py'):
    raw = (head / name).read_bytes()
    (local / name).write_bytes(raw)
    copied[name] = dict(source=str(head / name), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
for name in ('pvground_tail_preflight.py', 'pvground_semantic_assignment.py', 'pvground_source_query.py',
             'pvground_observation_query.py', 'pvground_task_observation_query.py'):
    src = prior / 'complete_tail_fused_retry/source' / name
    raw = src.read_bytes()
    (local / name).write_bytes(raw)
    copied[name] = dict(source=str(src), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
whole_files = {name: hashlib.sha256((local / name).read_bytes()).hexdigest() for name in
               ('pvground_whole_mask_box_refiner.py', 'whole_mask_range.py', 'whole_model_preflight_checks.py')}
template = json.loads((prior / 'preflight_tail_fused_spec.json').read_bytes())
for key in ('files', 'disk_budget', 'disk_free_before', 'reused_control_root', 'prior_p3_root'):
    template.pop(key)
template.update(whole_range_files=whole_files,
    comparison='engineering preflight only: original protected G, one cached native input, whole-range information on/off',
    preflight_only_at_install=True)
for arm in ('local_range', 'whole_range'):
    spec = dict(template, root='/root/autodl-tmp/pvground_whole_mask_preflight_20261003/' + arm,
                support_arm=arm, use_whole_range=arm == 'whole_range', fused_support=True)
    (local / (arm + '_spec.json')).write_text(json.dumps(spec, indent=2) + '\n', encoding='utf-8')
record = dict(status='prepared_not_executed', source=str(source),
    prior_runner_sha256=hashlib.sha256(original).hexdigest(), runner_sha256=hashlib.sha256(runner).hexdigest(),
    runner_bytes=len(runner), copied=copied, local_AST_pass=True, deployment_prepared=False,
    GPU_launched=False, active_training_modified=False, formal_training_supported=False,
    weight_files_created=0, scope='source preparation only; fixed original G engineering preflight')
(local / 'source_preparation.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
print(json.dumps(record))
