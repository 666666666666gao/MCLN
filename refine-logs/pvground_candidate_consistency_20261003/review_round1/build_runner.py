"""Reuse the sealed G continuation runner; change only contrastive targets."""
import hashlib
import json
from pathlib import Path
import py_compile
import shutil

local = Path(__file__).parent
parent = Path(r'C:\Users\gb\.codex\tmp\pvground_g_p2_20261002\complete\source')
text = (parent / 'run.py').read_text(encoding='utf-8')

def replace(before, after):
    global text
    assert text.count(before) == 1, before
    text = text.replace(before, after)

replace("    spec = json.loads(args.spec.read_bytes())\n",
        "    spec = json.loads(args.spec.read_bytes())\n    assert not spec['p2']\n")
replace("    from pvground_semantic_assignment import semantic_assignment_correction, verify_native_replacement\n",
        "    from pvground_semantic_assignment import semantic_assignment_correction, verify_native_replacement\n"
        "    assert sha(output/'pvground_candidate_consistency.py') == spec['consistency_module_sha256']\n"
        "    from pvground_candidate_consistency import candidate_consistency_correction, verify_consistency_replacement\n")
replace("        assert terminal['p2'] == spec['p2'] and terminal['step'] == 3723\n",
        "        assert terminal['p2'] == spec['p2'] and terminal['step'] == 3723\n"
        "        assert terminal['semantic_consistency'] == spec['semantic_consistency']\n")
replace("        loss=native+correction\n",
        "        contrastive_correction = native.new_zeros(())\n"
        "        assignment_counts['semantic_consistency'] = spec['semantic_consistency']\n"
        "        if spec['semantic_consistency']:\n"
        "            contrastive_correction, contrastive_counts, selected = candidate_consistency_correction(\n"
        "                predictions, batch, matching[1], set_criterion)\n"
        "            assignment_counts.update(contrastive_counts)\n"
        "            if not update:\n"
        "                assignment_counts['contrastive_witness'] = verify_consistency_replacement(\n"
        "                    predictions, batch, matching[1], set_criterion, contrastive_correction, selected)\n"
        "        assignment_counts['loss_contrastive_correction'] = float(contrastive_correction)\n"
        "        loss=native+correction+contrastive_correction\n")
begin = text.index("    if args.mode == 'preflight':\n")
end = text.index("    @torch.no_grad()\n    def evaluate(stage):", begin)
text = text[:begin] + '''    if args.mode == 'preflight':
        assert spec['semantic_consistency']
        probe_batch = next(iter(loader('fit', True, spec['seed'])))
        reset_rng(spec['seed']); model.train()
        optimizer = BaseTrainTester.get_optimizer(training, model)
        torch.cuda.reset_peak_memory_stats()
        capacity = step(probe_batch, optimizer, False)
        assert capacity['contrastive_reassigned_queries'] > 0
        assert not optimizer.state
        steps = [step(probe_batch, optimizer, True) for _ in range(2)]
        selected_names = set(trainable) | {n for n, _ in model.named_buffers() if n in initial}
        stream = io.BytesIO()
        torch.save({'delta': {n: v.detach().cpu() for n, v in model.state_dict().items() if n in selected_names},
                    'optimizer': optimizer.state_dict()}, stream)
        stream.seek(0); reloaded = torch.load(stream, map_location='cpu')
        restored = dict(initial); restored.update(reloaded['delta'])
        model.load_state_dict(restored, strict=True)
        optimizer.load_state_dict(reloaded['optimizer'])
        assert all(torch.equal(v.detach().cpu(), restored[n]) for n, v in model.state_dict().items())
        assert all(int(value['step']) == 2 for value in optimizer.state.values())
        receipt = dict(status='pass', time_cst=now(), batch_size=8, optimizer_steps=2,
            p2=False, semantic_consistency=True, g_strict_restore=True, new_model_states=0,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), capacity=capacity, steps=steps,
            serialization_bytes=stream.getbuffer().nbytes, optimizer_restore=True, formal_rows=0)
        write_json(output/'preflight.json', receipt)
        print('G_CANDIDATE_CONSISTENCY_PREFLIGHT_PASS '+json.dumps(receipt), flush=True)
        return

''' + text[end:]
replace("                    p2=spec['p2'],base_terminal_sha256=spec['base_terminal_sha256'])\n",
        "                    p2=spec['p2'],base_terminal_sha256=spec['base_terminal_sha256'],\n"
        "                    semantic_consistency=spec['semantic_consistency'],\n"
        "                    consistency_module_sha256=spec['consistency_module_sha256'])\n")
replace("                   p2=spec['p2'],base_terminal_sha256=spec['base_terminal_sha256'],fresh_optimizer=True)\n",
        "                   p2=spec['p2'],base_terminal_sha256=spec['base_terminal_sha256'],fresh_optimizer=True,\n"
        "                   semantic_consistency=spec['semantic_consistency'],\n"
        "                   consistency_module_sha256=spec['consistency_module_sha256'])\n")
(local / 'run.py').write_text(text, encoding='utf-8')
for name in ('pvground_semantic_assignment.py', 'pvground_task_observation_query.py',
             'pvground_observation_query.py', 'pvground_source_query.py'):
    shutil.copyfile(parent / name, local / name)
for name in ('run.py', 'pvground_candidate_consistency.py', 'cpu_test.py'):
    py_compile.compile(str(local / name), doraise=True)
record = dict(parent_runner=str(parent / 'run.py'), model_architecture_changed=False,
              new_model_parameters=0, regression_matching_changed=False,
              changed_training_path='last-layer semantic contrastive target replacement',
              generated_runner_sha256=hashlib.sha256((local / 'run.py').read_bytes()).hexdigest())
(local / 'runner_preparation.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
print(json.dumps(record))
