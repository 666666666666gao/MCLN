"""Fix the source review's concrete peak/timing gap; preserve the original."""
import ast
import datetime
import hashlib
import json
from pathlib import Path

local = Path(__file__).parent
assert not (local / 'instrumentation_patch.json').exists()
review = json.loads((local / 'EXPERIMENT_CODE_REVIEW.json').read_bytes())
assert review['execution_scope'] == 'SOURCE_ONLY' and not review['blocking_findings']
trace = local / '.aris/traces/experiment-bridge/2026-10-03_whole_model_preflight_run01'
for name, target in [('EXPERIMENT_CODE_REVIEW.md', '001-code-review.report.md'),
                     ('EXPERIMENT_CODE_REVIEW.json', '001-code-review.report.json'),
                     ('run_whole_mask_preflight.py', '001-runner-before-instrumentation.py'),
                     ('prepare_preflight.py', '001-builder-before-instrumentation.py')]:
    assert not (trace / target).exists()
    (trace / target).write_bytes((local / name).read_bytes())
changes = [
    ('def main():\n    parser = argparse.ArgumentParser()\n',
     'def main():\n    preflight_begin = time.perf_counter()\n    parser = argparse.ArgumentParser()\n'),
    ('    import torch\n    from torch.utils.data import DataLoader, Subset\n',
     "    import torch\n    from torch.utils.data import DataLoader, Subset\n"
     "    if args.mode == 'preflight':\n"
     "        torch.cuda.reset_peak_memory_stats()\n"),
    ('        torch.cuda.reset_peak_memory_stats()\n        steps=[]\n', '        steps=[]\n'),
    ("        if args.mode == 'preflight':\n"
     "            predictions,call_witness = observed_forward(model,inputs)\n",
     "        if args.mode == 'preflight':\n"
     "            torch.cuda.synchronize(); forward_begin = time.perf_counter()\n"
     "            predictions,call_witness = observed_forward(model,inputs)\n"
     "            torch.cuda.synchronize(); forward_seconds = time.perf_counter() - forward_begin\n"),
    ("            assignment_counts['whole_range_loss_route'] = whole_range_loss_routes(\n"
     "                predictions, spec['use_whole_range'])\n",
     "            assignment_counts['whole_range_loss_route'] = whole_range_loss_routes(\n"
     "                predictions, spec['use_whole_range'])\n"
     "            assignment_counts['full_model_forward_seconds'] = forward_seconds\n"),
    ('        with torch.no_grad():\n            with_p3,call_witness = observed_forward(model,inputs)\n',
     '        with torch.no_grad():\n'
     '            torch.cuda.synchronize(); forward_begin = time.perf_counter()\n'
     '            with_p3,call_witness = observed_forward(model,inputs)\n'
     '            torch.cuda.synchronize(); initial_forward_seconds = time.perf_counter() - forward_begin\n'),
    ('            pair_outputs = []\n',
     '            torch.cuda.synchronize(); replay_begin = time.perf_counter()\n'
     '            pair_outputs = []\n'),
    ("            read.use_whole_range = spec['use_whole_range']\n",
     "            read.use_whole_range = spec['use_whole_range']\n"
     '            torch.cuda.synchronize(); pair_replay_seconds = time.perf_counter() - replay_begin\n'),
    ('            without_p3,call_witness = observed_forward(model,inputs)\n',
     '            torch.cuda.synchronize(); forward_begin = time.perf_counter()\n'
     '            without_p3,call_witness = observed_forward(model,inputs)\n'
     '            torch.cuda.synchronize(); native_forward_seconds = time.perf_counter() - forward_begin\n'),
    ('            peak_reserved_bytes=torch.cuda.max_memory_reserved())\n',
     '            peak_reserved_bytes=torch.cuda.max_memory_reserved(),\n'
     '            memory_peak_scope="CUDA allocator from before native factory through restore",\n'
     '            initial_full_forward_seconds=initial_forward_seconds,\n'
     '            cached_information_pair_replay_seconds=pair_replay_seconds,\n'
     '            native_without_refiner_forward_seconds=native_forward_seconds,\n'
     '            runner_wall_seconds_through_restore=time.perf_counter() - preflight_begin)\n'),
]
runner_path = local / 'run_whole_mask_preflight.py'
before = runner_path.read_bytes()
text = before.decode()
for old, new in changes:
    assert text.count(old) == 1, old
    text = text.replace(old, new)
after = text.encode()
ast.parse(after)
builder_path = local / 'prepare_preflight.py'
builder_before = builder_path.read_bytes()
builder = builder_before.decode()
assert builder.count('runner = text.encode()') == 1
insertion = '\n'.join('replace(' + repr(old) + ', ' + repr(new) + ')' for old, new in changes)
builder = builder.replace('runner = text.encode()', insertion + '\nrunner = text.encode()')
builder_after = builder.encode()
ast.parse(builder_after)
runner_path.write_bytes(after)
builder_path.write_bytes(builder_after)
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(),
    cause='Concrete review finding: original peak reset omitted initial full forwards; initial timings absent',
    runner_sha256_before=hashlib.sha256(before).hexdigest(), runner_sha256_after=hashlib.sha256(after).hexdigest(),
    builder_sha256_before=hashlib.sha256(builder_before).hexdigest(), builder_sha256_after=hashlib.sha256(builder_after).hexdigest(),
    original_source_preparation_receipt_preserved=True, original_review_preserved=True,
    loss_or_model_or_spec_changed=False, real_model_executed=False, GPU_launched=False,
    weight_files_created=0)
(local / 'instrumentation_patch.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
print(json.dumps(record))
