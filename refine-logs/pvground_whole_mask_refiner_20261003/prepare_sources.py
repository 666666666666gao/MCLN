"""Prepare exact existing dependencies for the standalone CPU head witness."""
import hashlib
import json
from pathlib import Path
import py_compile

local = Path(__file__).parent
previous = local.parent / 'pvground_whole_mask_range_20261003'
tail = local.parent / 'pvground_fused_support_20261002/complete_tail_fused_retry/source'
assert json.loads((previous / 'cpu_receipt.json').read_bytes())['status'] == 'pass'
sources = {'whole_mask_range.py': previous / 'whole_mask_range.py',
    'pvground_candidate_box_refiner.py': tail / 'pvground_candidate_box_refiner.py',
    'pvground_tail_support_box_refiner.py': tail / 'pvground_tail_support_box_refiner.py',
    'native_losses.py': tail / 'imported/models.losses.py',
    'native_pv_ground.py': tail / 'imported/models.pv_ground.py'}
record = {}
for name, path in sources.items():
    raw = path.read_bytes()
    destination = local / name
    assert not destination.exists()
    destination.write_bytes(raw)
    record[name] = dict(source=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
for name in ('whole_mask_range.py', 'pvground_candidate_box_refiner.py',
             'pvground_tail_support_box_refiner.py', 'pvground_whole_mask_box_refiner.py',
             'cpu_head_witness.py', 'run_cpu_head_authorized.py'):
    py_compile.compile(str(local / name), doraise=True)
assert record['native_losses.py']['sha256'] == '920051cc7d1009493829eebbb6185da834a4efd037e9d73ad43ce918a81031de'
(local / 'source_preparation.json').write_text(json.dumps(dict(status='prepared',
    existing_dependency_sources=record, compiled=True, CPU_execution_started=False,
    active_training_modified=False, model_installed=False), indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(status='prepared', existing_dependency_count=len(record), compiled=True)))
