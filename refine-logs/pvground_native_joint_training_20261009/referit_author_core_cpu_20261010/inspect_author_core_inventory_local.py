"""Compare archived author tensor inventories; no model or remote call."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
repo = Path('C:/Users/gb/.codex_mcln_g0_20260905/refine-logs')
names = dict(scanrefer='pvground_scan_checkpoint_inspection_20260908_v1',
    nr3d='pvground_nr_checkpoint_inspection_20260908_v2',
    sr3d='pvground_sr_checkpoint_inspection_20260908_v1')
inputs = {}
inventories = {}
for dataset, leaf in names.items():
    path = repo / leaf / 'state_inventory.json'
    raw = path.read_bytes()
    inventory = json.loads(raw)
    assert all(name.startswith('module.') for name in inventory)
    inventories[dataset] = {name[7:]:meta for name,meta in inventory.items()}
    inputs[str(path)] = dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
scan = inventories['scanrefer']
comparisons = {}
for dataset in ('nr3d','sr3d'):
    other = inventories[dataset]
    comparisons[dataset] = dict(tensors=len(other),
        author_only=sorted(set(other)-set(scan)), scan_only=sorted(set(scan)-set(other)),
        common_shape_dtype_mismatches=[name for name in sorted(set(scan)&set(other))
            if scan[name]['shape'] != other[name]['shape'] or scan[name]['dtype'] != other[name]['dtype']],
        author_only_metadata={name:other[name] for name in sorted(set(other)-set(scan))})
value = dict(scope='Current local archived author inventory comparison only; no current checkpoint/model load or tensor-value check',
    ScanRefer_tensors=len(scan), comparisons=comparisons, inputs=inputs,
    neural_calls=0, GPU_calls=0, remote_training_queries=0)
target = root / 'AUTHOR_CORE_INVENTORY_LOCAL_COMPARISON.json'
assert not target.exists()
target.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
print(json.dumps(value, indent=2))
