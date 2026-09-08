"""CPU-only geometry at fixed query indices; no claimed model neighborhood read."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

root = Path('/root/autodl-tmp/mcln_pvground_support_capture_20260908_v1')
replay = root / 'observed_replay'
receipt = json.loads((replay / 'receipt.json').read_bytes())
support = json.loads((replay / 'support_receipt.json').read_bytes())
trace_path = replay / 'reference_trace.pt'
assert hashlib.sha256(trace_path.read_bytes()).hexdigest() == receipt['reference_trace_sha256']
trace = torch.load(str(trace_path), map_location='cpu')
inputs = json.loads((root / 'inputs/receipt.json').read_bytes())
records = []
for record in support['records']:
    path = replay / 'support' / record['packet']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record['packet_sha256']
    packet = np.load(str(path))
    report = json.loads(path.with_suffix('.json').read_bytes())
    assert hashlib.sha256(path.with_suffix('.json').read_bytes()).hexdigest() == record['report_sha256']
    rows = report['rows']
    observed = dict(available_min=min(r['available_support_rows'] for r in rows),
                    available_median=float(np.median([r['available_support_rows'] for r in rows])),
                    repeated_slot_queries=sum(r['selected_unique_rows'] < r['selected_slots'] for r in rows),
                    wrong_batch_slots=sum(r['selected_wrong_batch_slots'] for r in rows),
                    outside_radius_slots=sum(r['selected_outside_radius_slots'] for r in rows))
    for name in ['output.query_points_xyz', 'output.last_center']:
        query = trace[name].numpy().astype(np.float64)
        assert query.shape == (8, 256, 3)
        counts, nearest = [], []
        for batch in range(8):
            xyz = packet['support_xyz'][packet['support_batch'] == batch].astype(np.float64)
            for center in query[batch, ::8]:
                distances = np.linalg.norm(xyz - center, axis=1)
                counts.append(int((distances < record['radius_m']).sum()))
                nearest.append(float(distances.min()))
        records.append(dict(source=record['source'], radius_m=record['radius_m'],
                            position=name, queries=len(counts), empty=sum(n == 0 for n in counts),
                            support_median=float(np.median(counts)), nearest_max=max(nearest),
                            observed_vsa=observed))
result = dict(scope='CPU hypothetical center neighborhoods, not executed candidate feature reads',
              input_sha256=receipt['input_sha256'], trace_sha256=receipt['reference_trace_sha256'],
              scan_ids=[row['scan_id'] for row in inputs['rows']],
              sampling='32 indices per row: 0,8,...,248; no GT or score selection',
              model_forwards=0, optimizer_steps=0, cuda_initialized=torch.cuda.is_initialized(), records=records)
print(json.dumps(result, allow_nan=False))
