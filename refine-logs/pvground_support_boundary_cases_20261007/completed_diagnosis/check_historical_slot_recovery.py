"""A red-capable check of the actual historical coarse-box recovery claim."""
import json
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent
cases = json.loads((root / 'case_manifest.json').read_bytes())['diagnostic_rows']
case = next(item['cached'] for item in cases if item['cached']['row_id'] == 128)
with np.load(root / 'complete/arrays/row_00128.npz', allow_pickle=False) as arrays:
    query = int(arrays['queries'][1])
    actual = arrays['coarse_boxes'][1].astype(np.float64)
expected = np.asarray(case['prior_box'], dtype=np.float64)
assert query == case['query']
error = float(np.max(np.abs(actual - expected)))
print(json.dumps(dict(row_id=128, historical_numeric_query=query,
    actual_coarse_box=actual.tolist(), expected_coarse_box=expected.tolist(),
    max_absolute_difference_m=error, tolerance_m=2e-6)), flush=True)
assert error <= 2e-6, 'Historical numeric slot does not recover the historical prediction'
