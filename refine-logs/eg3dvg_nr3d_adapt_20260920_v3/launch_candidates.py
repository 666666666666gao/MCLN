import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

root = Path('/root/autodl-tmp/mcln_eg3dvg_nr3d_adapt_20260920_v3')
assert not (root / 'candidate_analysis_launch.json').exists()
env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
with (root / 'candidate_analysis_wait.log').open('xb') as log:
    proc = subprocess.Popen([sys.executable, '-u', str(root / 'wait_candidates.py')],
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)
record = {'pid': proc.pid, 'root': str(root), 'state': 'waiting_for_fixed_endpoint',
          'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
          'analyzer_sha256': hashlib.sha256((root / 'analyze_candidates.py').read_bytes()).hexdigest(),
          'model_forwards': 0, 'optimizer_steps': 0, 'training_configuration_changed': False}
with (root / 'candidate_analysis_launch.json').open('x') as f:
    json.dump(record, f, indent=2)
print(json.dumps(record))
