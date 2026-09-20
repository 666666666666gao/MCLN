import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

root = Path('/root/autodl-tmp/mcln_eg3dvg_sr3d_transfer_20260920_v1')
assert not (root / 'failure_visual_export_launch.json').exists()
env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
with (root / 'failure_visual_export_wait.log').open('xb') as log:
    proc = subprocess.Popen([sys.executable, '-u', str(root / 'wait_failure_visuals.py')],
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)
record = {'pid': proc.pid, 'root': str(root), 'state': 'waiting_for_sr_zero_update_formal_audit',
          'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
          'exporter_sha256': hashlib.sha256((root / 'export_failure_visuals.py').read_bytes()).hexdigest(),
          'model_forwards': 0, 'optimizer_steps': 0, 'training_configuration_changed': False}
with (root / 'failure_visual_export_launch.json').open('x') as f:
    json.dump(record, f, indent=2)
print(json.dumps(record))
