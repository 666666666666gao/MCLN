import subprocess
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parent
while not (root / 'controller.exit').exists():
    time.sleep(300)
if (root / 'controller.exit').read_text().strip() != '0':
    (root / 'failure_visual_export.exit').write_text('1\n')
    raise SystemExit('Sr3D zero-update evaluation did not complete cleanly')
with (root / 'failure_visual_export.log').open('xb') as log:
    code = subprocess.call([sys.executable, '-u', str(root / 'export_failure_visuals.py'),
                            '--datasets', 'Sr3D', '--out',
                            '/root/autodl-tmp/mcln_eg3dvg_sr3d_failure_visuals_20260920_v1'], stdout=log, stderr=subprocess.STDOUT)
(root / 'failure_visual_export.exit').write_text(str(code) + '\n')
raise SystemExit(code)
