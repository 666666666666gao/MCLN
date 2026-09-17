import os,subprocess
from pathlib import Path
root=Path('/root/autodl-tmp/mcln_pvground_referit_serial_queue_20260917_v1')
with (root/'controller.pid').open('x') as f:f.write(str(os.getpid())+'\n')
run=subprocess.run(['/root/miniconda3/envs/bdetr/bin/python','-u',str(root/'scripts/run_pvground_referit_serial_queue.py'),'--spec',str(root/'spec.json')])
with (root/'controller.exit').open('x') as f:f.write(str(run.returncode)+'\n')
raise SystemExit(run.returncode)
