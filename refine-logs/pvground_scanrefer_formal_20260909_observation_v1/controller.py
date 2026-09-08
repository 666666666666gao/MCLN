import subprocess
from pathlib import Path
root=Path('/root/autodl-tmp/mcln_pvground_scanrefer_formal_20260909_observation_v1')
result=subprocess.run(['/root/miniconda3/envs/bdetr/bin/python','-u',str(root/'formal_queue.py')])
(root/'controller.exit').write_text(str(result.returncode)+'\n')
raise SystemExit(result.returncode)
