import subprocess
from pathlib import Path
root=Path('/root/autodl-tmp/mcln_pvground_observation_checkpoint_audit_20260909_v1')
result=subprocess.run(['/root/miniconda3/envs/bdetr/bin/python','-u',str(root/'checkpoint_queue.py')])
(root/'controller.exit').write_text(str(result.returncode)+'\n')
raise SystemExit(result.returncode)
