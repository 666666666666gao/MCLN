import subprocess
from pathlib import Path
root=Path('/root/autodl-tmp/mcln_pvground_scanrefer_endpoint_audit_20260908_v1')
result=subprocess.run(['/root/miniconda3/envs/bdetr/bin/python','-u',str(root/'queue.py')])
(root/'controller.exit').write_text(str(result.returncode)+'\n')
raise SystemExit(result.returncode)
