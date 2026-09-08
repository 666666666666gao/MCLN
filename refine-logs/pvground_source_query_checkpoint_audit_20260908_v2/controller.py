import subprocess
from pathlib import Path
root=Path('/root/autodl-tmp/mcln_pvground_source_query_checkpoint_audit_20260908_v2')
result=subprocess.run(['/root/miniconda3/envs/bdetr/bin/python','-u',str(root/'checkpoint_queue.py')])
(root/'controller.exit').write_text(str(result.returncode)+'\n')
raise SystemExit(result.returncode)
