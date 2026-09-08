import os,subprocess
from pathlib import Path
root=Path(__file__).parent
(root/'controller.pid').write_text(str(os.getpid())+'\n')
with (root/'run.log').open('xb') as stream:
    result=subprocess.run(['/root/miniconda3/envs/bdetr/bin/python','-u',str(root/'queue.py')],stdout=stream,stderr=subprocess.STDOUT)
(root/'controller.exit').write_text(str(result.returncode)+'\n')
raise SystemExit(result.returncode)
