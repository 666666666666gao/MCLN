import os,subprocess
from pathlib import Path
root=Path(__file__).parent
(root/"controller.pid").write_text(str(os.getpid())+"\n")
with (root/"queue.log").open("xb") as log:
    result=subprocess.run(["/root/miniconda3/envs/bdetr/bin/python","-u",str(root/"queue.py")],stdout=log,stderr=subprocess.STDOUT)
(root/"controller.exit").write_text(str(result.returncode)+"\n")
raise SystemExit(result.returncode)
