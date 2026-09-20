import os,subprocess,sys
from pathlib import Path
r=Path(__file__).resolve().parent
controller=r/'inspect_wait.py'
controller.write_text("import subprocess,sys,time\nfrom pathlib import Path\nr=Path(__file__).resolve().parent\nwhile not (r/'checkpoint_download.json').exists():time.sleep(180)\nwith (r/'checkpoint_inspection.log').open('xb') as f:\n code=subprocess.call([sys.executable,'-u',str(r/'inspect_checkpoint.py')],stdout=f,stderr=subprocess.STDOUT)\n(r/'checkpoint_inspection.exit').write_text(str(code)+'\\n')\n")
with (r/'inspect_wait.log').open('xb') as log:
 p=subprocess.Popen([sys.executable,'-u',str(controller)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print('inspection_queue_pid',p.pid)
