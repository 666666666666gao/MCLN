import subprocess,time
from pathlib import Path
r=Path(__file__).resolve().parent
while not (r/'controller.exit').exists():time.sleep(300)
if (r/'controller.exit').read_text().strip()!='0':raise SystemExit('Adaptation did not complete cleanly')
with (r/'paired_rec.log').open('xb') as f: code=subprocess.call(['/root/mcln_eg3dvg_torch112_20260920_v1/venv/bin/python','-u',str(r/'analyze_pair.py'),'--start','/root/autodl-tmp/mcln_eg3dvg_nr3d_transfer_20260920_v1','--adaptation',str(r)],stdout=f,stderr=subprocess.STDOUT)
(r/'paired_rec.exit').write_text(str(code)+'\n')
raise SystemExit(code)
