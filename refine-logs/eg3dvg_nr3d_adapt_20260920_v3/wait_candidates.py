import subprocess
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parent
while not (root / 'controller.exit').exists():
    time.sleep(300)
if (root / 'controller.exit').read_text().strip() != '0':
    (root / 'candidate_analysis.exit').write_text('1\n')
    raise SystemExit('Adaptation did not complete cleanly')
with (root / 'candidate_analysis.log').open('xb') as log:
    code = subprocess.call([sys.executable, '-u', str(root / 'analyze_candidates.py'),
                            '--root', str(root / 'evaluation')], stdout=log, stderr=subprocess.STDOUT)
(root / 'candidate_analysis.exit').write_text(str(code) + '\n')
raise SystemExit(code)
