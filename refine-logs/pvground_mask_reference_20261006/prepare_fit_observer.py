"""Reuse the single estimated observer and collect only after controller closure."""
import ast
from pathlib import Path

root=Path(__file__).resolve().parent
old=root.parent/'pvground_support_reference_20261005/observe_fit_authorized.py'
text=old.read_text(encoding='utf-8').replace('control_spec.json','native_reference_spec.json')
text=text.replace('import time\n','import time\nimport sys\nimport subprocess\n')
text=text.replace("        client.close()\n        raise SystemExit(0)",
    "        client.close()\n        assert exitcode==0 and status['status']=='complete'\n        subprocess.check_call([sys.executable,'-B',str(local/'collect_closed_fit_authorized.py')])\n        raise SystemExit(0)")
ast.parse(text)
(root/'observe_fit_authorized.py').write_text(text,encoding='utf-8')
print('FIT_OBSERVER_PREPARED_NOT_STARTED')
