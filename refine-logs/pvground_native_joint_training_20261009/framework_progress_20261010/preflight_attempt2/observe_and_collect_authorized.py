"""One scheduled observer, followed only on success by the closed intake."""
import json
from pathlib import Path
import runpy

root = Path(__file__).resolve().parent
runpy.run_path(str(root / 'observe_authorized.py'), run_name='__main__')
closed = json.loads((root / 'NATIVE_PREFLIGHT_WAIT.json').read_bytes())
assert closed['observer_closed'] and closed['terminal']['exitcode'] == 0
runpy.run_path(str(root / 'collect_authorized.py'), run_name='__main__')
