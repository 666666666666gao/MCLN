"""Prepare a closed-control-only intake and CPU recount while member fit runs."""
import ast
from pathlib import Path


root = Path(__file__).resolve().parent
old = root.parent / 'pvground_query_supported_geometry_20261005'
source = (old / 'collect_closed_control_receipts.py').read_text(encoding='utf-8')
changes = {
    "observation = json.loads((local / 'fit_observations/observation_9.json').read_bytes())":
        "observation_path=local/'fit_observations/observation_11.json'\nobservation=json.loads(observation_path.read_bytes())",
    "observation['status']['arm'] == 'query_supported'": "observation['status']['arm'] == 'member_target'",
    "'control/train.exit', 'control/formal.exit']":
        "'control/train.exit', 'control/formal.exit', 'control/formal/rows.jsonl',\n         'control/initial/receipt.json', 'control/terminal/receipt.json']",
    "assert restore['terminal_sha256'] == fit['terminal_sha256']":
        "assert restore['terminal_sha256'] == fit['terminal_sha256']\nassert identities['control/formal/rows.jsonl']['sha256']==formal['rows_sha256']",
    'downloaded_weights=0, downloaded_full_rows=0': 'downloaded_weights=0, downloaded_full_rows=9508',
}
for before, after in changes.items():
    assert source.count(before)==1, before
    source=source.replace(before,after)
assert not (root/'collect_closed_control_receipts.py').exists()
ast.parse(source)
(root/'collect_closed_control_receipts.py').write_text(source,encoding='utf-8')
print('Prepared closed-control intake; not executed or polled.')
