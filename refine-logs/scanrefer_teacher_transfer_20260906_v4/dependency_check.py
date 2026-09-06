import hashlib, json
from pathlib import Path
directory = Path('/root/autodl-tmp/mcln_scanrefer_teacher_transfer_20260906_v4')
manifest = json.loads((directory / 'input_manifest.json').read_text())
probe = Path(manifest['prerequisite_probe'])
assert (probe / 'controller.exit').read_text().strip() == '0'
receipt = json.loads((probe / 'receipt.json').read_text())
assert receipt['status'] == 'pass' and receipt['optimizer_steps'] == 0
assert receipt['manifest_sha256'] == manifest['prerequisite_manifest_sha256']
assert hashlib.sha256((directory / 'cpu_receipt.json').read_bytes()).hexdigest() == manifest['cpu_receipt_sha256']
print('Prerequisite native probe passed; ScanRefer teacher audit starts with zero updates', flush=True)
