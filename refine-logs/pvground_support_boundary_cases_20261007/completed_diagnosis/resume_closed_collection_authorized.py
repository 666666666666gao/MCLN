"""Resume the intentionally stopped slow transfer; prefetch remaining exact files."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex

import paramiko

local = Path(__file__).resolve().parent
assert not (local / 'complete/INTAKE.json').exists()
transfer_stop = json.loads((local / 'transfer_stop.json').read_bytes())
assert transfer_stop['original_collector_closed'] and transfer_stop['resume_reason'] == 'measured_serial_sftp_slow'
wait = json.loads((local / 'wait.json').read_bytes())
assert wait['observer_closed'] and not wait['terminal']['controller_alive']
assert wait['terminal']['exitcode'] == 0 and wait['terminal']['status']['status'] == 'complete'
spec = json.loads((local / 'diagnostic_spec.json').read_bytes())
client = paramiko.SSHClient(); client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
code = '''import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]);receipt=json.loads((root/'receipt.json').read_bytes())
assert receipt['status']=='PASS_ACTUAL_READ_ONLY_DIAGNOSIS' and receipt['cases']==191
assert receipt['optimizer_updates']==receipt['weights_created']==0
assert json.loads((root/'status.json').read_bytes())['status']=='complete'
assert (root/'controller.exit').read_text().strip()=='0'
names=['receipt.json','rows.jsonl','collector.log','collector.exit','controller.log','controller.exit','status.json','imports.json','load.json','diagnostic_spec.json','case_manifest.json','collect_support_cases.py','support_evidence.py','controller.py']
arrays=sorted((root/'arrays').glob('row_*.npz'));assert len(arrays)==191
names += [str(path.relative_to(root)) for path in arrays]
items=[]
for name in names:
    path=root/name;raw=path.read_bytes();items.append(dict(name=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
print(json.dumps(dict(files=items,cases=191,bytes=sum(item['bytes'] for item in items))))
'''
_, stdout, stderr = client.exec_command(shlex.join([spec['runtime']+'/venv/bin/python', '-B', '-c', code, spec['root']]), timeout=120)
raw = stdout.read(); assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
intake = json.loads(raw)
destination = local / 'complete'; destination.mkdir(exist_ok=True)
sftp = client.open_sftp()
for item in intake['files']:
    relative = Path(item['name'])
    assert not relative.is_absolute() and '..' not in relative.parts
    path = destination / relative; path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        saved = path.read_bytes()
        assert len(saved) == item['bytes'] and hashlib.sha256(saved).hexdigest() == item['sha256']
        continue
    with sftp.open(spec['root']+'/'+item['name'], 'rb') as stream:
        stream.prefetch(file_size=item['bytes'])
        raw = stream.read()
    assert len(raw) == item['bytes'] and hashlib.sha256(raw).hexdigest() == item['sha256']
    path.write_bytes(raw)
sftp.close(); client.close()
intake.update(transfer_resumed=True, prefetch=True, original_collector_interruption='intentional after measured slow transfer', time_cst=datetime.datetime.now().astimezone().isoformat(), status='CLOSED_ARTIFACTS_COLLECTED', neural_replay=False,
    optimizer_updates=0, weights_copied=0)
(destination / 'INTAKE.json').write_text(json.dumps(intake, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(status=intake['status'], cases=intake['cases'], files=len(intake['files']), bytes=intake['bytes'])), flush=True)
