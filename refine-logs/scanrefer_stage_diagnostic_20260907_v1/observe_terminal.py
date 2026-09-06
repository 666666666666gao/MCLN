import datetime
import hashlib
import json
import os
from pathlib import Path
import runpy
import time

import paramiko

root = Path('C:/Users/gb/.codex_mcln_g0_20260905/refine-logs/scanrefer_stage_diagnostic_20260907_v1')
remote = '/root/autodl-tmp/mcln_scanrefer_stage_diagnostic_20260907_v1'
zone = datetime.timezone(datetime.timedelta(hours=8))
observation = json.loads((root / 'progress_20260907_020126.json').read_bytes())
progress = observation['progress']['SCANREFER STAGE DIAGNOSTIC']
assert observation['screen_live'] and progress['rows'] == 1536 and progress['total'] == 9508
first = datetime.datetime.fromisoformat(observation['time_cst']) + datetime.timedelta(
    seconds=progress['estimated_remaining_seconds'] - 180)
probe = root / 'probe_from_local.py'
schedule = {'time_cst': datetime.datetime.now(zone).isoformat(), 'observer_pid': os.getpid(),
            'screen_pid': 45890, 'first_check_cst': first.isoformat(), 'interval_seconds': 240,
            'basis': 'Conservative observed1536-row remaining-time estimate, first check180 seconds earlier;not first-batch warmup speed.',
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'probe_sha256': hashlib.sha256(probe.read_bytes()).hexdigest()}
with (root / 'terminal_observation_schedule.json').open('x') as stream:
    json.dump(schedule, stream, indent=2, sort_keys=True)
print(json.dumps(schedule), flush=True)
time.sleep(max(0., (first - datetime.datetime.now(zone)).total_seconds()))
while True:
    assert hashlib.sha256(probe.read_bytes()).hexdigest() == schedule['probe_sha256']
    state = runpy.run_path(str(probe))['result']
    if not state['screen_live']:
        break
    time.sleep(240)
assert state['controller_exit'] == 0, state
client = paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro', port=33476, username='root',
               password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
destination = root / 'diagnostic_result'
destination.mkdir()
hashes = {}
for name in ['receipt.json', 'stage_rows.json', 'stage_summary.json', 'normalized_features.json',
             'rows.json', 'native_rows.json', 'protocol.json']:
    with sftp.open(remote + '/diagnostic_result/' + name, 'rb') as stream:
        stream.prefetch(file_size=stream.stat().st_size)
        raw = stream.read()
    (destination / name).write_bytes(raw)
    hashes[name] = hashlib.sha256(raw).hexdigest()
receipt = json.loads((destination / 'receipt.json').read_bytes())
assert receipt['status'] == 'complete' and receipt['formal_rows'] == 0 and receipt['diagnostic_rows'] == 9508
assert not receipt['used_for_promotion']
for name, field in [('stage_rows.json', 'stage_rows_sha256'), ('stage_summary.json', 'stage_summary_sha256'),
                    ('normalized_features.json', 'normalized_features_sha256'), ('rows.json', 'rows_sha256'),
                    ('native_rows.json', 'native_rows_sha256')]:
    assert hashes[name] == receipt[field], name
proof = {'time_cst': datetime.datetime.now(zone).isoformat(), 'downloaded_sha256': hashes,
         'weights_downloaded': 0, 'formal_rows': 0, 'diagnostic_rows': 9508,
         'status': 'completed diagnostic collected;independent stage audit pending'}
raw = (json.dumps(proof, indent=2, sort_keys=True) + '\n').encode()
(root / 'collection_receipt.json').write_bytes(raw)
with sftp.open(remote + '/collection_receipt.json', 'wx') as stream:
    stream.write(raw)
sftp.close()
client.close()
print(json.dumps(proof), flush=True)
