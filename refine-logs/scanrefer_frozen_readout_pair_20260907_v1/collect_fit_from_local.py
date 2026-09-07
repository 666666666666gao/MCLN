import datetime
import hashlib
import json
import os
from pathlib import Path

import paramiko

repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
local = repo / 'refine-logs/scanrefer_frozen_readout_pair_20260907_v1'
remote = '/root/autodl-tmp/mcln_scanrefer_frozen_readout_pair_20260907_v1'
client = paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
observed_files = {item.filename: item for item in sftp.listdir_attr(remote)}
assert 'fit_complete.json' in observed_files
downloaded = {}
for name in ['fit_complete.json', 'fit_point_batches.json', 'input_manifest.json', 'protocol.json']:
    with sftp.open(remote + '/' + name, 'rb') as stream:
        stream.prefetch(file_size=observed_files[name].st_size)
        downloaded[name] = stream.read()
assert downloaded['input_manifest.json'] == (local / 'input_manifest.json').read_bytes()
assert hashlib.sha256(downloaded['input_manifest.json']).hexdigest() == 'eccc3b6037ca2e723100791b8907c826de8df69dd25dbf3ed8df59a75468a35b'
assert downloaded['protocol.json'] == (local / 'protocol.json').read_bytes()
fit = json.loads(downloaded['fit_complete.json'])
batches = json.loads(downloaded['fit_point_batches.json'])
protocol = json.loads(downloaded['protocol.json'])
assert fit['steps_per_arm'] == len(batches) == 2482
assert [batch['step'] for batch in batches] == list(range(1, 2483))
assert all(len(batch['row_ids']) == len(batch['point_sha256']) == 12 for batch in batches[:-1])
assert len(batches[-1]['row_ids']) == len(batches[-1]['point_sha256']) == 6
rows = [row for batch in batches for row in batch['row_ids']]
assert len(rows) == len(set(rows)) == 29778
assert sorted(rows) == protocol['row_ids']['fit']
assert not set(rows).intersection(protocol['row_ids']['holdout'])
paths = [remote + '/' + arm + '_frozen_readout_state.pt' for arm in ['native_only', 'frozen_gt']]
_, output, error = client.exec_command('sha256sum -- ' + ' '.join(paths), timeout=60)
hash_output = output.read().decode()
assert output.channel.recv_exit_status() == 0, error.read().decode()
actual_hashes = {line.split(None, 1)[1]: line.split(None, 1)[0] for line in hash_output.splitlines()}
for arm, path in zip(['native_only', 'frozen_gt'], paths):
    checkpoint = fit['checkpoints'][arm]
    assert checkpoint['path'] == path
    assert checkpoint['sha256'] == actual_hashes[path]
    assert checkpoint['bytes'] == observed_files[Path(path).name].st_size
for name in ['fit_complete.json', 'fit_point_batches.json']:
    with (local / name).open('wb') as stream:
        stream.write(downloaded[name])
now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
proof = {'schema': 'mcln-frozen-readout-fit-endpoint-check-v1', 'time_cst': now.isoformat(),
    'steps_per_arm': fit['steps_per_arm'], 'fit_rows': len(rows), 'last_batch_rows': len(batches[-1]['row_ids']),
    'training_seconds_from_fit_record': fit['training_seconds'], 'checkpoints': fit['checkpoints'],
    'fit_rows_exactly_once_and_holdout_disjoint': True, 'endpoint_hashes_independently_verified': True,
    'model_tensor_semantics_independently_checked': False,
    'terminal_receipt_present_at_start': 'receipt.json' in observed_files,
    'controller_exit_present_at_start': 'controller.exit' in observed_files,
    'fit_complete_sha256': hashlib.sha256(downloaded['fit_complete.json']).hexdigest(),
    'fit_batches_sha256': hashlib.sha256(downloaded['fit_point_batches.json']).hexdigest(),
    'limits': 'Fit endpoint and recorded batch coverage check only; not terminal quality or formal promotion.',
    'goal_complete': False}
with (local / 'fit_endpoint_check.json').open('x', encoding='utf-8') as stream:
    json.dump(proof, stream, indent=2, sort_keys=True)
    stream.write('\n')
(local / 'collect_fit_from_local.py').write_bytes(Path(__file__).read_bytes())
sftp.close()
client.close()
print(json.dumps(proof), flush=True)
