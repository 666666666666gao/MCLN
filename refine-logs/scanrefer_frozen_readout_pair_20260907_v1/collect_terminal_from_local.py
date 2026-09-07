import datetime
import hashlib
import json
import os
from pathlib import Path

import paramiko

repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
local = repo / 'refine-logs/scanrefer_frozen_readout_pair_20260907_v1'
local_queue = repo / 'refine-logs/scanrefer_frozen_readout_posttraining_20260907_v1'
remote = '/root/autodl-tmp/mcln_scanrefer_frozen_readout_pair_20260907_v1'
queue = '/root/autodl-tmp/mcln_scanrefer_frozen_readout_posttraining_20260907_v1'
digest = lambda raw: hashlib.sha256(raw).hexdigest()
client = paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
raws = {}
for root, names in [(remote, ['controller.exit', 'receipt.json', 'terminal_rows.json', 'terminal_metrics.json',
                             'terminal_native_metrics.json', 'independent_audit.json', 'run.log']),
                    (queue, ['training_audit.txt', 'training_observations.jsonl'])]:
    for name in names:
        with sftp.open(root + '/' + name, 'rb') as stream:
            stream.prefetch(file_size=stream.stat().st_size)
            raws[(root, name)] = stream.read()
assert raws[(remote, 'controller.exit')].strip() == b'0'
receipt = json.loads(raws[(remote, 'receipt.json')])
audit = json.loads(raws[(remote, 'independent_audit.json')])
manifest = json.loads((local / 'input_manifest.json').read_bytes())
assert receipt['status'] == 'complete' and receipt['steps_per_arm'] == 2482 and receipt['formal_rows'] == 0
assert receipt['manifest_sha256'] == digest((local / 'input_manifest.json').read_bytes())
assert receipt['manifest_sha256'] == 'eccc3b6037ca2e723100791b8907c826de8df69dd25dbf3ed8df59a75468a35b'
assert receipt['terminal_rows_sha256'] == digest(raws[(remote, 'terminal_rows.json')])
assert receipt['baseline_rows_sha256'] == digest((local / 'baseline_rows.json').read_bytes())
assert receipt['fit_batches_sha256'] == digest((local / 'fit_point_batches.json').read_bytes())
assert receipt['checkpoints'] == json.loads((local / 'fit_endpoint_check.json').read_bytes())['checkpoints']
assert audit['schema'] == 'mcln-scanrefer-frozen-readout-pair-independent-audit-v1' and audit['integrity_pass']
assert audit['receipt_sha256'] == digest(raws[(remote, 'receipt.json')])
assert audit['audit_script_sha256'] == manifest['files']['scripts/audit_scanrefer_frozen_readout_pair.py']
assert audit['eligible_for_fixed_terminal_formal_evaluation'] == receipt['eligible_for_fixed_terminal_formal_evaluation']
for arm in ['native_only', 'frozen_gt']:
    checked = audit['checkpoints']['arms'][arm]
    assert checked['checkpoint_sha256'] == receipt['checkpoints'][arm]['sha256']
    assert checked['optimizer_steps'] == 2482 and checked['optimizer_parameter_tensors'] == 66
    assert checked['all_readout_parameters_and_metadata_unchanged'] and checked['frozen_core_parameters_and_buffers_unchanged']
for (root, name), raw in raws.items():
    target = local if root == remote else local_queue
    (target / ('run_terminal_complete.txt' if name == 'run.log' else name)).write_bytes(raw)
proof = {'schema': 'mcln-frozen-readout-terminal-collection-v1',
         'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
         'training_exit': 0, 'training_receipt_sha256': audit['receipt_sha256'],
         'training_audit_sha256': digest(raws[(remote, 'independent_audit.json')]),
         'terminal_rows_sha256': receipt['terminal_rows_sha256'], 'integrity_pass': audit['integrity_pass'],
         'metrics': audit['metrics'], 'native_rec_metrics': audit['native_rec_metrics'],
         'system_effects': {name: value['effects'] for name, value in audit['comparisons'].items()},
         'native_rec_comparisons': audit['native_rec_comparisons'],
         'eligible_for_fixed_terminal_formal_evaluation': audit['eligible_for_fixed_terminal_formal_evaluation'],
         'formal_quality_result_collected': False, 'nr3d_sr3d_training_started': False, 'goal_complete': False}
proof_raw = (json.dumps(proof, indent=2, sort_keys=True) + '\n').encode()
(local / 'terminal_collection_check.json').write_bytes(proof_raw)
(local / 'collect_terminal_from_local.py').write_bytes(Path(__file__).read_bytes())
for name in ['terminal_collection_check.json', 'collect_terminal_from_local.py']:
    raw = (local / name).read_bytes()
    with sftp.open(remote + '/' + name, 'wx') as stream:
        stream.write(raw)
sftp.close()
client.close()
print(json.dumps(proof), flush=True)
