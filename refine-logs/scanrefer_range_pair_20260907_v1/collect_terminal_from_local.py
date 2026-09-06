import datetime
import hashlib
import json
import os
from pathlib import Path

import paramiko

repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
local_train = repo / 'refine-logs/scanrefer_range_pair_20260907_v1'
local_queue = repo / 'refine-logs/scanrefer_range_posttraining_20260907_v1'
local_formal = repo / 'refine-logs/scanrefer_range_official_20260907_v1'
training = '/root/autodl-tmp/mcln_scanrefer_range_pair_20260907_v1'
queue = '/root/autodl-tmp/mcln_scanrefer_range_posttraining_20260907_v1'
formal = '/root/autodl-tmp/mcln_scanrefer_range_official_20260907_v1'
client = paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
raws = {}
for root, names in [(training, ['controller.exit', 'receipt.json', 'terminal_rows.json', 'terminal_metrics.json', 'independent_audit.json', 'run.log']),
                    (queue, ['training_audit.txt', 'training_observations.jsonl']),
                    (formal, ['input_manifest.json', 'launch.json'])]:
    for name in names:
        with sftp.open(root + '/' + name, 'rb') as stream:
            stream.prefetch(file_size=stream.stat().st_size)
            raws[(root, name)] = stream.read()
assert raws[(training, 'controller.exit')].strip() == b'0'
receipt = json.loads(raws[(training, 'receipt.json')])
audit = json.loads(raws[(training, 'independent_audit.json')])
manifest = json.loads(raws[(formal, 'input_manifest.json')])
launch = json.loads(raws[(formal, 'launch.json')])
digest = lambda raw: hashlib.sha256(raw).hexdigest()
assert receipt['status'] == 'complete' and receipt['steps_per_arm'] == 2482 and receipt['formal_rows'] == 0
assert receipt['manifest_sha256'] == digest((local_train / 'input_manifest.json').read_bytes())
assert receipt['terminal_rows_sha256'] == digest(raws[(training, 'terminal_rows.json')])
assert receipt['baseline_rows_sha256'] == digest((local_train / 'baseline_rows.json').read_bytes())
assert receipt['fit_batches_sha256'] == digest((local_train / 'fit_point_batches.json').read_bytes())
assert audit['schema'] == 'mcln-scanrefer-range-independent-audit-v1' and audit['integrity_pass']
assert audit['receipt_sha256'] == digest(raws[(training, 'receipt.json')])
assert audit['audit_script_sha256'] == json.loads((local_train / 'input_manifest.json').read_bytes())['files']['scripts/audit_scanrefer_range_pair.py']
assert manifest['training_receipt_sha256'] == audit['receipt_sha256']
assert manifest['training_audit_sha256'] == digest(raws[(training, 'independent_audit.json')])
assert manifest['trained_checkpoints'] == receipt['checkpoints']
assert manifest['candidate_predeclared'] == 'local_v99' and manifest['formal_rows'] == 9508
assert launch['manifest_sha256'] == digest(raws[(formal, 'input_manifest.json')])
pid = int(launch['process_pid'])
_, output, error = client.exec_command('ps -p ' + str(pid) + ' -o pid,stat,etime,args', timeout=30)
process = output.read().decode()
assert output.channel.recv_exit_status() == 0, error.read().decode()
assert 'evaluate_scanrefer_range_official.py' in process
local_formal.mkdir(exist_ok=False)
for (root, name), raw in raws.items():
    target = {training: local_train, queue: local_queue, formal: local_formal}[root]
    filename = 'run_terminal_complete.txt' if name == 'run.log' else name
    (target / filename).write_bytes(raw)
proof = {'schema': 'mcln-range-terminal-collection-v1',
    'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    'training_exit': 0, 'training_receipt_sha256': audit['receipt_sha256'],
    'training_independent_audit_sha256': manifest['training_audit_sha256'],
    'formal_manifest_sha256': launch['manifest_sha256'], 'formal_process_pid': pid, 'formal_process': process,
    'integrity_pass': audit['integrity_pass'], 'module_metrics': audit['metrics'],
    'comparisons': audit['comparisons'], 'module_nonregression': audit['development_dual_rec_nonregression'],
    'formal_results_obtained': False, 'nr3d_sr3d_training_started': False, 'goal_complete': False}
(local_train / 'terminal_collection_check.json').write_text(json.dumps(proof, indent=2, sort_keys=True) + '\n', encoding='utf-8')
(local_train / 'collect_terminal_from_local.py').write_bytes(Path(__file__).read_bytes())
sftp.close()
client.close()
print(json.dumps({key: proof[key] for key in ['time_cst', 'training_exit', 'integrity_pass', 'module_metrics', 'module_nonregression',
    'training_receipt_sha256', 'training_independent_audit_sha256', 'formal_process_pid', 'formal_process', 'formal_manifest_sha256']}), flush=True)
