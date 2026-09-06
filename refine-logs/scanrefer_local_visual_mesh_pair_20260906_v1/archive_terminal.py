import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

import paramiko

repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
local = repo / 'refine-logs/scanrefer_local_visual_mesh_pair_20260906_v1'
remote = '/root/autodl-tmp/mcln_scanrefer_local_visual_mesh_pair_20260906_v1'
sys.path.insert(0, str(repo))
from scripts.evaluate_scanrefer_local_visual_official import row_metrics
receipt = json.loads((local / 'receipt.json').read_bytes())
audit = json.loads((local / 'independent_audit.json').read_bytes())
assert audit['integrity_pass']
assert audit['receipt_sha256'] == hashlib.sha256((local / 'receipt.json').read_bytes()).hexdigest()
assert (local / 'controller.exit').read_text().strip() == '0'
assert receipt['steps_per_arm'] == 2482 and receipt['formal_rows'] == 0
for arm in ['control', 'local']:
    assert audit['checkpoints'][arm]['optimizer_steps'] == 2482
client = paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro', port=33476, username='root',
               password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
downloaded = {}
for name in ['receipt.json', 'independent_audit.json', 'terminal_rows.json', 'fit_point_batches.json', 'run.log']:
    with sftp.open(remote + '/' + name, 'rb') as stream:
        stream.prefetch(file_size=stream.stat().st_size)
        raw = stream.read()
    destination = local / ('terminal_run.txt' if name == 'run.log' else name)
    # The first download completed before its float-equality check stopped.
    assert destination.read_bytes() == raw, name
    downloaded[name] = hashlib.sha256(raw).hexdigest()
assert downloaded['terminal_rows.json'] == receipt['terminal_rows_sha256']
assert downloaded['fit_point_batches.json'] == receipt['fit_batches_sha256']
baseline = json.loads((local / 'baseline_rows.json').read_bytes())
terminal = json.loads((local / 'terminal_rows.json').read_bytes())
miou_error = {}
for arm in ['control', 'local']:
    assert len(terminal[arm]) == len(baseline[arm]) == 6887
    computed = row_metrics(terminal[arm])
    recorded = receipt['terminal_metrics'][arm]
    for key in ['rows', 'rec_hits025', 'rec_hits050', 'mask_hits025', 'mask_hits050']:
        assert computed[key] == recorded[key]
    miou_error[arm] = abs(computed['mask_miou'] - recorded['mask_miou'])
    assert miou_error[arm] < 1e-8  # Same tolerance as the existing evaluator's mIoU check.
    for before, after in zip(baseline[arm], terminal[arm]):
        for key in ['row_id', 'scan_id', 'physical_space', 'point_sha256']:
            assert before[key] == after[key]
manifest = json.loads((local / 'input_manifest.json').read_bytes())
paths = {name: item['path'] for name, item in manifest['artifacts'].items()}
paths['nr_protected'] = '/root/autodl-tmp/DATA_ROOT/output/network_v99_baseline_gt/nr3d/control/official_rec_monitor/official_best_rec025_epoch_57_0p56652741.pth'
paths['nr_resume_e57'] = '/root/autodl-tmp/DATA_ROOT/output/network_v99_baseline_gt/nr3d/audit/nr3d_mcln_joint_butdcls_v99_relation_cf_conservative_anchor_density_v2_audit_e58_b100_b16x1_w4p2_one_shot/resume_e57.pth'
protected_sizes = {name: sftp.stat(path).st_size for name, path in paths.items()}
_, output, error = client.exec_command("/root/miniconda3/envs/bdetr/bin/python -c 'import json,shutil; print(json.dumps(shutil.disk_usage(\"/root/autodl-tmp\")._asdict()))'", timeout=30)
disk = json.loads(output.read())
assert output.channel.recv_exit_status() == 0, error.read().decode()
proof = {'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
         'status': 'terminal_archived_and_row_metrics_recomputed', 'downloaded_sha256': downloaded,
         'rows_per_arm': 6887, 'optimizer_steps_per_arm': 2482, 'formal_rows': 0,
         'terminal_metrics': receipt['terminal_metrics'], 'point_identity_matches_baseline': True,
         'local_miou_recompute_absolute_error': miou_error,
         'protected_file_sizes': protected_sizes, 'disk': disk, 'weights_downloaded': 0,
         'new_gpu_forwards': 0, 'formal_evaluation_not_duplicated': True}
raw = (json.dumps(proof, indent=2, sort_keys=True) + '\n').encode()
(local / 'terminal_archive_verification.json').write_bytes(raw)
with sftp.open(remote + '/terminal_archive_verification.json', 'wx') as stream:
    stream.write(raw)
archive = Path(__file__).read_bytes()
(local / 'archive_terminal.py').write_bytes(archive)
with sftp.open(remote + '/archive_terminal.py', 'wx') as stream:
    stream.write(archive)
sftp.close()
client.close()
print(json.dumps(proof))
