import csv
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import shlex
import paramiko


CST = timezone(timedelta(hours=8))
REMOTE = '/root/cs_mcln_scanrefer_readback_20260930'
MAIN = Path(r'C:\Users\gb\.codex_mcln_g0_20260905')
NAME = 'cs_mcln_scanrefer_readback_20260930'
DEST = MAIN / 'docs/results' / NAME


def validate_epoch(epoch, payload):
    metrics = payload if epoch == 0 else payload['validation']
    assert metrics['samples'] == 9508
    for suffix in ('025', '050'):
        assert 0 <= metrics['hits' + suffix] <= 9508
        assert abs(metrics['acc' + suffix] - metrics['hits' + suffix] / 9508) < 1e-12
    if epoch:
        assert payload['epoch'] == epoch
        assert payload['steps'] == 4055 and payload['samples'] == 48655
    return metrics


def collect():
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.connect('region-9.autodl.pro', port=33476, username='root',
                   password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
    sftp = client.open_sftp()
    raw_dir = DEST / 'raw'
    raw_dir.mkdir(exist_ok=True)
    sources = []

    def pull(name):
        with sftp.open(REMOTE + '/' + name, 'rb') as stream:
            data = stream.read()
        (raw_dir / name).write_bytes(data)
        sources.append({'file': 'raw/' + name, 'remote_path': REMOTE + '/' + name,
                        'sha256': hashlib.sha256(data).hexdigest()})
        return json.loads(data)

    files = set(sftp.listdir(REMOTE))
    receipt_name = ('monitor_terminal.json' if 'monitor_terminal.json' in files
                    else 'monitor_latest_observation.json')
    receipt = pull(receipt_name)
    state = pull('monitor_state.json')
    through = receipt['completed_epoch']
    rows = []
    if through:
        assert 1 <= through <= 21
        restore_path = DEST / 'restore_witness.json'
        if not restore_path.exists():
            helper = Path(r'C:\Users\gb\.codex\tmp\check_cs_readback_restore_20260930.py')
            remote_helper = REMOTE + '/check_restore.py'
            sftp.put(str(helper), remote_helper)
            command = ('/root/miniconda3/envs/bdetr/bin/python -u ' + shlex.quote(remote_helper))
            _, stdout, stderr = client.exec_command(command, timeout=300)
            output = stdout.read().decode()
            error = stderr.read().decode()
            code = stdout.channel.recv_exit_status()
            witness = {'remote_exit_code': code, 'stdout': output, 'stderr': error,
                       'helper_sha256': hashlib.sha256(helper.read_bytes()).hexdigest(),
                       'checked_at_cst': datetime.now(CST).isoformat()}
            if code == 0:
                witness['result'] = json.loads(output.splitlines()[-1])
            restore_path.write_text(json.dumps(witness, ensure_ascii=False, indent=2), encoding='utf-8')
        for epoch in range(through + 1):
            payload = pull(f'epoch_{epoch}.json')
            metrics = validate_epoch(epoch, payload)
            if epoch == through:
                assert payload == receipt['epoch']
            if epoch == 0:
                initial = metrics
            row = {'epoch': epoch, 'samples': 9508,
                   'hits025': metrics['hits025'], 'hits050': metrics['hits050'],
                   'acc025_percent': metrics['acc025'] * 100,
                   'acc050_percent': metrics['acc050'] * 100,
                   'delta_hits025_vs_own_e0': metrics['hits025'] - initial['hits025'],
                   'delta_hits050_vs_own_e0': metrics['hits050'] - initial['hits050'],
                   'training_seconds_only': payload['seconds'] if epoch else '',
                   'source_sha256': sources[-1]['sha256']}
            for label, folder in (('old_cs', 'cs_mcln_scanrefer_20260923'),
                                  ('native', 'cs_mcln_scanrefer_native_20260926')):
                old = json.loads((MAIN / 'docs/results' / folder / 'raw' / f'epoch_{epoch}.json').read_text(encoding='utf-8'))
                old_metrics = validate_epoch(epoch, old)
                row[f'delta_hits025_vs_{label}_same_epoch'] = metrics['hits025'] - old_metrics['hits025']
                row[f'delta_hits050_vs_{label}_same_epoch'] = metrics['hits050'] - old_metrics['hits050']
            rows.append(row)
        best = max(rows[1:], key=lambda row: (
            min(row['hits025'] / 5572, row['hits050'] / 4797),
            row['hits025'] + row['hits050']))
        assert receipt['best']['epoch'] == best['epoch']
        assert receipt['best']['metrics']['hits025'] == best['hits025']
        assert receipt['best']['metrics']['hits050'] == best['hits050']
        buffer = io.StringIO(newline='')
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        csv_data = buffer.getvalue().encode('utf-8')
        (DEST / 'epoch_metrics.csv').write_bytes(csv_data)
        Path(r'C:\Users\gb\Desktop\document\CS_MCLN_R_ScanRefer_epoch_metrics.csv').write_bytes(csv_data)
        manifest = {
            'collected_at_cst': datetime.now(CST).isoformat(),
            'completed_through_epoch': through, 'planned_epochs': 21,
            'protocol': 'full9508 native last/bbs; batch12; seed2027; E71 fresh init; R plus M1 numerical correction',
            'best_training_epoch': best['epoch'], 'best_metrics': receipt['best']['metrics'],
            'latest_metrics': receipt['epoch']['validation'],
            'csv_sha256': hashlib.sha256(csv_data).hexdigest(), 'sources': sources,
            'attribution_limit': 'Old CS/native comparisons also differ in M1 numerical protocol. Not an isolated R ablation.',
            'estimated_epoch21_cst': receipt['estimated_epoch21_cst'],
            'disk_free_bytes': receipt['disk_free_bytes'],
            'required_extra_atomic_save_bytes': receipt['required_extra_atomic_save_bytes'],
            'terminal': receipt_name == 'monitor_terminal.json',
            'exit_code': receipt['exit_code'],
        }
        (DEST / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    sftp.close()
    client.close()
    for root in (Path(r'C:\Users\gb\.codex_mcln_v99_internal_20260928\docs/results'),
                 Path(r'C:\Users\gb\Desktop\document\results')):
        target = root / NAME
        for source in (raw_dir, DEST / 'manifest.json', DEST / 'epoch_metrics.csv', DEST / 'restore_witness.json'):
            if source.exists():
                if source.is_dir():
                    shutil.copytree(source, target / source.name, dirs_exist_ok=True)
                else:
                    shutil.copyfile(source, target / source.name)
    terminal = receipt_name == 'monitor_terminal.json'
    next_check = None if terminal else datetime.fromisoformat(state['next_check_cst']) + timedelta(seconds=30)
    packet = {'collected_at_cst': datetime.now(CST).isoformat(),
              'completed_through_epoch': through, 'terminal': terminal,
              'next_check_cst': next_check.isoformat() if next_check else None,
              'latest_metrics': receipt['epoch']['validation'] if through else None,
              'exit_code': receipt['exit_code']}
    (DEST / 'local_collection_state.json').write_text(json.dumps(packet, indent=2), encoding='utf-8')
    print(json.dumps(packet, ensure_ascii=False), flush=True)
    return packet


if __name__ == '__main__':
    collect()
