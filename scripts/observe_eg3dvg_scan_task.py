"""Read-only ScanRefer campaign observer; credentials supplied by the authorized wrapper."""
import datetime
import json
import os
from pathlib import Path

import paramiko


def main():
    base = Path('D:/Program Files/UserCache/gb/codex/tmp/eg3dvg_acceptance_20260920')
    remote = '/root/autodl-tmp/mcln_eg3dvg_scan_task_20260921_v1'
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.connect('region-9.autodl.pro', port=33476, username='root',
                   password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
    sftp = client.open_sftp()
    report = {'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()}

    def read(name, tail=False):
        with sftp.open(remote + '/' + name) as f:
            if tail:
                f.seek(max(0, f.stat().st_size - 7000))
            value = f.read().decode('utf-8', errors='replace')
        return value.splitlines()[-8:] if tail else (json.loads(value) if name.endswith('.json') else value)

    available = set(sftp.listdir(remote))
    for name in ['input_receipt.json', 'box_alignment.json', 'launch.json', 'campaign.json',
                 'decision.json', 'epoch_results.json', 'controller.exit', 'stage.exit']:
        if name in available:
            report[name] = read(name)
    for name in sorted(available):
        if name.endswith('.log'):
            report[name] = read(name, tail=True)
    for arm in ['task', 'native']:
        if arm not in available:
            continue
        children = set(sftp.listdir(remote + '/' + arm))
        if 'initial_equality.json' in children:
            report[arm + '/initial_equality.json'] = read(arm + '/initial_equality.json')
        for child in sorted(children):
            if child == 'preflight' or child.startswith('epoch_'):
                files = set(sftp.listdir(remote + '/' + arm + '/' + child))
                for name in ['receipt.json', 'updates.jsonl']:
                    if name in files:
                        key = arm + '/' + child + '/' + name
                        report[key] = read(key, tail=name.endswith('.jsonl'))
            if child.startswith('evaluation_') and 'formal' in sftp.listdir(remote + '/' + arm + '/' + child):
                files = set(sftp.listdir(remote + '/' + arm + '/' + child + '/formal'))
                for name in ['receipt.json', 'audit.json']:
                    if name in files:
                        key = arm + '/' + child + '/formal/' + name
                        report[key] = read(key)
    _, stdout, stderr = client.exec_command('nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader', timeout=30)
    report['gpu'] = stdout.read().decode()
    assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
    sftp.close()
    client.close()
    (base / 'scan_task_latest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    concise = {k:v for k,v in report.items() if k in ['time_cst','gpu','launch.json','controller.exit','decision.json','epoch_results.json']
               or k.endswith('updates.jsonl') or k.endswith('equality.json')}
    for name in ['stage.log','initial_equality.log','task_preflight.log','native_preflight.log','task_epoch_01.log']:
        if name in report:
            concise[name] = report[name][-3:]
    print(json.dumps(concise, indent=2))


if __name__ == '__main__':
    main()
