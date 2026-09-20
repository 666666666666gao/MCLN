"""Read an existing EG-3DVG acceptance run, without launching or changing it."""
import datetime
import json
import os
from pathlib import Path

import paramiko


def main():
    repo = Path(__file__).resolve().parents[1]
    archive = repo / 'refine-logs/eg3dvg_pretrained_acceptance_20260920_v1'
    archive.mkdir(exist_ok=True)
    root = '/root/autodl-tmp/mcln_eg3dvg_acceptance_20260920_v1'
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.connect('region-9.autodl.pro', port=33476, username='root',
                   password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
    sftp = client.open_sftp()
    names = sftp.listdir(root)
    record = {'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()}
    keep = ['source_patch.json', 'dataset_repair.json', 'import_kernel_receipt.json',
            'checkpoint_download.json', 'checkpoint_inspection.json', 'data_receipt.json',
            'spec.json', 'launch.json', 'controller.exit', 'preflight.exit', 'formal.exit',
            'audit.exit', 'preflight.log', 'formal.log', 'audit.log', 'data.log']
    for name in keep:
        if name not in names:
            continue
        assert sftp.stat(root + '/' + name).st_size < 10_000_000
        sftp.get(root + '/' + name, str(archive / name))
        if name.endswith('.json'):
            record[name] = json.loads((archive / name).read_text())
        elif name.endswith('.exit'):
            record[name] = int((archive / name).read_text())
        else:
            record[name] = (archive / name).read_text().splitlines()[-5:]
    for stage in ['preflight', 'formal']:
        if stage not in names:
            continue
        stage_names = sftp.listdir(root + '/' + stage)
        (archive / stage).mkdir(exist_ok=True)
        for name in ['receipt.json', 'audit.json']:
            if name in stage_names:
                sftp.get(root + '/' + stage + '/' + name, str(archive / stage / name))
                record[stage + '/' + name] = json.loads((archive / stage / name).read_text())
    if 'official_scanrefer.pth' in names:
        record['checkpoint_bytes_now'] = sftp.stat(root + '/official_scanrefer.pth').st_size
    commands = {'gpu': 'nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader',
                'disk': 'df -B1 --output=avail /root/autodl-tmp',
                'processes': 'ps -eo pid,args | grep "[m]cln_eg3dvg_acceptance_20260920_v1"'}
    for key, cmd in commands.items():
        _, out, err = client.exec_command(cmd, timeout=30)
        record[key] = out.read().decode().strip()
        assert out.channel.recv_exit_status() in [0, 1], err.read().decode()
    sftp.close()
    client.close()
    (archive / 'observation_latest.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
