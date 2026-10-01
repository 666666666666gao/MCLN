"""Collect the scheduled diagnosis after its estimated end; never run a model."""

from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import shlex
import time

import paramiko


CST = timezone(timedelta(hours=8))
MAIN = Path(r'C:\Users\gb\.codex_mcln_g0_20260905')
NAME = 'cs_mcln_readback_paired_diagnostic_20261001'
DIRECTORY = MAIN / 'docs/results' / NAME
REMOTE = '/root/autodl-tmp/cs_mcln_readback_diagnostic_20261001_v1'
OUTPUT = '/root/cs_mcln_scanrefer_readback_20260930/readback_diagnostic'


def collect():
    launch = json.loads((DIRECTORY / 'posttrain_launch.json').read_text())
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.connect('region-9.autodl.pro', port=33476, username='root',
                   password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
    sftp = client.open_sftp()
    for name, expected in launch['diagnostic_files_sha256'].items():
        with sftp.open(REMOTE + '/' + name, 'rb') as stream:
            assert hashlib.sha256(stream.read()).hexdigest() == expected
    files = set(sftp.listdir(REMOTE))
    terminal = 'controller.exit.txt' in files
    if not terminal:
        _, stdout, stderr = client.exec_command('ps -p 220304 -o args=', timeout=30)
        process = stdout.read().decode().strip()
        code = stdout.channel.recv_exit_status()
        assert code == 0 and 'run_cs_mcln_readback_posttrain_diagnostic.py' in process, stderr.read().decode()
        packet = {'collected_at_cst': datetime.now(CST).isoformat(),
                  'terminal': False, 'controller_pid': 220304, 'process': process}
        (DIRECTORY / 'diagnostic_collection_state.json').write_bytes(
            (json.dumps(packet, indent=2) + '\n').encode('utf-8'),
        )
    else:
        raw = DIRECTORY / 'raw'
        raw.mkdir(exist_ok=True)
        sources = []
        for remote_folder, names in (
            (REMOTE, ('controller.log', 'controller.exit.txt')),
            (OUTPUT, ('started.json', 'execution.json', 'sanity_first12.log',
                      'sanity_first12.json', 'full_validation.log', 'full_validation.json')),
        ):
            existing = set(sftp.listdir(remote_folder))
            for name in names:
                if name not in existing:
                    continue
                with sftp.open(remote_folder + '/' + name, 'rb') as stream:
                    data = stream.read()
                (raw / name).write_bytes(data)
                sources.append({'file': 'raw/' + name, 'remote_path': remote_folder + '/' + name,
                                'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
        exit_code = int((raw / 'controller.exit.txt').read_text().strip())
        packet = {'collected_at_cst': datetime.now(CST).isoformat(),
                  'terminal': True, 'exit_code': exit_code, 'sources': sources}
        if exit_code == 0:
            result = json.loads((raw / 'full_validation.json').read_text())
            assert result['sample_count'] == len(result['rows']) == 9508
            assert result['full_validation'] and result['optimizer_steps'] == 0
            assert all(result['formal_result_match'].values())
            assert [row['row_id'] for row in result['rows']] == list(range(9508))
            for threshold, suffix in ((0.25, '025'), (0.5, '050')):
                readback = [row['readback_iou'] > threshold for row in result['rows']]
                bypass = [row['bypass_iou'] > threshold for row in result['rows']]
                summary = result['summary']
                assert sum(readback) == summary['readback_hits' + suffix]
                assert sum(bypass) == summary['bypass_hits' + suffix]
                assert sum(new and not old for new, old in zip(readback, bypass)) == summary['readback_repairs' + suffix]
                assert sum(old and not new for new, old in zip(readback, bypass)) == summary['readback_damages' + suffix]
                assert sum(readback) - sum(bypass) == summary['net_readback_hits' + suffix]
            packet['checkpoint_epoch'] = result['checkpoint_epoch']
            packet['summary'] = result['summary']
        (DIRECTORY / 'diagnostic_terminal_collection.json').write_bytes(
            (json.dumps(packet, indent=2) + '\n').encode('utf-8'),
        )
        for target in (Path(r'C:\Users\gb\.codex_mcln_v99_internal_20260928\docs\results') / NAME,
                       Path(r'C:\Users\gb\Desktop\document\results') / NAME):
            (target / 'raw').mkdir(exist_ok=True)
            for file in raw.iterdir():
                shutil.copyfile(file, target / 'raw' / file.name)
            shutil.copyfile(DIRECTORY / 'diagnostic_terminal_collection.json', target / 'diagnostic_terminal_collection.json')
    sftp.close()
    client.close()
    return packet


if __name__ == '__main__':
    due = datetime(2026, 10, 4, 3, 0, tzinfo=CST)
    print('WAIT_UNTIL', due.isoformat(), flush=True)
    time.sleep(max(0, due.timestamp() - time.time()))
    while True:
        packet = collect()
        print(json.dumps(packet), flush=True)
        if packet['terminal']:
            break
        time.sleep(300)
