"""Audit the fixed mesh ScanRefer result, then conditionally launch native preflight."""

import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import paramiko


ROOT = Path(__file__).parent
REPO = ROOT.parents[1]
ZONE = datetime.timezone(datetime.timedelta(hours=8))
POST = REPO / 'refine-logs/scanrefer_local_visual_mesh_posttraining_20260906_v1'
FORMAL = REPO / 'refine-logs/scanrefer_local_visual_mesh_official_20260906_v1'
REMOTE_FORMAL = '/root/autodl-tmp/mcln_scanrefer_local_visual_mesh_official_20260906_v1'
NATIVE = REPO / 'refine-logs/native_local_preflight_20260906_v2'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def main():
    plan = json.loads((ROOT / 'plan.json').read_bytes())
    for name, digest in plan['files'].items():
        assert sha(ROOT / name) == digest, name
    for name, digest in plan['dependencies'].items():
        assert sha(REPO / name) == digest, name
    assert not (POST / 'executed.json').exists()
    schedule = json.loads((POST / 'observation_schedule.json').read_bytes())
    assert schedule['local_observer_pid'] == plan['existing_observer_pid'] == 50584
    wait_command = [
        'powershell.exe', '-NoProfile', '-NonInteractive', '-Command',
        'Wait-Process -Id 50584 -ErrorAction Stop',
    ]
    write_json(ROOT / 'wait_started.json', {
        'time_cst': datetime.datetime.now(ZONE).isoformat(),
        'local_worker_pid': os.getpid(),
        'waiting_for_existing_observer_pid': 50584,
        'wait_command': wait_command,
        'plan_sha256': sha(ROOT / 'plan.json'),
        'status': 'waiting for existing formal-launch collector;no new GPU work yet',
    })
    print('Waiting for existing observer 50584.', flush=True)
    subprocess.check_call(wait_command, creationflags=subprocess.CREATE_NO_WINDOW)
    launch = json.loads((POST / 'executed.json').read_bytes())
    assert launch == json.loads((FORMAL / 'launch.json').read_bytes())
    assert sha(FORMAL / 'input_manifest.json') == launch['manifest_sha256']
    assert launch['formal_rows_planned'] == 9508
    session = launch['screen_session']
    assert len(session) == 1 and '.mcln_scanrefer_mesh_official_v1' in session[0]
    screen_pid = int(session[0].split('.', 1)[0])
    first = datetime.datetime.fromisoformat(launch['time_cst']) + datetime.timedelta(seconds=1440)
    write_json(ROOT / 'formal_observation_schedule.json', {
        'time_cst': datetime.datetime.now(ZONE).isoformat(),
        'formal_screen_pid': screen_pid,
        'formal_manifest_sha256': launch['manifest_sha256'],
        'first_check_cst': first.isoformat(),
        'interval_seconds': 240,
        'eta_basis': 'Previous same-source paired9508 GPU evaluation took1537.14 seconds plus startup;first check at24 minutes.',
    })
    print('Formal first check: ' + first.isoformat(), flush=True)
    delay = (first - datetime.datetime.now(ZONE)).total_seconds()
    if delay > 0:
        time.sleep(delay)
    while True:
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect('region-9.autodl.pro', port=33476, username='root',
                       password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
        sftp = client.open_sftp()
        _, output, error = client.exec_command(
            'ps -p ' + str(screen_pid) + ' -o pid,stat,etime,args', timeout=30)
        process = output.read().decode('utf-8')
        status = output.channel.recv_exit_status()
        assert status in (0, 1), error.read().decode('utf-8')
        live = status == 0 and 'mcln_scanrefer_mesh_official_v1' in process
        with sftp.open(REMOTE_FORMAL + '/run.log', 'rb') as stream:
            stream.seek(max(0, stream.stat().st_size - 64000))
            tail = stream.read().decode('utf-8')
        progress = {}
        for line in tail.splitlines():
            for label in ['SCANREFER LOCAL VISUAL OFFICIAL ',
                          'SCANREFER LOCAL VISUAL OFFICIAL COMPLETE ']:
                if line.startswith(label + '{'):
                    progress[label.strip()] = json.loads(line[len(label):])
        now = datetime.datetime.now(ZONE)
        observation = {
            'time_cst': now.isoformat(), 'formal_screen_live': live,
            'process': process, 'progress': progress,
        }
        stem = 'formal_progress_' + now.strftime('%Y%m%d_%H%M%S')
        write_json(ROOT / (stem + '.json'), observation)
        (ROOT / (stem + '.txt')).write_text(tail, encoding='utf-8')
        print(json.dumps(observation), flush=True)
        if not live:
            with sftp.open(REMOTE_FORMAL + '/controller.exit', 'rb') as stream:
                assert stream.read().strip() == b'0'
        sftp.close()
        client.close()
        if not live:
            break
        time.sleep(240)
    with (ROOT / 'formal_audit_controller.txt').open('xb') as output:
        subprocess.check_call([sys.executable, str(ROOT / 'audit_terminal.py')],
                              stdout=output, stderr=subprocess.STDOUT,
                              creationflags=subprocess.CREATE_NO_WINDOW)
    audit_path = FORMAL / 'result/independent_audit.json'
    audit = json.loads(audit_path.read_bytes())
    assert audit['integrity_pass'] and audit['formal_rows'] == 9508
    assert audit['receipt_sha256'] == sha(FORMAL / 'result/receipt.json')
    decision = {
        'time_cst': datetime.datetime.now(ZONE).isoformat(),
        'formal_directory': REMOTE_FORMAL,
        'formal_receipt_sha256': audit['receipt_sha256'],
        'formal_audit_sha256': sha(audit_path),
        'promotion': audit['promotion'], 'metrics': audit['metrics'],
        'native_gpu_preflight_launched': False,
        'nr3d_sr3d_training_started': False,
        'checkpoint_writes_by_this_queue': 0,
    }
    if audit['promotion']['advance_to_nr3d_sr3d_rec']:
        with (ROOT / 'native_launch_controller.txt').open('xb') as output:
            subprocess.check_call([sys.executable, str(ROOT / 'launch_native_conditional.py')],
                                  stdout=output, stderr=subprocess.STDOUT,
                                  creationflags=subprocess.CREATE_NO_WINDOW)
        decision['native_gpu_preflight_launched'] = True
        decision['native_launch_sha256'] = sha(NATIVE / 'launch.json')
        decision['status'] = 'Scan passed;disposable native GPU preflight launched;not native training or metrics'
    else:
        decision['status'] = 'Scan formal did not qualify;native GPU preflight not launched'
    write_json(ROOT / 'decision.json', decision)
    print(json.dumps(decision), flush=True)


if __name__ == '__main__':
    main()
