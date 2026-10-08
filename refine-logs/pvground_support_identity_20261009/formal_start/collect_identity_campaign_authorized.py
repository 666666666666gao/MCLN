"""One scheduled observer of the original identity campaign, no relaunch."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import time
import paramiko


local = Path(__file__).resolve().parent
launch = json.loads((local / 'CAMPAIGN_LAUNCH.json').read_bytes())
spec = json.loads((local / 'campaign_spec.json').read_bytes())
parent = json.loads((local.parent / 'pvground_compressed_geometry_support_20261008/pair_spec.json').read_bytes())
deadline = datetime.datetime.fromisoformat(launch['first_observation_cst'])
assert not (local / 'campaign_actual/INTAKE.json').exists()
remaining = (deadline - datetime.datetime.now().astimezone()).total_seconds()
if remaining > 0:
    time.sleep(remaining)
while True:
    now = datetime.datetime.now().astimezone()
    assert now >= deadline
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
    code = """import datetime,hashlib,json,sys
from pathlib import Path
b=json.load(sys.stdin);r=Path(b['root']);p=r/'campaign.exit'
assert r==Path('/root/autodl-tmp/pvground_support_identity_20261009')
files=[r/'campaign.exit',r/'campaign.log']
if p.exists():files+=sorted(x for x in (r/'campaign').rglob('*') if x.is_file())
print(json.dumps(dict(exitcode=int(p.read_text()) if p.exists() else None,process_live=Path('/proc/'+str(b['pid'])).exists(),exit_marker_mtime_cst=datetime.datetime.fromtimestamp(p.stat().st_mtime,datetime.timezone(datetime.timedelta(hours=8))).isoformat() if p.exists() else None,files=[dict(name=str(x.relative_to(r)),bytes=x.stat().st_size,sha256=hashlib.sha256(x.read_bytes()).hexdigest()) for x in files if x.exists()])))
"""
    stdin, stdout, stderr = client.exec_command(shlex.join([parent['runtime'] + '/venv/bin/python', '-B', '-c', code]), timeout=180)
    stdin.write(json.dumps(dict(root=launch['root'], pid=launch['pid'])))
    stdin.channel.shutdown_write()
    observed = json.loads(stdout.read())
    assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
    observed['observed_cst'] = now.isoformat()
    with (local / 'CAMPAIGN_OBSERVATIONS.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(observed) + '\n')
    if observed['exitcode'] is not None:
        destination = local / 'campaign_actual'
        destination.mkdir()
        sftp = client.open_sftp()
        for item in observed['files']:
            relative = Path(item['name'])
            assert not relative.is_absolute() and '..' not in relative.parts
            path = destination / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            sftp.get(launch['root'] + '/' + item['name'], str(path))
            assert path.stat().st_size == item['bytes']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
        sftp.close()
        client.close()
        (destination / 'INTAKE.json').write_text(json.dumps(observed, indent=2) + '\n', encoding='utf-8')
        if observed['exitcode'] == 0:
            report = json.loads((destination / 'campaign/receipt.json').read_bytes())
            assert report['status'] == 'complete'
            print(json.dumps(report['formal']), flush=True)
        else:
            print((destination / 'campaign.log').read_text(encoding='utf-8')[-6000:], flush=True)
        print(json.dumps(observed), flush=True)
        break
    client.close()
    assert observed['process_live'], 'Original campaign absent without exit marker; inspect authoritative state, do not restart.'
    time.sleep(240)
