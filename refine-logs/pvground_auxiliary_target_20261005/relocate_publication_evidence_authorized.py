"""Move inert published evidence to the data disk, preserving its logical path."""
import json
import os
import shlex
from pathlib import Path

import paramiko


local = Path(__file__).resolve().parent
assert not (local / 'PUBLICATION_STORAGE_RELOCATION.json').exists()
inspection = json.loads((local / 'PUBLICATION_STORAGE_INSPECTION.json').read_bytes())
assert inspection['evidence_device'] == inspection['root_device'] != inspection['data_device']
assert not inspection['evidence_is_symlink']
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root',
               password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
python = '/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python'
probe = '''
import datetime,hashlib,json,os,shutil,sys
from pathlib import Path
project=Path('/home/gb/new butd/butd_detr-main/MCLN-main').resolve()
source=project/'refine-logs'
data=Path('/root/autodl-tmp').resolve()
target=data/'mcln_published_evidence_20261005'
assert source.resolve()==project/'refine-logs' and source.is_dir() and not source.is_symlink()
assert not target.exists() and target.parent==data
assert source.stat().st_dev==Path('/').stat().st_dev!=data.stat().st_dev
def manifest(folder):
    entries={}
    for path in sorted(folder.rglob('*')):
        if path.is_symlink():
            link=os.readlink(path)
            assert os.path.isabs(link),str(path)
            entries[str(path.relative_to(folder))]=dict(symlink=link)
        elif path.is_file():
            raw=path.read_bytes()
            entries[str(path.relative_to(folder))]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    return entries
before=manifest(source)
assert before
content_bytes=sum(entry.get('bytes',0) for entry in before.values())
assert shutil.disk_usage(data).free>content_bytes+284880097
system_before=shutil.disk_usage('/').free
shutil.move(str(source),str(target))
assert manifest(target)==before
assert not source.exists()
source.symlink_to(target,target_is_directory=True)
assert source.is_symlink() and source.resolve()==target
assert manifest(source)==before
record=dict(time_cst=datetime.datetime.now().astimezone().isoformat(),
    status='PUBLISHED_EVIDENCE_RELOCATED_LOGICAL_PATH_PRESERVED',
    logical_path=str(source),data_location=str(target),files=len(before),content_bytes=content_bytes,
    existing_absolute_symlinks_preserved=sum('symlink' in entry for entry in before.values()),
    source_and_destination_file_sizes_and_sha256_exact=True,
    system_free_bytes_before=system_before,system_free_bytes_after=shutil.disk_usage('/').free,
    data_free_bytes_after=shutil.disk_usage(data).free,
    records_deleted=0,weights_deleted=0,training_or_optimizer_replayed=False)
(data/'mcln_published_evidence_relocation_20261005.json').write_text(json.dumps(record,indent=2)+'\\n')
print(json.dumps(record))
'''
_, stdout, stderr = client.exec_command(shlex.join([python, '-B', '-c', probe]), timeout=180)
raw = stdout.read()
assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
record = json.loads(raw)
assert record['source_and_destination_file_sizes_and_sha256_exact']
assert record['records_deleted'] == record['weights_deleted'] == 0
(local / 'PUBLICATION_STORAGE_RELOCATION.json').write_bytes(raw)
client.close()
print(json.dumps(record), flush=True)
