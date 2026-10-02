"""Archive one completed support arm and its raw evidence without a GPU rerun."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import paramiko


parser = argparse.ArgumentParser()
parser.add_argument('--arm', choices=('tail_raw', 'tail_fused'), required=True)
args = parser.parse_args()
local = Path(r'C:\Users\gb\.codex\tmp\pvground_fused_support_20261002')
target = local/('complete_'+args.arm)
archive = Path(r'C:\Users\gb\.codex\archives')/('pvg_'+args.arm+'_20261002')
assert not target.exists() and not archive.exists()
root = '/root/autodl-tmp/pvground_tail_support_20261002'
directory = root+'/'+args.arm
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root',
               password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()


def read(path):
    with sftp.open(path, 'rb') as stream:
        return stream.read()


status_raw = read(root+'/'+args.arm+'_status.json')
status = json.loads(status_raw)
assert status['status'] == 'complete' and status['support_arm'] == args.arm
train_raw = read(directory+'/receipt.json')
train = json.loads(train_raw)
assert train['status'] == 'complete' and train['training_steps'] == 3723
assert train['fit_seen_exactly_once'] and train['fit_batch_row_order_matches_control']
assert train['support_arm'] == args.arm and train['fresh_optimizer']
assert train['fused_support'] == (args.arm == 'tail_fused')
formal_raw = read(directory+'/formal/receipt.json')
formal = json.loads(formal_raw)
assert formal['status'] == 'pass' and formal['rows'] == 9508
assert int(read(directory+'/train.exit')) == int(read(directory+'/formal.exit')) == 0
spec = json.loads(read(directory+'/spec.json'))
imports = json.loads(read(directory+'/imports.json'))
manifest = json.loads(read(spec['input_manifest']))
files = {'status.json': root+'/'+args.arm+'_status.json',
         'run.py': root+'/run.py', 'controller.py': root+'/controller.py',
         'prepare_source.py': root+'/prepare_source.py',
         'source/source_port.json': spec['source_port'],
         'source/input_manifest.json': spec['input_manifest'],
         'source/split_protocol.json': manifest['split_protocol']}
for name in ('spec.json', 'load.json', 'imports.json', 'receipt.json',
             'train.jsonl', 'train.log', 'formal.log', 'train.exit', 'formal.exit',
             'initial_control_comparison.json'):
    files['arm/'+name] = directory+'/'+name
for stage in ('initial', 'terminal', 'formal'):
    for name in ('rows.jsonl', 'receipt.json'):
        files['arm/'+stage+'/'+name] = directory+'/'+stage+'/'+name
for name in ('spec.json', 'preflight.json', 'load.json', 'imports.json',
             'preflight.exit', 'preflight.log'):
    files['preflight/'+name] = root+'/preflight_'+args.arm+'/'+name
for name in ('pvground_candidate_box_refiner.py', 'pvground_tail_support_box_refiner.py',
             'pvground_tail_preflight.py', 'pvground_source_query.py',
             'pvground_observation_query.py', 'pvground_task_observation_query.py',
             'pvground_semantic_assignment.py'):
    files['source/'+name] = directory+'/'+name
for name, path in imports['files'].items():
    files['source/imported/'+name+'.py'] = path
target.mkdir()
records = {}
for name, path in files.items():
    raw = read(path)
    destination = target/name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(raw)
    records[name] = dict(remote=path, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
assert (target/'status.json').read_bytes() == status_raw
assert (target/'arm/receipt.json').read_bytes() == train_raw
assert (target/'arm/formal/receipt.json').read_bytes() == formal_raw
for stage in ('initial', 'terminal', 'formal'):
    receipt = json.loads((target/'arm'/stage/'receipt.json').read_bytes())
    assert records['arm/'+stage+'/rows.jsonl']['sha256'] == receipt['rows_sha256']
assert records['arm/train.jsonl']['sha256'] == train['train_log_sha256']
assert records['run.py']['sha256'] == train['script_sha256']
assert records['arm/spec.json']['sha256'] == train['spec_sha256']
assert records['source/source_port.json']['sha256'] == spec['source_port_sha256']
for name, digest in imports['sha256'].items():
    assert records['source/imported/'+name+'.py']['sha256'] == digest

checkpoint = sftp.stat(directory+'/terminal.pth')
assert shutil.disk_usage(archive.parent).free > checkpoint.st_size+32*1024**2
archive.mkdir()
_, stream, error = client.exec_command('cat -- '+shlex.quote(directory+'/terminal.pth'), timeout=300)
with (archive/'terminal.pth').open('xb') as destination:
    for chunk in iter(lambda: stream.read(1024*1024), b''):
        destination.write(chunk)
assert stream.channel.recv_exit_status() == 0, error.read().decode()
after = sftp.stat(directory+'/terminal.pth')
assert (after.st_size, after.st_mtime) == (checkpoint.st_size, checkpoint.st_mtime)
size = (archive/'terminal.pth').stat().st_size
assert size == checkpoint.st_size
digest = hashlib.sha256()
with (archive/'terminal.pth').open('rb') as stream:
    for chunk in iter(lambda: stream.read(8*1024*1024), b''):
        digest.update(chunk)
assert digest.hexdigest() == train['terminal_sha256']
sftp.close()
client.close()
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(), root=root,
    support_arm=args.arm, status='complete', files=records, GPU_forward_rerun=False,
    checkpoint=dict(remote=directory+'/terminal.pth', bytes=size, sha256=digest.hexdigest(),
        retained_remote=True, archived_local=True, local=str(archive/'terminal.pth'),
        format='state_delta_with_optimizer_and_RNG_requires_retained_author_and_G_parents'))
(target/'INTAKE.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
(archive/'archive_receipt.json').write_text(json.dumps(record['checkpoint'], indent=2)+'\n', encoding='utf-8')
print(json.dumps(dict(status='complete', arm=args.arm, files=len(records),
                     training_steps=3723, formal_bbs=formal['metrics']['bbs'])))
