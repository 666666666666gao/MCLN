"""Collect completed local-arm metadata once, without replay or weight transfer."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex

import paramiko


local = Path(__file__).resolve().parent
output = local / 'local_phase_complete'
assert not output.exists()
launch = json.loads((local / 'launch.json').read_bytes())
spec = json.loads((local / 'local_range_spec.json').read_bytes())
probe = r'''
import hashlib,json,shutil,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1])
assert root==Path('/root/autodl-tmp/pvground_range_head_only_fit_20261004')
directory=root/'local_range'
status=json.loads((root/'status.json').read_bytes())
assert [(r['arm'],r['mode']) for r in status['completed'][:2]]==[
    ('local_range','train'),('local_range','formal')]
names=['spec.json','load.json','imports.json','receipt.json',
    'initial/receipt.json','terminal/receipt.json','formal/receipt.json',
    'train.exit','formal.exit','weight_retention.json']
files={}
for name in names:
    path=directory/name;raw=path.read_bytes()
    files[name]=dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
assert (directory/'train.exit').read_text().strip()==(directory/'formal.exit').read_text().strip()=='0'
fit=json.loads((directory/'receipt.json').read_bytes())
formal=json.loads((directory/'formal/receipt.json').read_bytes())
retention=json.loads((directory/'weight_retention.json').read_bytes())
assert fit['status']=='complete' and fit['training_steps']==3723 and fit['fit_rows']==29778
assert fit['fit_seen_exactly_once'] and fit['head_only'] and fit['original_g_state_unchanged']
assert fit['upstream_running_state_eval'] and fit['head_parameters']==400614
assert fit['script_sha256']==sys.argv[3] and fit['spec_sha256']==sys.argv[4]
assert formal['status']=='pass' and formal['rows']==formal['formal_rows']==9508
assert retention['formal_rows_sha256']==formal['rows_sha256']
assert retention['terminal_sha256']==fit['terminal_sha256']
assert retention['parent_sha256_after']==sys.argv[2]
assert not retention['model_or_optimizer_replayed'] and not retention['local_weight_archive_created']
for mode in ('bbs','bbf'):
    cpu=retention['cpu_box_threshold_recount'][mode]
    assert cpu['cpu_threshold_changes']==0
    assert all(cpu[key]==formal['metrics'][mode][key] for key in ('rec_hits25','rec_hits50'))
rows=directory/'formal/rows.jsonl'
assert hashlib.sha256(rows.read_bytes()).hexdigest()==formal['rows_sha256']
for deleted in retention['deleted']:
    path=Path(deleted['path'])
    assert path==directory/'terminal.pth' and not path.exists()
print(json.dumps(dict(files=files,status=status['status'],stage=status['stage'],
    completed=status['completed'],retained_best=status['retained_best'],
    local_formal_metrics=formal['metrics'],controller_cpu_recount=retention['cpu_box_threshold_recount'],
    local_train_finished_cst=status['completed'][0]['finished_cst'],
    local_formal_finished_cst=status['completed'][1]['finished_cst'],
    formal_rows_sha256=formal['rows_sha256'],
    system_disk_free_bytes=shutil.disk_usage('/').free,
    data_disk_free_bytes=shutil.disk_usage(root).free,
    gpu_snapshot=subprocess.check_output(['nvidia-smi',
        '--query-gpu=index,name,utilization.gpu,memory.used,memory.total',
        '--format=csv,noheader'],universal_newlines=True).strip(),
    gpu_compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory',
        '--format=csv,noheader'],universal_newlines=True).strip(),
    owned_weights=[dict(path=str(p),bytes=p.stat().st_size) for arm in ('local_range','whole_range')
        for p in (root/arm).glob('*.pth')],
    fresh_independent_row_recount=False)))
'''
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root',
    password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
command = shlex.join([spec['runtime'] + '/venv/bin/python', '-c', probe, launch['root'],
    spec['base_terminal_sha256'],
    hashlib.sha256((local / 'run_range_head_only.py').read_bytes()).hexdigest(),
    hashlib.sha256((local / 'local_range_spec.json').read_bytes()).hexdigest()])
_, stdout, stderr = client.exec_command(command, timeout=180)
raw = stdout.read()
assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
record = json.loads(raw)
output.mkdir()
sftp = client.open_sftp()
for relative, item in record['files'].items():
    path = output / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    sftp.get(item['path'], str(path))
    content = path.read_bytes()
    assert len(content) == item['bytes'] and hashlib.sha256(content).hexdigest() == item['sha256']
sftp.close()
client.close()
record.update(time_cst=datetime.datetime.now().astimezone().isoformat(),
    remote_root=launch['root'], collection_model_forwards=0, collection_optimizer_updates=0,
    weights_downloaded=0, weights_deleted=0, full_formal_rows_downloaded=False,
    limitation='Metrics and CPU box recount are completed-controller receipts; fresh full-pair audit remains pending.')
(output / 'INTAKE.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(output=str(output), files=len(record['files']),
    local_formal_metrics=record['local_formal_metrics'],
    local_formal_finished_cst=record['local_formal_finished_cst'],
    retained_best=record['retained_best']['name'],
    weights_downloaded=0, weights_deleted=0)), flush=True)
