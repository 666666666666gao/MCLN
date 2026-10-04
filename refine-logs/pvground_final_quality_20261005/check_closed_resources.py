"""Read actual terminal capacity and protected file identities, without replay."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex

import paramiko


local = Path(__file__).resolve().parent
assert json.loads((local/'fit_wait.json').read_bytes())['observer_closed']
destination = local/'CLOSED_RESOURCES.json'
assert not destination.exists()
probe = '''
import hashlib,json,shutil,subprocess
from pathlib import Path
parents={
'/root/autodl-tmp/pvground_boundary_fit_20261004/distribution/terminal.pth':'79e35068b8787a81c354c5fd3ed2bbc12fc62e06167b9cfbe781f86cbbd36c67',
'/root/autodl-tmp/mcln_pvground_scanrefer_finetune_20260918_semantic_assignment_v1/terminal.pth':'0575dfae333dabdf288a470fc09d8967f9867d1853a5f61d9cfc3cbcf7964522',
'/root/autodl-tmp/mcln_pvground_scan_checkpoint_inspection_20260908_v1/PV-Ground_ScanRefer.pth':'6f24c67cc3409f44befdef188f14383497a824d8ab309b8fd572a999ae8bdec3'}
identities={}
for name,digest in parents.items():
    path=Path(name); actual=hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual==digest
    identities[name]=dict(bytes=path.stat().st_size,sha256=actual)
deleted=Path('/root/autodl-tmp/pvground_final_quality_20261005/quality/terminal.pth')
assert not deleted.exists()
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits']).decode().strip()
processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits']).decode().strip()
print(json.dumps(dict(protected_weights=identities,nonbest_weight_absent=str(deleted),data_free_bytes=shutil.disk_usage('/root/autodl-tmp').free,system_free_bytes=shutil.disk_usage('/').free,gpu=gpu,gpu_compute_processes=processes,inference_or_optimizer_replayed=False)))
'''
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
_,stdout,stderr = client.exec_command(shlex.join(['/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python','-c',probe]),timeout=60)
raw = stdout.read()
assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
record = json.loads(raw)
record['time_cst'] = datetime.datetime.now().astimezone().isoformat()
destination.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
client.close()
print(json.dumps(record),flush=True)
