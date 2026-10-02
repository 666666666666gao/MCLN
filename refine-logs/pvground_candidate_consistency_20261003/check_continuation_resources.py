"""Read actual output-directory capacity and GPU/process state; no model work."""
import datetime
import json
import os
from pathlib import Path
import shlex
import paramiko

local = Path(__file__).parent
spec = json.loads((local / 'g_control_spec.json').read_bytes())
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root',
               password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
code = '''
import json, shutil, subprocess, sys
from pathlib import Path
root=Path(sys.argv[1])
paths=json.loads(sys.argv[2])
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.total','--format=csv,noheader'],text=True).strip()
active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True).splitlines()
pair=root/'pair_status.json'
launch=root/'pair_launch.json'
print(json.dumps(dict(output_directory=str(root),directory_free_bytes=shutil.disk_usage(root).free,
    gpu=gpu,compute_processes=active,pair_status=json.loads(pair.read_bytes()) if pair.exists() else None,
    pair_launch_exists=launch.exists(),files=[dict(path=p,exists=Path(p).is_file(),
        bytes=Path(p).stat().st_size if Path(p).is_file() else None) for p in paths])))
'''
proposal = json.loads((local / 'archived_negative_cleanup_proposal.json').read_bytes())
paths = [spec['base_terminal']] + [item['remote'] for item in proposal['files']]
prior_authorized = [
    '/root/autodl-tmp/pvground_p2_semantic_20261002/semantic/terminal.pth',
    '/root/autodl-tmp/cs_pvground_preflight_20261002/preflight_delta.pth',
    '/root/autodl-tmp/mcln_pvground_scanrefer_finetune_20260917_rec_competition_v1/terminal.pth',
    '/root/autodl-tmp/mcln_pvground_initial_replay_20260908_v2/process_a/reference_trace.pt']
paths += prior_authorized
python = spec['runtime'] + '/venv/bin/python'
remote = '/root/autodl-tmp/pvground_candidate_consistency_20261003'
_, stdout, stderr = client.exec_command(shlex.join([python, '-c', code, remote, json.dumps(paths)]), timeout=60)
raw = stdout.read()
assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
record = json.loads(raw)
record.update(time_cst=datetime.datetime.now().astimezone().isoformat(),read_only=True,
              measured_pair_required_bytes=1295019283, new_model_work=False,
              new_cleanup_authorized=False,deletion_executed=False,
              previously_authorized_paths=prior_authorized)
(local / 'continuation_resources_with_authorized_retention.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
client.close()
print(json.dumps(record))
