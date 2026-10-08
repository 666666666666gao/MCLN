"""Archive the actual selected trained content head after audit and CPU restore."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import paramiko

local=Path(__file__).resolve().parents[1]
summary=json.loads((local/'analysis/SUMMARY.json').read_bytes())
assert summary['status']=='ACTUAL_CLOSED_FULL_FORMAL_CPU_RECOUNT'
best=summary['metric_best_candidate']
assert best['arm']=='content' and best['stage']=='formal'
assert [best['rec_hits25'],best['rec_hits50']]==[5599,4859]
audit=json.loads((local/'analysis/EXPERIMENT_AUDIT.json').read_bytes())
assert audit['execution_scope']=='ACTUAL_CLOSED_TRAINED_PAIR' and not audit['blocking_findings']
inspection=json.loads((local/'postrun/checkpoint_inspection.json').read_bytes())
assert inspection['status']=='PASS_CLOSED_CPU_BOTH_TERMINALS_STRICT_RECONSTRUCTION'
assert inspection['full_cpu_state_tensors']==1314 and inspection['neural_forwards']==0
weight=next(row for row in inspection['weights'] if row['arm']=='content')
archive=Path('C:/Users/gb/.codex/archives/pvg_compressed_support_best_20261008')
assert not (archive/'terminal.pth').exists()
archive.mkdir(parents=True,exist_ok=True)
client=paramiko.SSHClient();client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
sftp=client.open_sftp();destination=archive/'terminal.pth'
sftp.get(weight['path'],str(destination));sftp.close();client.close()
assert destination.stat().st_size==weight['bytes']
assert hashlib.sha256(destination.read_bytes()).hexdigest()==weight['sha256']
old_archive=Path('C:/Users/gb/.codex/archives/pvg_mask_support_best_20261008/terminal.pth')
spec=json.loads((local/'pair_spec.json').read_bytes())
assert hashlib.sha256(old_archive.read_bytes()).hexdigest()==spec['warm_support_terminal_sha256']
record=dict(status='ACTUAL_SELECTED_CONTENT_ARCHIVED_AFTER_CPU_RESTORE',
    time_cst=datetime.datetime.now().astimezone().isoformat(),hits025=5599,hits050=4859,rows=9508,
    best_checkpoint=weight,best_local_archive=str(destination),archive_bytes_exact=True,archive_sha256_exact=True,
    required_dependencies=inspection['required_dependencies'],full_cpu_state_tensors=1314,
    full_cpu_strict_restore=True,full_gpu_cold_reconstruction=False,
    geometry_encoding='zero',optimizer_updates_this_round=3723,total_support_updates=7446,
    prior_warm_start_local_archive=str(old_archive),prior_warm_start_sha256=spec['warm_support_terminal_sha256'],
    final_cpu_reconstruction_loads_prior_warm_checkpoint=False,nonbest_weights_copied=0,
    experiment_audit_sha256=hashlib.sha256((local/'analysis/EXPERIMENT_AUDIT.json').read_bytes()).hexdigest(),
    spec_sha256=hashlib.sha256((local/'pair_spec.json').read_bytes()).hexdigest(),
    model_factory=str(local/'mask_support_model_factory.py'),
    parent_model_source=spec['model_source'],helper_root=spec['helper_root'],
    neural_forwards=0,optimizer_updates=0,deletions=0,three_effective_contributions_established=False,
    full_goal_complete=False)
(archive/'RESTORATION_MANIFEST.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
(local/'postrun/RETAINED_BEST.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status=record['status'],hits=[5599,4859],archive=str(destination),sha256=weight['sha256'])),flush=True)
