"""Collect the bounded diagnostic artifacts without restarting anything."""
import hashlib
import json
import os
from pathlib import Path
import paramiko

repo = Path(__file__).resolve().parents[1]
root = '/root/autodl-tmp/mcln_pvground_task_terminal_diagnostic_20260917_v2'
archive = repo/'refine-logs/pvground_task_terminal_diagnostic_20260917_v2'
c = paramiko.SSHClient(); c.load_system_host_keys()
c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
s = c.open_sftp()
names = s.listdir(root)
for name in ['controller.pid','controller.exit','run.log','diagnostic.json','boxes_scores.npz']:
    if name in names:
        s.get(root+'/'+name,str(archive/name))
print((archive/'run.log').read_text(encoding='utf-8')[-6500:])
if 'controller.exit' in names:
    print('controller.exit='+ (archive/'controller.exit').read_text().strip())
if 'diagnostic.json' in names:
    record = json.loads((archive/'diagnostic.json').read_bytes())
    assert record['status'] == 'complete' and record['model_forwards'] == 10
    assert hashlib.sha256((archive/'boxes_scores.npz').read_bytes()).hexdigest() == record['boxes_scores_sha256']
    for batch in record['batches']:
        summary = {condition:{key:value['max_abs'] for key,value in changes.items() if key in
            ['last_center','last_pred_size','last_sem_cls_scores','bbs_score','reader_semantic','reader_geometry','sp_last_pred_masks_0']}
            for condition,changes in batch['changes_vs_normal'].items()}
        print(json.dumps(dict(rows=batch['training_row_ids'],selections=batch['selections'],changes=summary)))
s.close(); c.close()
