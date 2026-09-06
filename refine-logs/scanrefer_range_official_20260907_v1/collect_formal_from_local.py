import datetime
import hashlib
import json
import math
import os
from pathlib import Path

import paramiko

repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
local = repo / 'refine-logs/scanrefer_range_official_20260907_v1'
local_queue = repo / 'refine-logs/scanrefer_range_posttraining_20260907_v1'
remote = '/root/autodl-tmp/mcln_scanrefer_range_official_20260907_v1'
queue = '/root/autodl-tmp/mcln_scanrefer_range_posttraining_20260907_v1'
client = paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
raws = {}
for root,names in [(remote,['controller.exit','input_manifest.json','run.log','result/receipt.json','result/independent_audit.json','result/protocol.json','result/rows.json','result/native_rows.json']),
                   (queue,['controller.exit','decision.json','formal_audit.txt'])]:
    for name in names:
        with sftp.open(root+'/'+name,'rb') as stream:
            stream.prefetch(file_size=stream.stat().st_size)
            raws[(root,name)] = stream.read()
assert raws[(remote,'controller.exit')].strip()==raws[(queue,'controller.exit')].strip()==b'0'
digest=lambda value:hashlib.sha256(value).hexdigest()
manifest=json.loads(raws[(remote,'input_manifest.json')])
assert raws[(remote,'input_manifest.json')]==(local/'input_manifest.json').read_bytes()
receipt=json.loads(raws[(remote,'result/receipt.json')])
audit=json.loads(raws[(remote,'result/independent_audit.json')])
decision=json.loads(raws[(queue,'decision.json')])
assert receipt['status']=='complete' and audit['integrity_pass']
assert receipt['formal_rows']==audit['formal_rows']==9508
assert receipt['manifest_sha256']==digest(raws[(remote,'input_manifest.json')])
assert receipt['candidate_predeclared']=='local_v99'
assert receipt['optimizer_steps']==receipt['checkpoint_writes']==0
assert receipt['trained_checkpoints']==manifest['trained_checkpoints']
assert audit['receipt_sha256']==decision['formal_receipt_sha256']==digest(raws[(remote,'result/receipt.json')])
assert decision['formal_audit_sha256']==digest(raws[(remote,'result/independent_audit.json')])
assert audit['audit_script_sha256']==manifest['files']['scripts/audit_scanrefer_range_official.py']
assert audit['promotion']==receipt['promotion']==decision['promotion']
assert audit['protocol_sha256']==digest(raws[(remote,'result/protocol.json')])
for name in ['rows','native_rows']:
    assert receipt[name+'_sha256']==audit[name+'_sha256']==digest(raws[(remote,'result/'+name+'.json')])
records=json.loads(raws[(remote,'result/rows.json')])
native=json.loads(raws[(remote,'result/native_rows.json')])
assert set(records)==set(native)=={'protected_v99','center_v99','local_v99'}
for arm in records:
    rows,nrows=records[arm],native[arm]
    assert [row['row_id'] for row in rows]==[row['row_id'] for row in nrows]==list(range(9508))
    for i,(row,nrow) in enumerate(zip(rows,nrows)):
        assert all(row[k]==nrow[k]==records['protected_v99'][i][k] for k in ['row_id','scan_id','point_sha256'])
        assert all(math.isfinite(value) and 0<=value<=1 for value in [row['rec_iou'],row['mask_iou'],nrow['rec_iou']])
    for threshold,suffix in [(.25,'025'),(.5,'050')]:
        assert sum(row['rec_iou']>threshold for row in rows)==audit['metrics'][arm]['rec_hits'+suffix]
        assert sum(row['mask_iou']>threshold for row in rows)==audit['metrics'][arm]['mask_hits'+suffix]
        assert sum(row['rec_iou']>threshold for row in nrows)==audit['native_rec_metrics'][arm]['rec_hits'+suffix]
    assert abs(sum(row['mask_iou'] for row in rows)*100/9508-audit['metrics'][arm]['mask_miou'])<1e-8
(local/'result').mkdir(exist_ok=False)
for (root,name),raw in raws.items():
    target=local if root==remote else local_queue
    filename='run_complete.txt' if name=='run.log' else name
    (target/filename).write_bytes(raw)
proof={'schema':'mcln-range-formal-local-collection-v1',
    'time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    'formal_rows':9508,'independent_audit_integrity_pass':True,'local_row_metrics_and_identity_verified':True,
    'formal_receipt_sha256':audit['receipt_sha256'],'formal_audit_sha256':decision['formal_audit_sha256'],
    'queue_decision_sha256':digest(raws[(queue,'decision.json')]),
    'metrics':audit['metrics'],'native_rec_metrics':audit['native_rec_metrics'],'promotion':audit['promotion'],
    'formal_evaluation_count':decision['formal_evaluation_count'],'gpu_forwards':0,'weight_files_downloaded':0,'goal_complete':False}
(local/'formal_collection_check.json').write_text(json.dumps(proof,indent=2,sort_keys=True)+'\n',encoding='utf-8')
(local/'collect_formal_from_local.py').write_bytes(Path(__file__).read_bytes())
sftp.close()
client.close()
print(json.dumps(proof),flush=True)
