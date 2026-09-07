import hashlib
import json
import os
from pathlib import Path
import shlex

import paramiko

repo=Path('C:/Users/gb/.codex_mcln_g0_20260905')
local=repo/'refine-logs/scanrefer_native_box_transfer_pair_20260907_v1'
remote='/root/autodl-tmp/mcln_scanrefer_native_box_transfer_pair_20260907_v1'
source=(repo/'refine-logs/scanrefer_frozen_readout_pair_20260907_v1/baseline_cross_run_audit.py').read_text()
source=source.replace("root=Path('/root/autodl-tmp/mcln_scanrefer_frozen_readout_pair_20260907_v1')",'root=Path('+repr(remote)+')')
source=source.replace("old=Path('/root/autodl-tmp/mcln_scanrefer_range_pair_20260907_v1')","old=Path('/root/autodl-tmp/mcln_scanrefer_frozen_readout_pair_20260907_v1')")
source=source.replace("rows['native_only']","rows['gt_only']").replace("rows['frozen_gt']","rows['gt_teacher_box']")
source=source.replace("recorded['native_only']","recorded['gt_only']").replace("recorded['frozen_gt']","recorded['gt_teacher_box']")
source=source.replace("read(old/'baseline_rows.json')['control']","read(old/'baseline_rows.json')['native_only']")
source=source.replace("['rec_iou','mask_iou','selected_variant_position']","['rec_iou','native_rec_iou','native_query_index','mask_iou','selected_variant_position']")
source=source.replace('mcln-frozen-readout-baseline-cross-run-audit-v1','mcln-native-box-transfer-baseline-cross-run-audit-v1')
source=source.replace('prior_range_baseline_rows_sha256','prior_frozen_readout_baseline_rows_sha256')
source=source.replace('retired_range_weights_not_read','retired_frozen_readout_weights_not_read')
client=paramiko.SSHClient();client.load_system_host_keys();client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
sftp=client.open_sftp()
files=sftp.listdir(remote)
assert 'baseline_native_metrics.json' in files,'The complete baseline is not available yet'
assert 'baseline_cross_run_audit.json' not in files
for name,data in [('baseline_cross_run_audit.py',source.encode()),('run_baseline_audit_from_local.py',Path(__file__).read_bytes())]:
    (local/name).write_bytes(data)
    with sftp.open(remote+'/'+name,'wx') as stream:stream.write(data)
_,output,error=client.exec_command('CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/bdetr/bin/python '+shlex.quote(remote+'/baseline_cross_run_audit.py'),timeout=30)
payload=output.read();errors=error.read().decode()
assert output.channel.recv_exit_status()==0,errors
result=json.loads(payload)
for name in ['baseline_cross_run_audit.json','baseline_rows.json','baseline_metrics.json','baseline_native_metrics.json']:
    size=sftp.stat(remote+'/'+name).st_size
    with sftp.open(remote+'/'+name,'rb') as stream:
        stream.prefetch(file_size=size);data=stream.read()
    assert len(data)==size
    (local/name).write_bytes(data)
for name,key in [('baseline_rows.json','baseline_rows_sha256'),('baseline_metrics.json','baseline_metrics_sha256'),('baseline_native_metrics.json','baseline_native_metrics_sha256')]:
    assert hashlib.sha256((local/name).read_bytes()).hexdigest()==result[key]
sftp.close();client.close()
print(json.dumps(result,indent=2))
