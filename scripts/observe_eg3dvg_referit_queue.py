"""Read current EG Nr adaptation and queued Sr transfer without changing either job."""
import datetime
import json
import os
from pathlib import Path
import paramiko

client=paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
sftp=client.open_sftp()
record={'time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()}
for label,root in [('nr','/root/autodl-tmp/mcln_eg3dvg_nr3d_adapt_20260920_v3'),('sr','/root/autodl-tmp/mcln_eg3dvg_sr3d_transfer_20260920_v1'),('sr_adapt','/root/autodl-tmp/mcln_eg3dvg_sr3d_adapt_20260920_v1'),('task_read','/root/autodl-tmp/mcln_eg3dvg_nr3d_task_read_20260920_v1')]:
    names=sftp.listdir(root)
    item={}
    for name in ['preflight.exit','fit.exit','formal.exit','audit.exit','controller.exit','train_inputs.exit','train_dataset_preflight.exit','evaluation_preflight.exit','evaluation_formal.exit','evaluation_audit.exit','paired_rec.exit','candidate_analysis.exit','failure_visual_export.exit','initial_equality.exit']:
        if name in names:
            with sftp.open(root+'/'+name,'rb') as f:item[name]=f.read().decode().strip()
    for name in ['fit.log','preflight.log','formal.log','controller.log','train_inputs.log','train_dataset_preflight.log','evaluation_preflight.log','evaluation_formal.log','evaluation_audit.log','candidate_analysis.log','failure_visual_export.log','initial_equality.log']:
        if name in names:
            with sftp.open(root+'/'+name,'rb') as f:
                f.seek(max(0,sftp.stat(root+'/'+name).st_size-16000))
                item[name]=f.read().decode(errors='replace').splitlines()[-3:]
    for directory in ['fit','formal','train_input_cache']:
        if directory in names:
            for name in ['receipt.json','audit.json']:
                if name in sftp.listdir(root+'/'+directory):
                    with sftp.open(root+'/'+directory+'/'+name,'rb') as f:item[directory+'/'+name]=json.loads(f.read())
    for receipt in ['initial_equality.json','cpu_check.json','launch.json','analysis_launch.json']:
        if receipt in names:
            with sftp.open(root+'/'+receipt,'rb') as f:item[receipt]=json.loads(f.read())
    if 'decision.json' in names:
        with sftp.open(root+'/decision.json','rb') as f:item['decision.json']=json.loads(f.read())
    if 'evaluation' in names and 'formal' in sftp.listdir(root+'/evaluation'):
        for name in ['receipt.json','audit.json','candidate_analysis.json']:
            if name in sftp.listdir(root+'/evaluation/formal'):
                with sftp.open(root+'/evaluation/formal/'+name,'rb') as f:item['evaluation/formal/'+name]=json.loads(f.read())
    record[label]=item
for name,command in [('processes','ps -p 5317,5318,5662,6128,6387,7083,12201,12456 -o pid,etime,args --no-headers'),
                     ('gpu','nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader'),
                     ('disk','df -B1 /root /root/autodl-tmp')]:
    _,stdout,stderr=client.exec_command(command,timeout=30)
    record[name]=stdout.read().decode().strip()
sftp.close();client.close()
output=Path('D:/Program Files/UserCache/gb/codex/tmp/eg3dvg_acceptance_20260920/referit_queue_latest.json')
output.write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(record,indent=2))
