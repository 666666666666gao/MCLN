import datetime
import hashlib
import json
import os
from pathlib import Path

import paramiko

repo=Path('C:/Users/gb/.codex_mcln_g0_20260905')
archive=repo/'refine-logs/scanrefer_native_box_transfer_pair_20260907_v1'
queue=repo/'refine-logs/scanrefer_native_box_transfer_posttraining_20260907_v1'
audit=json.loads((archive/'baseline_cross_run_audit.json').read_bytes())
assert not audit['identity_difference_row_ids']
assert all(not rows for rows in audit['cross_run_difference_row_ids'].values())
observation_path=sorted(queue.glob('observation_*.json'))[-1]
observation=json.loads(observation_path.read_bytes())
label='SCANREFER NATIVE BOX TRANSFER TRAIN '
progress=[json.loads(line[len(label):]) for line in observation['jobs']['training']['run.log'] if line.startswith(label)][-1]
assert progress['step']>=64
assert '58023' in observation['processes'] and '58296' in observation['processes']
zone=datetime.timezone(datetime.timedelta(hours=8))
now=datetime.datetime.now(zone).isoformat()
seconds_per_step=progress['elapsed_seconds']/progress['step']
remaining=seconds_per_step*(2482-progress['step'])
train_eta=datetime.datetime.fromisoformat(observation['time_cst'])+datetime.timedelta(seconds=remaining)
terminal_eta=train_eta+datetime.timedelta(seconds=1839.525)
master=repo/'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
desktop=Path('C:/Users/gb/Desktop/document')/master.name
old=master.read_bytes()
assert hashlib.sha256(old).hexdigest()=='24f586c90627f4b3dcc9d92721105eb7259b9e33675317ecc6d3a9f0b47d2239'
assert desktop.read_bytes()==old
addition='\n\n### 20.125 原生框转移起点逐行一致；固定两臂进入训练（'+now+'）\n\n'
addition+='6887条零更新评估实际耗时1839.52秒，两臂所有行完全相同。CPU复核重新计算完整系统REC6684/6426、原生REC6572/5955，Mask6511/6097、mIoU77.810860787%。相对上一轮冻结读出配对的相同E71起点，6887条样本身份、点SHA、原生Query、原生/系统IoU、Mask IoU和所选几何变体均无一处差异；同一保护artifact、冻结来源、mesh目录和split已核对。这是输入与起点一致性，不是新方法增益或未见场景指标。\n\n'
addition+=f"本轮观察{observation['time_cst']}：原训练Python58023和接续Python58296均实际存活，每臂已记录{progress['step']}/2482更新，训练elapsed{progress['elapsed_seconds']:.2f}秒，平均{seconds_per_step:.3f}秒/两臂同批更新。首步与当前记录梯度有限，周期检查确认冻结语义响应、对比表示及原始Mask未变，固定训练继续。按当前累计吞吐保守估计更新于{train_eta.strftime('%H:%M')}附近完成，再约31分钟终态评估，即{terminal_eta.strftime('%H:%M')}附近进入CPU审计；这是时间估计，不按到点判断完成。\n\n"
addition+=f"磁盘剩余{observation['free_bytes']} bytes（约{observation['free_bytes']/2**30:.3f}GiB）；当前未生成终点或正式结果。保持仅16项回归头小终点设计。接续队列已按240秒实际观察原进程，完整训练后沿原已锁定流程执行审计及条件正式评估；不据训练loss提前选候选、调参或改epoch。Scan正式通过现行REC/Mask底线才接Nr/Sr REC，三数据集整体目标未完成。\n\n"
addition+='起点rows SHA `'+audit['baseline_rows_sha256']+'`；CPU核对结果SHA `'+hashlib.sha256((archive/'baseline_cross_run_audit.json').read_bytes()).hexdigest()+'`。队列及监督对应分析已发布GitHub main `e97926c2ed658bc89200f52ab7decb37d6249578`；训练仍使用原启动manifest和source commit，未热修改。\n'
new=old+addition.encode()
tracker=repo/'refine-logs/EXPERIMENT_TRACKER.md'
lines=tracker.read_text(encoding='utf-8').splitlines()
lines[2]=f"Updated: {now}. Section20.125: baseline6887 exactly matches prior E71/mesh rows;actual trainPID58023 at{progress['step']}/2482 per arm;queue58296 live;no new formal."
for i,line in enumerate(lines):
    if line.startswith('| Native teacher-box transfer |'):
        lines[i]=f"| Native teacher-box transfer | Baseline6887 exact prior parity;trainPID58023 at{progress['step']}/2482 per arm;only16box tensors | Fixed GT-only vs GT+teacher;native6572/5955 and system6684/6426 at start;terminal/conditional formal pending;noNr/Sr |"
tracker_bytes=('\n'.join(lines)+'\n').encode()
client=paramiko.SSHClient();client.load_system_host_keys();client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
sftp=client.open_sftp();runtime='/home/gb/new butd/butd_detr-main/MCLN-main/'
with sftp.open(runtime+'docs/'+master.name,'rb') as stream:
    stream.prefetch(file_size=len(old));assert stream.read()==old
with sftp.open(runtime+'docs/'+master.name,'wb') as stream:
    stream.set_pipelined(True);stream.write(new)
with sftp.open(runtime+'docs/'+master.name,'rb') as stream:
    stream.prefetch(file_size=len(new));assert stream.read()==new
with sftp.open(runtime+'refine-logs/EXPERIMENT_TRACKER.md','wb') as stream:stream.write(tracker_bytes)
master.write_bytes(new);desktop.write_bytes(new);tracker.write_bytes(tracker_bytes)
proof={'section':'20.125','time_cst':now,'master_sha256':hashlib.sha256(new).hexdigest(),'master_bytes':len(new),
       'three_master_copies_equal':True,'observed_steps_per_arm':progress['step'],'seconds_per_step':seconds_per_step,
       'training_eta_cst':train_eta.isoformat(),'terminal_eta_cst':terminal_eta.isoformat(),
       'observation_file':observation_path.name,'new_formal_rows':0,'goal_complete':False}
for name,data in [('baseline_handoff_sync.json',(json.dumps(proof,indent=2)+'\n').encode()),('document_baseline_from_local.py',Path(__file__).read_bytes())]:
    (archive/name).write_bytes(data)
    with sftp.open('/root/autodl-tmp/mcln_scanrefer_native_box_transfer_pair_20260907_v1/'+name,'wx') as stream:stream.write(data)
sftp.close();client.close();print(json.dumps(proof),flush=True)
