import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess

import paramiko

repo=Path('C:/Users/gb/.codex_mcln_g0_20260905')
local=repo/'refine-logs/scanrefer_frozen_readout_pair_20260907_v1'
remote='/root/autodl-tmp/mcln_scanrefer_frozen_readout_pair_20260907_v1'
runtime='/home/gb/new butd/butd_detr-main/MCLN-main/'
master=repo/'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
desktop=Path('C:/Users/gb/Desktop/document')/master.name
sha=lambda raw:hashlib.sha256(raw).hexdigest()
old=master.read_bytes()
assert sha(old)=='8990d19a3c6d743e63a32e8cc43aee28197b80d35bd116777ee765ee352ffa00'
assert desktop.read_bytes()==old
audit_raw=(local/'baseline_cross_run_audit.json').read_bytes()
audit=json.loads(audit_raw)
assert audit['current_arms_exact_row_parity'] and not audit['identity_difference_row_ids']
assert not any(audit['cross_run_difference_row_ids'].values())
assert audit['baseline_rows_sha256']==sha((local/'baseline_rows.json').read_bytes())
observation_path=sorted(local.glob('observation_*.json'))[-1]
observation=json.loads(observation_path.read_bytes())
progress=observation['progress']['SCANREFER FROZEN READOUT TRAIN']
assert progress['step']>=64 and progress['total']==2482
assert observation['controller_exit'] is None and observation['queue_controller_exit'] is None
client=paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
sftp=client.open_sftp()
with sftp.open(runtime+'docs/'+master.name,'rb') as stream:
 stream.prefetch(file_size=len(old))
 assert stream.read()==old
for name in ['baseline_cross_run_audit.json',observation_path.name]:
 with sftp.open(remote+'/'+name,'rb') as stream:
  stream.prefetch(file_size=(local/name).stat().st_size)
  assert stream.read()==(local/name).read_bytes()
_,output,error=client.exec_command('ps -p 52529,52535 -o pid,ppid,stat,etime,args',timeout=30)
processes=output.read().decode()
assert output.channel.recv_exit_status()==0,error.read().decode()
assert '52529' in processes and '52535' in processes
now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
when=now.strftime('%Y-%m-%d %H:%M CST')
addition='\n\n### 20.115 冻结读出配对完成一致起点并进入实际更新（'+when+'）\n\n'
addition+='''本轮起点评估完整覆盖6887行，耗时1835.916秒；后台08:30:21观察已确认评估完成且两臂完成首个optimizer更新。没有更换配置或重新启动进程。\n\n
| 同一零更新E71，6887条模块留出，两臂逐行完全一致 | REC hits@0.25/@0.50 | Mask hits@0.25/@0.50 | Mask mIoU |
|---|---:|---:|---:|
| 原生REC读出 | 6572/5955 | 此行不混入其他Mask路径 | 此行不混入其他Mask路径 |
| 冻结完整V99读出 | 6684/6426 | 6511/6097 | 77.8108607870% |

上述是训练前且主干见过场景的模块留出记录，不是正式9508结果，更不是本轮训练收益。当前固定旧系统相对原生的起点净差为+112/+471；终态将以相同路径分别比较，不能用其中任一列替换正式历史保护线。\n\n
**零GPU的跨运行起点复核已完成。** 重算当前两臂6887行原生/系统REC与Mask；再与之前同一E71、同一mesh根目录、同一split和同一四份保护artifact的range零更新记录比较。row_id、scan_id、physical_space、point_sha256全部一致，逐行REC IoU、Mask IoU和最终variant位置也没有差异。该审计只读取既有baseline rows，没有读取已退休range权重，没有新推理、调参或增加晋级门。它证明本次起点可与已绑定数据版本的旧起点对应，不把数据修正算成训练增量。\n\n
首个更新中两臂的native loss均10.82071018、readout loss均10.26990128；前向和GT标签一致，梯度范数分别0.75908947与2.63097954，符合是否截断辅助梯度的控制设计。首步loss相同不意味着更新相同，也不能作为训练有效性结论。所有已记录更新均通过有限值检查；当前没有终态参数变化审计或质量成绩。\n\n
'''
addition+='**实际训练继续。** '+observation['time_cst']+'观察，两臂各'+str(progress['step'])+'/2482步，训练elapsed'+format(progress['elapsed_seconds'],'.2f')+'秒，日志估计剩余'+format(progress['estimated_training_remaining_seconds'],'.2f')+'秒；当前GPU为`'+observation['gpu']+'`。该估计从最新已记录批次计算，存在日志间隔延迟，不是完成承诺。原52529和52535仍存活；训练达到固定预算后先保存两个终点，再做完整终态评估与原定独立审计。只有模块REC双阈值相对起点及native_only均不退化，才接续唯一固定9508正式评估。Nr/Sr未启动，目标继续active。\n\n'
addition+='磁盘本次观察剩余'+str(observation['disk_free'])+' bytes（约'+format(observation['disk_free']/1024**3,'.3f')+'GiB），尚未写出训练终点，也没有新删除权重。baseline rows SHA`'+audit['baseline_rows_sha256']+'`；跨运行审计SHA`'+sha(audit_raw)+'`。本次完整起点文件和运行观察已收取并同步，历史正式最好保持不变。\n'
new=old+addition.encode()
tracker=repo/'refine-logs/EXPERIMENT_TRACKER.md'
lines=tracker.read_text(encoding='utf-8').splitlines()
lines[2]='Updated: '+when+'. §20.115: baseline6887 exactparity verified against priorcorrectmesh;actual fixedpair '+str(progress['step'])+'/2482 updates/arm,52529/52535 live;no terminal quality result.'
for i,line in enumerate(lines):
 if line.startswith('| Frozen protected readout compatibility |'):
  lines[i]='| Frozen protected readout compatibility | Actualupdates '+str(progress['step'])+'/2482 each;baseline6887 exactrowparity,system6684/6426,native6572/5955 | Fixedfrozen_gt candidate;modulegate then oneformal;no terminal results,no Nr/Sr training |'
tracker_raw=('\n'.join(lines)+'\n').encode()
for name,raw in [('docs/'+master.name,new),('refine-logs/EXPERIMENT_TRACKER.md',tracker_raw)]:
 with sftp.open(runtime+name,'wb') as stream:
  stream.set_pipelined(True)
  stream.write(raw)
 with sftp.open(runtime+name,'rb') as stream:
  stream.prefetch(file_size=len(raw))
  assert stream.read()==raw
master.write_bytes(new)
desktop.write_bytes(new)
tracker.write_bytes(tracker_raw)
proof={'time_cst':now.isoformat(),'section':'20.115','bytes':len(new),'sha256':sha(new),
 'three_master_copies_equal':master.read_bytes()==desktop.read_bytes()==new,
 'verified_live_processes':processes,'baseline_audit_sha256':sha(audit_raw),
 'actual_steps_per_arm_last_logged':progress['step'],'observation_time_cst':observation['time_cst'],
 'new_terminal_quality_result':False,'goal_complete':False}
for name,raw in [('handoff_sync_20_115.json',(json.dumps(proof,indent=2,sort_keys=True)+'\n').encode()),
                 ('publish_training_start_from_local.py',Path(__file__).read_bytes())]:
 (local/name).write_bytes(raw)
 with sftp.open(remote+'/'+name,'wx') as stream:stream.write(raw)
sftp.close()
client.close()
paths=['docs/'+master.name,'refine-logs/EXPERIMENT_TRACKER.md',str(local.relative_to(repo))]
assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=repo).strip()
subprocess.run(['git','add','--']+paths,cwd=repo,check=True)
staged=subprocess.check_output(['git','diff','--cached'],cwd=repo)
assert os.environ['MCLN_SSH_PASSWORD'].encode() not in staged
names=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=repo,text=True).splitlines()
assert not any(Path(name).suffix in ['.pt','.pth'] for name in names)
for name in names:
 if name.startswith('refine-logs/scanrefer_frozen_readout_pair_'):
  assert subprocess.check_output(['git','show',':'+name],cwd=repo)==(repo/name).read_bytes(),name
subprocess.run(['git','diff','--cached','--check'],cwd=repo,check=True)
subprocess.run(['git','commit','-m','Verify identical ScanRefer baseline and actual frozen-readout paired updates'],cwd=repo,check=True)
subprocess.run(['git','push','origin','HEAD:main'],cwd=repo,check=True)
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
assert subprocess.check_output(['git','ls-remote','origin','refs/heads/main'],cwd=repo,text=True).split()[0]==head
proof['published_main']=head
print(json.dumps(proof),flush=True)
