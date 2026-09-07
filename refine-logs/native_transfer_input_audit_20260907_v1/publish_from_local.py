import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess

import paramiko

repo=Path('C:/Users/gb/.codex_mcln_g0_20260905')
local=repo/'refine-logs/native_transfer_input_audit_20260907_v1'
remote='/root/autodl-tmp/mcln_native_transfer_input_audit_20260907_v1'
runtime='/home/gb/new butd/butd_detr-main/MCLN-main/'
master=repo/'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
desktop=Path('C:/Users/gb/Desktop/document')/master.name
sha=lambda raw:hashlib.sha256(raw).hexdigest()
old=master.read_bytes()
assert sha(old)=='e904f7ff2762af21417c80f4081f6f0d2cf06be53bd2aa1fc47a73a19f2dac0c'
assert desktop.read_bytes()==old
schema_raw=(local/'receipt.json').read_bytes()
init_raw=(local/'native_initialization_receipt.json').read_bytes()
schema,init=json.loads(schema_raw),json.loads(init_raw)
assert schema['identical_parameter_schema'] and init['status']=='pass'
assert init['upstream_input_audit_sha256']==sha(schema_raw)
assert init['model_factories_executed']==init['native_checkpoint_loads']==2
assert init['gpu_forwards']==init['optimizer_steps']==init['checkpoint_writes']==0
for value in init['protocols'].values():
 assert value['restored_tensors']==1144 and value['all_tensor_values_equal_protected_e71']
 assert value['optimizer_state_entries']==0 and not value['local_or_range_module_installed']
client=paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
sftp=client.open_sftp()
with sftp.open(runtime+'docs/'+master.name,'rb') as stream:
 stream.prefetch(file_size=len(old))
 assert stream.read()==old
for name in ['receipt.json','native_initialization_receipt.json']:
 with sftp.open(remote+'/'+name,'rb') as stream:assert stream.read()==(local/name).read_bytes()
_,output,error=client.exec_command('ps -p 52529,52535 -o pid,ppid,stat,etime,args',timeout=30)
processes=output.read().decode()
assert output.channel.recv_exit_status()==0,error.read().decode()
assert '52529' in processes and '52535' in processes
now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
when=now.strftime('%Y-%m-%d %H:%M CST')
addition='\n\n### 20.114 等待ScanRefer期间完成原生权重接续检查；未占用GPU或启动Nr/Sr训练（'+when+'）\n\n'
addition+='''当前52529训练及52535接续持续正常；08:11:43最近日志已记录baseline1536/6887行、elapsed410.66秒，证明首批之后继续推进，尚无完整baseline指标或参数更新记录。按该阶段吞吐粗估零更新评估约08:32结束，实际以日志为准；不依据日志输出间隔认定卡住或重启。本次同步前再次确认两个原screen存活。\n\n
**实际预训练参数与加载检查。** 08:10:36在CPU读取并重新核对受保护E71和Nr权重SHA，两者均1144份state张量（含96份backbone），canonical名称、shape及dtype逐项一致，没有范围/局部分支。随后08:14:34使用原生618-file准备源码，实际构造Nr3D与Sr3D模型并调用原生model-only checkpoint加载器：两种配置全部1144张量与E71逐值一致，优化器状态为空、start_epoch=1、RoBERTa冻结、butd_cls和既定SourceChoice协议保留。完整E71已覆盖backbone，因此本次模型构造显式不再重复加载独立detector初始化。两项检查均零GPU forward、零优化更新、零新权重写出。\n\n
**接续的明确边界。** 当前配对将保存未带module.前缀的model字典；原生训练器使用带module.前缀的完整模型，并要求独立model-only初始化，不能直接恢复只有本轮68项核心训练范围的optimizer到原生完整优化器。ScanRefer通过后应基于实际frozen_gt终点明确导出原生模型格式，再逐张量核验并执行真实原生输入检查；本次尚未读取不存在的未来终点，也未证明其迁移性能。旧native-range预检绑定extent分支和旧range正式receipt，不能重新启动旧队列来替代本轮接续。E71与Nr权重schema相同不意味着Sr历史最好权重已恢复，也不保证跨数据集指标。\n\n
原生数据协议另外重新核对9份标注/划分文件SHA：Nr3D训练32919语言+11990 detection=44909行、正式7899行；Sr3D训练65846语言+11990 detection=77836行、正式17726行。检测混训、butd_cls对象输入和原有划分保持透明；这些是数据规模，不是本轮训练次数或成绩。当前Scan固定614-file运行源码与原生618-file准备源码的数据集/evaluator文件SHA相同，main_utils/train_dist_mod不同；本次未修改任一冻结运行来源，未来接续仍须绑定所用版本。\n\n
'''
addition+='两个原生配置CPU加载回执SHA`'+sha(init_raw)+'`；权重/标注schema审计SHA`'+sha(schema_raw)+'`。当前正式最好不变，Nr/Sr训练预算尚未因本次CPU检查而启动或锁定；仍由Scan正式保护线通过后推进。08:11:43磁盘剩余9349513216 bytes，约8.708GiB，本次未新增或删除权重。整体目标继续active。\n'
new=old+addition.encode()
tracker=repo/'refine-logs/EXPERIMENT_TRACKER.md'
lines=tracker.read_text(encoding='utf-8').splitlines()
lines[2]='Updated: '+when+'. §20.114: Scanpair52529/post52535 live,baseline1536 logged;actual Nr/Sr nativeCPU model-only loadingPASS1144 tensors;no native training or new weights.'
lines.insert(6,'| Native pretrained transfer input/load audit | E71/Nr schema1144 equal;actual E71 CPUload underNr/Sr configs1144/1144, freshoptimizer | No GPU/updates;future trained endpoint still needs explicit native format export and actual-data check afterScanpromotion |')
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
proof={'time_cst':now.isoformat(),'section':'20.114','bytes':len(new),'sha256':sha(new),
 'three_master_copies_equal':master.read_bytes()==desktop.read_bytes()==new,'verified_live_processes':processes,
 'native_cpu_load_receipt_sha256':sha(init_raw),'input_schema_receipt_sha256':sha(schema_raw),
 'gpu_forwards_in_this_audit':0,'nr3d_sr3d_training_started':False,'goal_complete':False}
for name,raw in [('handoff_sync_20_114.json',(json.dumps(proof,indent=2,sort_keys=True)+'\n').encode()),
                 ('publish_from_local.py',Path(__file__).read_bytes())]:
 (local/name).write_bytes(raw)
 with sftp.open(remote+'/'+name,'wx') as stream:stream.write(raw)
sftp.close()
client.close()
paths=['.gitattributes','docs/'+master.name,'refine-logs/EXPERIMENT_TRACKER.md',str(local.relative_to(repo)),
 'refine-logs/scanrefer_frozen_readout_pair_20260907_v1']
assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=repo).strip()
subprocess.run(['git','add','--']+paths,cwd=repo,check=True)
staged=subprocess.check_output(['git','diff','--cached'],cwd=repo)
assert os.environ['MCLN_SSH_PASSWORD'].encode() not in staged
names=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=repo,text=True).splitlines()
assert not any(Path(name).suffix in ['.pt','.pth'] for name in names)
for name in names:
 if name.startswith('refine-logs/native_transfer_input_audit_'):
  assert subprocess.check_output(['git','show',':'+name],cwd=repo)==(repo/name).read_bytes()
subprocess.run(['git','diff','--cached','--check'],cwd=repo,check=True)
subprocess.run(['git','commit','-m','Verify pretrained native Nr3D and Sr3D initialization while ScanRefer pair runs'],cwd=repo,check=True)
subprocess.run(['git','push','origin','HEAD:main'],cwd=repo,check=True)
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
assert subprocess.check_output(['git','ls-remote','origin','refs/heads/main'],cwd=repo,text=True).split()[0]==head
proof['published_main']=head
print(json.dumps(proof),flush=True)
