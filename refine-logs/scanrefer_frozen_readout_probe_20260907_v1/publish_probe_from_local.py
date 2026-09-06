import datetime
import hashlib
import json
import os
from pathlib import Path

import paramiko

repo=Path('C:/Users/gb/.codex_mcln_g0_20260905')
local=repo/'refine-logs/scanrefer_frozen_readout_probe_20260907_v1'
cleanup=repo/'refine-logs/weight_cleanup_20260907_v1'
remote='/root/autodl-tmp/mcln_scanrefer_frozen_readout_probe_20260907_v1'
remote_cleanup='/root/autodl-tmp/mcln_weight_cleanup_20260907_v1'
runtime='/home/gb/new butd/butd_detr-main/MCLN-main/'
master=repo/'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
desktop=Path('C:/Users/gb/Desktop/document')/master.name
digest=lambda raw:hashlib.sha256(raw).hexdigest()
old=master.read_bytes()
assert digest(old)=='66d45ee0f472649e26b21d955ef46e83ed3a7310dae81ce9c74b1a66e2af23c3'
assert desktop.read_bytes()==old
receipt_raw=(local/'receipt.json').read_bytes()
receipt=json.loads(receipt_raw)
manifest_raw=(local/'input_manifest.json').read_bytes()
manifest=json.loads(manifest_raw)
assert receipt['manifest_sha256']==digest(manifest_raw)
assert receipt['status']=='pass' and receipt['real_train_rows']==16 and receipt['formal_rows']==0
assert receipt['backbone_forwards']==6 and receipt['checkpoint_writes']==0
assert receipt['readout_frozen_and_unchanged'] and receipt['frozen_core_and_buffers_unchanged']
assert len(receipt['observations'])==2 and sum(row['rows'] for row in receipt['observations'])==16
assert all(len(row['losses'])==row['optimizer_steps']==2 for row in receipt['disposable_updates'].values())
for name,expected in manifest['files'].items():assert digest((local/name).read_bytes())==expected,name
clean_raw=(cleanup/'receipt.json').read_bytes()
clean=json.loads(clean_raw)
assert clean['status']=='complete' and clean['plan_sha256']==digest((cleanup/'cleanup_plan.json').read_bytes())
assert len(clean['deleted'])==2 and clean['freed_allocated_bytes']==1240719360
client=paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
sftp=client.open_sftp()
with sftp.open(runtime+'docs/'+master.name,'rb') as stream:
    stream.prefetch(file_size=len(old))
    assert stream.read()==old
for root,names in [(remote,['controller.exit','receipt.json','input_manifest.json']),
                   (remote_cleanup,['receipt.json','cleanup_plan.json'])]:
    for name in names:
        with sftp.open(root+'/'+name,'rb') as stream:
            assert stream.read()==((local if root==remote else cleanup)/name).read_bytes()
_,output,error=client.exec_command('ps -p 51683,51686 -o pid,ppid,stat,etime,args',timeout=30)
processes=output.read().decode()
assert output.channel.recv_exit_status()==1 and len(processes.strip().splitlines())==1
now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
when=now.strftime('%Y-%m-%d %H:%M CST')
addition='\n\n### 20.112 冻结旧读出实际梯度检查通过；退休两份失败范围权重（'+when+'）\n\n'
addition+=('依照§20.111从受保护E71重新构造原网络，没有加载失败范围权重或安装范围模块。'
    '原bdetr环境的实际GPU检查于07:07:40启动screen51683，07:14:44完成，controller.exit=0；本次同步前确认screen和Python51686均已正常退出。'
    '共16条固定fit表达、两个批次12+4，原有完整场景表达保留以构造干扰物；1201份正确mesh训练superpoint经SHA核对。'
    '实际执行6次backbone forward和两组各2次一次性optimizer更新，零正式验证行、零新模块、零权重写出。9.572秒仅是数据准备之后的检查耗时，整个进程约7分钟。\n\n'
    '| 训练输入批次 | 原生GT梯度L2 | 冻结读出GT梯度L2 | 两者梯度cosine |\n'
    '|---|---:|---:|---:|\n'
    '| 12条 | 0.822447 | 6.742116 | +0.325447 |\n'
    '| 4条 | 0.996360 | 9.790834 | +0.429686 |\n\n'
    '冻结Parent/Geometry/V99的连续输出确实向当前原生候选参数传递GT监督；与detach输入的forward逐张量一致，detach后的辅助损失没有梯度。'
    '两组一次性更新后，三个读出器参数及归一化元数据均逐项保持不变，非训练核心参数及buffer亦不变；每次辅助标签均由当前候选框与训练root GT重新计算，不绑定旧缓存Query编号。'
    '这只是工程路径PASS，16条小样本不能证明全数据任务冲突、兼容性能或泛化。原生与辅助梯度在这两个批次夹角均为正，也不能预先声称本方法解决了梯度冲突。\n\n'
    '**下一训练对照尚未启动。** 已在本地开始准备独立的native-only与frozen-readout-GT配对入口；同一E71、相同数据及更新步数，三个旧读出始终冻结。'
    '拟沿用原固定1epoch、29778条fit/6887条模块holdout、每臂2482步的短周期预算，核心LR1e-6、辅助权重1/3，更新范围限最后Decoder及预测头。'
    '正式启动前仍需完成该入口、终态独立审计和唯一正式候选的接续约定；不能把当前probe的PASS当作完成训练。'
    '辅助梯度乘1/3后在这两个批次仍约为原生的2.7—3.3倍，此事实应保留解释，当前未因此临时扫描权重。'
    '此方向保留原候选构造、几何变体及Pareto决策，属于固定旧读出的训练适配，不声称取消后处理或产生新架构创新。\n\n'
    '**执行用户要求的无用权重清理。** 范围两臂及全部接续均已结束，完整训练/正式审计和逐行文件已收回并推送。'
    '07:14:14按确切路径退休control_range_visual_state.pt和local_range_visual_state.pt，各620352201 bytes，'
    '删除前核对其SHA、普通文件真实路径、单链接及无运行进程依赖；新probe的artifact列表确认只读取E71及三个旧读出。'
    '实际释放1240719360 allocated bytes（1.156GiB），磁盘余量由8109547520增加至9350262784 bytes（约8.708GiB）。'
    '六份受保护权重删除前后SHA均一致，全部日志、rows/native_rows、协议和审计保留。'
    '§20.109—20.111中的权重审计是删除前事实；两份失败range权重现已不存在，未来不能再声称它们仍可加载或重跑其权重审计。\n\n')
addition+='冻结读出probe receipt SHA`'+digest(receipt_raw)+'`；manifest SHA`'+digest(manifest_raw)+'`；清理receipt SHA`'+digest(clean_raw)+'`。正式最好不变，Nr/Sr仍无新训练成绩，整体目标继续active。\n'
new=old+addition.encode()
tracker=repo/'refine-logs/EXPERIMENT_TRACKER.md'
lines=tracker.read_text(encoding='utf-8').splitlines()
lines[2]='Updated: '+when+'. §20.112: frozen-readout16-row actual gradient+2steps/arm probePASS,not quality;retired2 failed range weights,+1.156GiB;new full pair not launched.'
lines.insert(6,'| Frozen protected readout compatibility | Actual16train rows,6forwards,2disposable steps/arm PASS;readout metadata/core buffers unchanged | Engineering only;no range module,no formal rows,no new weights;fixed paired training still in preparation |')
for i,line in enumerate(lines):
    if line.startswith('| Weight storage cleanup |'):
        lines[i]='| Weight storage cleanup | Prior9.529GiB + current1.156GiB released;8.708GiB free at07:14 | Two failed range endpoints removed after full audit;all logs/rows and six protected hashes retained |'
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
proof={'time_cst':now.isoformat(),'section':'20.112','bytes':len(new),'sha256':digest(new),
    'three_master_copies_equal':master.read_bytes()==desktop.read_bytes()==new,'probe_receipt_sha256':digest(receipt_raw),
    'probe_manifest_sha256':digest(manifest_raw),'cleanup_receipt_sha256':digest(clean_raw),
    'probe_processes':processes,'actual_probe_pass':True,'formal_training_started':False,'goal_complete':False}
for name,raw in [('handoff_sync_20_112.json',(json.dumps(proof,indent=2,sort_keys=True)+'\n').encode()),
                 ('publish_probe_from_local.py',Path(__file__).read_bytes()),
                 ('observe_probe_from_local.py',Path('C:/Users/gb/.codex/tmp/observe_mcln_frozen_readout_probe_20260907.py').read_bytes())]:
    (local/name).write_bytes(raw)
    with sftp.open(remote+'/'+name,'wx') as stream:stream.write(raw)
    with sftp.open(remote+'/'+name,'rb') as stream:assert stream.read()==raw
sftp.close()
client.close()
print(json.dumps(proof),flush=True)
