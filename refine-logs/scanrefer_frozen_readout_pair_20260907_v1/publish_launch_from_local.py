import datetime
import hashlib
import json
import os
from pathlib import Path

import paramiko

repo=Path('C:/Users/gb/.codex_mcln_g0_20260905')
local=repo/'refine-logs/scanrefer_frozen_readout_pair_20260907_v1'
remote='/root/autodl-tmp/mcln_scanrefer_frozen_readout_pair_20260907_v1'
runtime='/home/gb/new butd/butd_detr-main/MCLN-main/'
master=repo/'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
desktop=Path('C:/Users/gb/Desktop/document')/master.name
sha=lambda raw:hashlib.sha256(raw).hexdigest()
old=master.read_bytes()
assert sha(old)=='700c810a152b8ed1b8df63e276a63e6158067b1cff01c12b158f659a7bc8d4bd'
assert desktop.read_bytes()==old
manifest_raw=(local/'input_manifest.json').read_bytes()
manifest=json.loads(manifest_raw)
launch=json.loads((local/'launch.json').read_bytes())
prepared=json.loads((local/'preparation.json').read_bytes())
assert launch['manifest_sha256']==prepared['manifest_sha256']==sha(manifest_raw)
assert prepared['checks']['original_environment_cli_and_imports']
assert prepared['checks']['promotion_nonregression_and_mask_checks']
observation_path=sorted(local.glob('observation_*.json'))[-1]
observation=json.loads(observation_path.read_bytes())
assert observation['controller_exit'] is None and observation['queue_controller_exit'] is None
client=paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
sftp=client.open_sftp()
with sftp.open(runtime+'docs/'+master.name,'rb') as stream:
    stream.prefetch(file_size=len(old))
    assert stream.read()==old
for name in ['input_manifest.json','launch.json','preparation.json',observation_path.name]:
    with sftp.open(remote+'/'+name,'rb') as stream:
        stream.prefetch(file_size=(local/name).stat().st_size)
        assert stream.read()==(local/name).read_bytes()
_,output,error=client.exec_command('ps -p 52529,52535 -o pid,ppid,stat,etime,args',timeout=30)
processes=output.read().decode()
assert output.channel.recv_exit_status()==0,error.read().decode()
assert '52529' in processes and '52535' in processes
now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
when=now.strftime('%Y-%m-%d %H:%M CST')
addition='\n\n### 20.113 冻结读出GT配对已锁定并真实启动；按模块REC筛选接续唯一正式终点（'+when+'）\n\n'
addition+='''本轮承接§20.112已通过的实际GPU梯度检查，所有新入口及manifest于运行前提交GitHub `49f2a54e2c3a6f301dc3bfc2565f1f6bedc661d8`。原Python3.7.11环境完成五个入口的导入/CLI检查、全部overlay AST、历史真实9508-row原生REC重算，以及晋级函数的历史线、同次保护线和Mask底线检查；这些只是工程验证，不是本轮精度结果。没有重建环境、增加模块或加载退休range权重。\n\n
| 固定项目 | 本轮约定 |
|---|---|
| 起点 | 同一受保护E71；Parent/Geometry/V99实际权重与归一化元数据始终冻结 |
| 两臂 | native_only；唯一候选frozen_gt |
| 唯一差别 | frozen_gt允许GT读出辅助损失反传核心；native_only在同一读出输入处detach，仍保留完整原生GT损失 |
| 更新范围 | 最后Decoder和最后预测头68个张量；probe已观测其中66个具有梯度，norm1两项无梯度 |
| 优化预算 | 每臂1epoch、2482步、batch12；AdamW，核心LR1e-6、weight_decay0.0005、clip0.1；辅助权重固定1/3 |
| 数据 | 正确mesh根目录，1201训练和312验证superpoint逐文件绑定；原29778fit/6887模块holdout划分；相同采样点，无新增增强 |
| 输出与存储 | 两个固定fit终点，包含核心、冻结读出、optimizer；无中途epoch选优；正式评估不另写权重 |

**真实启动。** 07:52:20 screen52529启动配对训练，07:52:22 screen52535启动终态接续，两者均为后台持久进程。07:53:17实查进程正常，GPU占用3029MiB，日志处于“Begin text decoupling”，尚未出现baseline预测或optimizer更新；不能把启动写成已完成训练。本次同步前再次确认两个screen存活。训练从正确数据完成两臂相同的零更新评估后才进入固定fit，fit结束先保存终点，再完成终态6887行。\n\n
**评估与晋级预先固定。** 模块评估和正式评估都在同一次forward上保存原生REC与完整系统REC，保留逐行采样点身份；Mask使用原完整系统语义路径。终态CPU审计重算REC/Mask、修复/破坏、IoU区间转移，并检查冻结权重/元数据及66份optimizer状态的2482步。6887行仍是主干见过的训练场景，不是整个网络的新场景泛化集。只有frozen_gt的完整系统REC两项同时不低于零更新起点和native_only终点，才执行一次固定9508行正式评估；失败则记录module_rec_screen_not_passed并正常结束，不改选native_only，也不补扫epoch/权重。此规则与上一range协议不同，是本轮运行前新锁定的规则，不修改旧实验判定。\n\n
正式三组固定为protected_v99、native_only_v99、frozen_gt_v99，唯一晋级候选是frozen_gt_v99。历史REC5572/4797及同次保护不退化均保留，ScanMask三项仍须达到58.70/50.70/44.72；59/51为努力方向，不增加为硬门。通过后尽快推进Nr/Sr REC；本轮队列不会在Scan未通过前启动Nr/Sr，Nr/Sr Mask不构成门槛。\n\n
**研究边界。** 此实验检验固定旧读出是否能通过GT监督改善新核心的系统可用性，不声称已经证明接口不兼容是唯一根因。相比失败joint试验，本轮旧读出不会更新；相比range试验，本轮没有局部/范围分支。原候选构造、几何变体及Pareto规则仍保留，不能把本轮描述成取消后处理或新的主干架构。即使某一路指标变化，也必须区分原生能力、完整系统收益与同预算native_only控制。\n\n
'''
addition+='07:53:17实查磁盘剩余'+str(observation['disk_free'])+' bytes（约'+format(observation['disk_free']/1024**3,'.3f')+'GiB）。仅计划写两个固定终点，预计合计约1.2GiB，既有六份保护权重继续保留；本次未再删除其他权重。训练manifest SHA`'+sha(manifest_raw)+'`，接续manifest SHA`'+prepared['queue_manifest_sha256']+'`。本轮尚无新训练终态或正式成绩，整体目标继续active。\n'
new=old+addition.encode()
tracker=repo/'refine-logs/EXPERIMENT_TRACKER.md'
lines=tracker.read_text(encoding='utf-8').splitlines()
lines[2]='Updated: '+when+'. §20.113: fixed frozen-readout GT pair live52529, postqueue52535;correctmesh,E71,no range module;no new quality result.'
for i,line in enumerate(lines):
    if line.startswith('| Frozen protected readout compatibility |'):
        lines[i]='| Frozen protected readout compatibility | Actual GPU probePASS;fixed2482steps/arm pair launched07:52, screens52529/52535 | Frozen Parent/Geometry/V99, candidatefrozen_gt;6887 module REC nonregression versus baseline+native_only gates one9508 formal;no result yet |'
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
proof={'time_cst':now.isoformat(),'section':'20.113','bytes':len(new),'sha256':sha(new),
    'three_master_copies_equal':master.read_bytes()==desktop.read_bytes()==new,
    'training_manifest_sha256':sha(manifest_raw),'published_prelaunch_commit':launch['published_source_commit'],
    'verified_processes':processes,'new_quality_result_available':False,'goal_complete':False}
for name,raw in [('handoff_sync_20_113.json',(json.dumps(proof,indent=2,sort_keys=True)+'\n').encode()),
                 ('publish_launch_from_local.py',Path(__file__).read_bytes()),
                 ('observe_from_local.py',Path('C:/Users/gb/.codex/tmp/observe_mcln_frozen_readout_pair_20260907.py').read_bytes())]:
    (local/name).write_bytes(raw)
    with sftp.open(remote+'/'+name,'wx') as stream:stream.write(raw)
    with sftp.open(remote+'/'+name,'rb') as stream:assert stream.read()==raw
sftp.close()
client.close()
print(json.dumps(proof),flush=True)
