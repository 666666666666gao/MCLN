import datetime
import hashlib
import json
import os
from pathlib import Path

import paramiko

repo=Path('C:/Users/gb/.codex_mcln_g0_20260905')
local=repo/'refine-logs/scanrefer_range_official_20260907_v1'
local_native=repo/'refine-logs/native_range_preflight_queue_20260907_v1'
remote='/root/autodl-tmp/mcln_scanrefer_range_official_20260907_v1'
native='/root/autodl-tmp/mcln_native_range_preflight_queue_20260907_v1'
runtime='/home/gb/new butd/butd_detr-main/MCLN-main/'
archive='/root/autodl-tmp/mcln_scanrefer_range_formal_check_20260907_v1'
master=repo/'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
desktop=Path('C:/Users/gb/Desktop/document')/master.name
old=master.read_bytes()
digest=lambda raw:hashlib.sha256(raw).hexdigest()
assert digest(old)=='68492e4faba15bbca4eaa67618ad14156f721d1961a21b6e80893a3ea2072ec3'
assert desktop.read_bytes()==old
check_raw=(local/'formal_collection_check.json').read_bytes()
check=json.loads(check_raw)
audit_raw=(local/'result/independent_audit.json').read_bytes()
audit=json.loads(audit_raw)
assert digest(audit_raw)==check['formal_audit_sha256']
assert digest((local/'result/receipt.json').read_bytes())==check['formal_receipt_sha256']
assert audit['integrity_pass'] and not audit['promotion']['advance_to_nr3d_sr3d_rec']
client=paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
sftp=client.open_sftp()
with sftp.open(runtime+'docs/'+master.name,'rb') as stream:
    stream.prefetch(file_size=len(old))
    assert stream.read()==old
for name in ['controller.exit','result/receipt.json','result/independent_audit.json']:
    with sftp.open(remote+'/'+name,'rb') as stream:
        assert stream.read()==(local/name).read_bytes()
native_raws={}
for name in ['controller.exit','decision.json','upstream_observations.jsonl','queue.log']:
    with sftp.open(native+'/'+name,'rb') as stream:
        stream.prefetch(file_size=stream.stat().st_size)
        native_raws[name]=stream.read()
decision=json.loads(native_raws['decision.json'])
assert native_raws['controller.exit'].strip()==b'0'
assert decision['status']=='scanrefer_not_promoted'
assert decision['promotion']==audit['promotion']
assert not decision['native_preflight_started'] and not decision['nr3d_sr3d_training_started']
_,output,error=client.exec_command('ps -p 47112,47242,47245,48128,50454,50455 -o pid,ppid,stat,etime,args',timeout=30)
processes=output.read().decode()
assert output.channel.recv_exit_status()==1,processes
assert len(processes.strip().splitlines())==1
_,output,error=client.exec_command('df -B1 /root/autodl-tmp',timeout=30)
disk=output.read().decode()
assert output.channel.recv_exit_status()==0,error.read().decode()
free_bytes=int(disk.splitlines()[-1].split()[3])
now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
when=now.strftime('%Y-%m-%d %H:%M CST')
addition='\n\n### 20.111 范围读取固定正式9508终态：完整性通过、REC未晋级，条件接续正常关闭（'+when+'）\n\n'
addition+=('本次正式评估于06:50:41完成，耗时2412.15秒；独立审计于06:50:50通过，正式及posttraining controller.exit均为0。'
    '614份固定来源文件、312份正确mesh验证superpoint、四份保护模型/读出及两份训练终点的实际SHA均经审计复核。'
    '本地另收取完整rows/native_rows/protocol/receipt，并重算9508行三组REC/Mask、行身份和采样点SHA，确认与审计一致；没有新GPU推理。\n\n'
    '| 正式9508，同批输入 | 系统REC hits@0.25/@0.50 | 系统REC百分比 | 原生REC hits@0.25/@0.50 | Mask hits@0.25/@0.50 | Mask mIoU |\n'
    '|---|---:|---:|---:|---:|---:|\n')
for arm,label in [('protected_v99','同次受保护V99'),('center_v99','center机制控制'),('local_v99','extent预定候选')]:
    m=audit['metrics'][arm]
    n=audit['native_rec_metrics'][arm]
    addition+='| '+label+' | '+str(m['rec_hits025'])+'/'+str(m['rec_hits050'])+' | '+format(m['rec_hits025']/9508*100,'.6f')+'% / '+format(m['rec_hits050']/9508*100,'.6f')+'% | '+str(n['rec_hits025'])+'/'+str(n['rec_hits050'])+' | '+str(m['mask_hits025'])+'/'+str(m['mask_hits050'])+' | '+format(m['mask_miou'],'.8f')+'% |\n'
addition+=('\n**判定：本版本REC失败封存，历史最好不变。** extent相对同次保护为−25/−54；@0.25修复41、破坏66，@0.50修复84、破坏138。'
    '相对历史V995572/4797为−29/−57。ScanMask三项仍超过原论文58.70/50.70/44.72底线，但REC四项历史/配对检查均未通过。'
    '同次保护5568/4794相对历史少4/3，属于本次控制复核差异；未重算或降低历史保护线，也没有把差异记成方法贡献。\n\n'
    '**机制结论进一步明确。** 原生extent相对保护为−13/+8，而系统为−25/−54；系统相对各自原生的@0.50净收益由保护的+383变为extent的+321，减少62个命中。'
    '中心控制同样出现原生−5/+12、系统−13/−54，说明此现象并非extent分组独有。'
    'extent相对center为原生−8/−4、系统−12/0，未支持此次范围分组采样。系统@0.25 extent−center按141个physical space聚类的95%区间[−0.2185,−0.0301]pp；'
    '这些单seed区间只作诊断，不能代替多seed或逐阶段因果干预。系统仍给两个新核心带来正收益，并非全部失效；现有记录不能独立证明旧读出不兼容是唯一原因，'
    '也不能把IoU降低全归为选错实例。\n\n'
    '**接续已真实结束。** 原生条件队列48128于06:53:01写出scanrefer_not_promoted，native_preflight_started=false、nr3d_sr3d_training_started=false、controller.exit=0。'
    '本次同步前实际检查训练、formal及两个队列PID均已不存在，并与完整退出回执对应；这是正常完成后的不晋级，不是运行阻塞或需要重启。'
    '未更换center为候选、没有补跑epoch或调整正式阈值。\n\n'
    '**下一步收缩为部署兼容训练的独立问题。** 此次center/extent和旧local均不能同时保护系统REC，不在失败range终点继续加边界分布、质量头或新Gate。'
    '回到受保护E71和真实冻结Parent/Geometry/V99，先核验GT读出损失在冻结读出时是否向原生候选参数提供有效梯度，再决定固定预算的native-only与冻结读出GT耦合对照。'
    '它与旧JointRecReadout trial的明确区别是旧试验两个分支都更新读出，本方向固定旧读出解释规则；不新增range分支、不模仿旧错误答案、不绑定缓存Query编号。'
    '这只是下一项待验证工程机制，尚未启动新训练，也不声称取消后处理或形成网络创新。必须先完成实际梯度/候选一致性检查，再锁定训练输入、步数和唯一候选；'
    '通过ScanRefer保护线后仍立即推进Nr/Sr REC，不恢复已取消的长期baseline训练。\n\n'
    '补充训练记录边界：40个日志批次中的extent−center瞬时loss均值−0.00678、median+0.000117，每臂10个新增参数张量实际改变；'
    '这仅排除分支从未更新，不能证明收敛、有效表征或加长训练会提升。完整样本loss均值未记录。\n\n')
addition+='当前磁盘实查剩余'+str(free_bytes)+' bytes（'+format(free_bytes/1024**3,'.3f')+'GiB），本次未删除权重。正式receipt SHA`'+check['formal_receipt_sha256']+'`；独立audit SHA`'+check['formal_audit_sha256']+'`；本地收取复核SHA`'+digest(check_raw)+'`。\n'
new=old+addition.encode()
tracker=repo/'refine-logs/EXPERIMENT_TRACKER.md'
lines=tracker.read_text(encoding='utf-8').splitlines()
lines[2]='Updated: '+when+'. §20.111: fixed Scan range9508 formal+audit complete;extent5543/4740 vs5568/4794 protected;RECFAIL,MaskfloorsPASS;conditional native queue ended without GPU.'
for i,line in enumerate(lines):
    if line.startswith('| ScanRefer matched center/extent range reading |'):
        lines[i]='| ScanRefer matched center/extent range reading | Complete2482updates/arm+6887module+9508formal;integrityPASS,RECFAIL | System extent5543/4740 vs protected5568/4794 and center5555/4740;native5502/4419 vs5515/4411;no promotion or continuation |'
    if line.startswith('| Native range conditional GPU preflight |'):
        lines[i]='| Native range conditional GPU preflight | Queue48128 exited0 at06:53;scanrefer_not_promoted | Native GPU preflight and Nr/Sr training never started;prepared integration not an experiment result |'
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
for name,raw in native_raws.items():
    (local_native/('queue_complete.txt' if name=='queue.log' else name)).write_bytes(raw)
proof={'time_cst':now.isoformat(),'section':'20.111','bytes':len(new),'sha256':digest(new),
    'three_master_copies_equal':master.read_bytes()==desktop.read_bytes()==new,'former_processes':processes,
    'formal_collection_check_sha256':digest(check_raw),'formal_receipt_sha256':check['formal_receipt_sha256'],
    'formal_audit_sha256':check['formal_audit_sha256'],'native_queue_decision_sha256':digest(native_raws['decision.json']),
    'formal_result_obtained':True,'promotion':False,'nr3d_sr3d_training_started':False,'new_training_started':False,
    'free_bytes':free_bytes,'goal_complete':False}
proof_raw=(json.dumps(proof,indent=2,sort_keys=True)+'\n').encode()
sftp.mkdir(archive)
for name,raw in [('formal_collection_check.json',check_raw),('handoff_sync_20_111.json',proof_raw),
                 ('publish_formal_from_local.py',Path(__file__).read_bytes()),
                 ('collect_formal_from_local.py',(local/'collect_formal_from_local.py').read_bytes())]:
    (local/name).write_bytes(raw)
    with sftp.open(archive+'/'+name,'wx') as stream:stream.write(raw)
    with sftp.open(archive+'/'+name,'rb') as stream:assert stream.read()==raw
sftp.close()
client.close()
print(json.dumps(proof),flush=True)
