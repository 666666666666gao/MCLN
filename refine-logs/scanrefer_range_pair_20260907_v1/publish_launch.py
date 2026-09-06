import datetime
import hashlib
import json
import os
from pathlib import Path

import paramiko

repo=Path('C:/Users/gb/.codex_mcln_g0_20260905')
preflight=repo/'refine-logs/scanrefer_range_preflight_20260907_v1'
training=repo/'refine-logs/scanrefer_range_pair_20260907_v1'
queue=repo/'refine-logs/scanrefer_range_posttraining_20260907_v1'
master=repo/'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
desktop=Path('C:/Users/gb/Desktop/document')/master.name
old=master.read_bytes()
assert hashlib.sha256(old).hexdigest()=='8002ec472028ddef159acb11078a15372b3f870784efa90c3075aac587f4e915'
assert desktop.read_bytes()==old
probe=json.loads((preflight/'receipt.json').read_bytes())
launch=json.loads((training/'launch.json').read_bytes())
queue_launch=json.loads((queue/'launch.json').read_bytes())
progress=json.loads((training/'progress_20260907_030312.json').read_bytes())
assert probe['status']=='pass' and probe['formal_rows']==0 and (preflight/'controller.exit').read_text().strip()=='0'
assert progress['screen_live'] and progress['progress']['SCANREFER RANGE EVAL']['stage']=='baseline'
assert progress['progress']['SCANREFER RANGE EVAL']['rows']==12
assert launch['manifest_sha256']==hashlib.sha256((training/'input_manifest.json').read_bytes()).hexdigest()
client=paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
sftp=client.open_sftp()
runtime='/home/gb/new butd/butd_detr-main/MCLN-main/'
with sftp.open(runtime+'docs/'+master.name,'rb') as stream:
    stream.prefetch(file_size=len(old))
    assert stream.read()==old
code="import json,shutil,subprocess; p=subprocess.run(['ps','-p','47112,47242','-o','pid,stat,etime,args'],stdout=subprocess.PIPE,check=True); print(json.dumps({'disk':shutil.disk_usage('/root/autodl-tmp')._asdict(),'processes':p.stdout.decode()}))"
_,out,err=client.exec_command('/root/miniconda3/envs/bdetr/bin/python - <<\'PY\'\n'+code+'\nPY',timeout=30)
raw=out.read()
assert out.channel.recv_exit_status()==0,err.read().decode()
state=json.loads(raw)
assert '47112' in state['processes'] and '47242' in state['processes']
now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
when=now.strftime('%Y-%m-%d %H:%M CST')
state['time_cst']=now.isoformat()
(training/'resource_state_at_handoff.json').write_text(json.dumps(state,indent=2,sort_keys=True)+'\n',encoding='utf-8')
addition='\n\n### 20.104 范围读取实现、真实预检通过，ScanRefer配对任务实际启动（'+when+'）\n\n'
addition+=('**上一轮判定为实质进展，目标仍未完成。** §20.102正式负结果及§20.103逐级诊断维持原结论。本轮新增范围读取对照，'
    '没有替换历史V99或启动Nr/Sr。\n\n'
    '**具体新增结构。** `models/candidate_range_visual.py`在最后一层`cross_v`沿用既有局部残差接口；'
    '点输入仍为128维预训练SA1特征、RGB、相对坐标及轴缩放坐标，映射144维，4头读取后输出288维零初始化残差。'
    '两臂均145008个新增参数，共用区域内点注意力和Query条件的区域注意力。center在共同窗口内取中心最近64点再按八象限分组；'
    'extent在每个象限分别取最多8个不同点，支撑位置为各轴±0.5半尺寸，合计64槽位。共同读取窗口为各轴1.5倍半尺寸，'
    '半尺寸下限沿用0.05m。空槽有明确valid mask，零观测不贡献占位point0，也不复制点补满。'
    '保留全局视觉路径、原生损失和冻结Parent/Geometry/V99，训练限decoder.5及prediction_heads.5和局部读取参数。'
    '这是空间分组读取的最小结构对照，六面边界分布、质量头和BCT损失均未加入；不能提前声称提高精度或形成新颖性结论。\n\n'
    '**原环境CPU3项测试及16条真实训练表达预检通过。** 输入来自既定fit中的16个固定ID；两臂共用相同点SHA，'
    '零初始化下Box、语义、Query投影、两路Mask、融合权重及完整V99运行输出均与原起点完全一致。每臂两次临时更新后，'
    'point/query/key/value读取权重梯度均非零且有限；未训练参数、buffer和旧读出未变。预检权重已随进程结束丢弃，checkpoint_writes=0、formal_rows=0。'
    '实际前向/梯度检查阶段约10.75秒，峰值4825.35MiB；从screen启动到完成约7分钟，主要时间是原生数据加载/文本解析，不能把10.75秒说成包含全部启动耗时。\n\n'
    '| 预检中输入框IoU>0.25的652个Query | center | extent |\n'
    '|---|---:|---:|\n'
    '| 平均有效点数 | 64.0000 | 61.9356 |\n'
    '| 平均有观测分区数（最多8） | 5.4724 | 7.7745 |\n'
    '| 平均归一化空间跨度XYZ | 0.7726/0.7820/0.6592 | 1.5813/1.5610/1.5371 |\n'
    '| 完全空Query数 | 0 | 0 |\n\n'
    '全4096个Query的共同窗口中，两臂各有526个空Query。这是实际观察到的空域，已被valid mask处理。'
    '上述仅证明范围组读取更多区域和更大的空间范围，不是正式REC提升；GT IoU只用于预检统计，未进入采样或模型前向。\n\n'
    '**一次启动观察超时已核实。** 首个预检启动器使用`screen -DmS`并等待其退出，SSH30秒观察超时；'
    '随后确认原screen46874/Python46877继续运行，未重启。原任务于02:48:53正常结束、controller=0，回执与coverage已收集。'
    '后续启动器使用`screen -dmS`，训练和队列均已实际返回启动回执。\n\n'
    '**配对训练任务已启动，当前仍处于零更新基线。** 02:54:11创建screen`47112.mcln_scanrefer_range_pair_v1`，'
    '03:03:12实际观察baseline12/6887条。两臂都从受保护E71及完全相同的零输出区域读取初值开始，'
    '采用既定正确mesh1201train/312val文件、29778fit/6887模块holdout、batch12、每臂2482更新、1epoch；'
    'core LR1e−6、reader LR1e−4、AdamW decay0.0005、clip0.1。holdout仍是预训练主干见过的训练场景。'
    '尚未观察到参数更新或终态指标，不能将任务启动写成训练完成。只保存两个固定终点，不根据holdout选epoch。\n\n'
    '**终态接续已经排队。** screen`47242.mcln_scanrefer_range_posttraining_v1`在03:01:44启动，'
    '训练/审计/正式评估/正式审计4个实际环境入口导入通过。队列首查03:04:11，之后240秒检查原screen47112；'
    '训练完整结束且独立CPU审计通过后，执行一次固定9508条的保护V99、center控制、extent候选三组评估，同时记录原生REC和完整系统REC/Mask。'
    'extent是预先指定的晋级候选，不根据validation临时挑组。晋级仍须历史5572/4797、同次保护不退化、ScanMask58.70/50.70/44.72；'
    '不等待59/51，不恢复Nr/Sr Mask门。通过后记录应立即进行新范围模块Nr/Sr原生预检；队列本身未冒充已经启动Nr/Sr训练。\n\n')
addition+='训练manifest SHA`'+launch['manifest_sha256']+'`；计划SHA`'+launch['plan_sha256']+'`；预检回执SHA`'+hashlib.sha256((preflight/'receipt.json').read_bytes()).hexdigest()+'`。'
addition+='运行目录分别为`/root/autodl-tmp/mcln_scanrefer_range_preflight_20260907_v1`、`mcln_scanrefer_range_pair_20260907_v1`和`mcln_scanrefer_range_posttraining_20260907_v1`。\n\n'
addition+='本次磁盘实查剩余'+str(state['disk']['free'])+' bytes（'+format(state['disk']['free']/1024**3,'.3f')+'GiB）；新预检无权重文件，当前配对尚未写终点。原有受保护权重及需复核的两个旧终点继续保留，之前累计清理约9.529GiB不变。目标保持active。\n'
new=old+addition.encode()
tracker=repo/'refine-logs/EXPERIMENT_TRACKER.md'
lines=tracker.read_text(encoding='utf-8').splitlines()
lines[2]='Updated: '+when+'. §20.104:range reader real preflightPASS;pair47112 observedbaseline12/6887;posttraining47242 queued. Formal protected results unchanged.'
lines.insert(6,'| ScanRefer matched center/extent range reading | CPU3PASS;real16train preflightPASS;pair47112 baseline12/6887 | Same145008params and64slots;2482updates/arm pending;queue47242 audits then fixed9508 protected/center/extent;no new metrics |')
tracker_raw=('\n'.join(lines)+'\n\n'+when+': §20.104 records real range preflight and live paired-task/queue;no new formal improvement claimed.\n').encode()
archives=[(preflight,'prepare_from_local.py','prepare_mcln_range_preflight_20260907.py'),
    (preflight,'launch_fixed_for_future.py','launch_mcln_range_preflight_20260907.py'),
    (preflight,'recover_launch_receipt.py','recover_mcln_range_launch_receipt_20260907.py'),
    (preflight,'probe_from_local.py','probe_mcln_range_preflight_20260907.py'),
    (training,'prepare_launch.py','prepare_launch_mcln_range_pair_20260907.py'),
    (training,'probe_from_local.py','probe_mcln_range_pair_20260907.py'),
    (queue,'launch_from_local.py','launch_mcln_range_queue_20260907.py')]
for folder,name,temporary in archives:
    raw=(Path('C:/Users/gb/.codex/tmp')/temporary).read_bytes()
    (folder/name).write_bytes(raw)
    remote_folder='/root/autodl-tmp/mcln_'+folder.name
    with sftp.open(remote_folder+'/'+name,'wx') as stream: stream.write(raw)
for path in [training/'resource_state_at_handoff.json',training/'progress_20260907_030312.json',training/'progress_20260907_025745.json']:
    with sftp.open('/root/autodl-tmp/mcln_'+training.name+'/'+path.name,'wx') as stream: stream.write(path.read_bytes())
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
proof={'time_cst':now.isoformat(),'section':'20.104','bytes':len(new),'sha256':hashlib.sha256(new).hexdigest(),
    'three_master_copies_equal':master.read_bytes()==desktop.read_bytes()==new,
    'training_screen':47112,'posttraining_screen':47242,'last_observed_stage':'baseline','last_observed_rows':12,
    'training_manifest_sha256':launch['manifest_sha256'],'preflight_complete':True,'new_formal_result':False,'goal_complete':False}
for name,raw in [('handoff_sync_20_104.json',(json.dumps(proof,indent=2,sort_keys=True)+'\n').encode()),
                 ('publish_launch.py',Path(__file__).read_bytes())]:
    (training/name).write_bytes(raw)
    with sftp.open('/root/autodl-tmp/mcln_'+training.name+'/'+name,'wx') as stream: stream.write(raw)
sftp.close()
client.close()
with (repo/'MANIFEST.md').open('ab') as stream:
    stream.write(('| '+when+' | direct continuation | refine-logs/'+training.name+'/launch.json | experiment | Real range preflightPASS;paired47112 baseline12;queue47242;no new formal result |\n').encode())
print(json.dumps(proof))
