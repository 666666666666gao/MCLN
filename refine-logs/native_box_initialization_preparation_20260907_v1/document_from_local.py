import datetime
import hashlib
import json
import os
from pathlib import Path

import paramiko

repo=Path('C:/Users/gb/.codex_mcln_g0_20260905')
archive=repo/'refine-logs/native_box_initialization_preparation_20260907_v1'
receipt=json.loads((archive/'receipt.json').read_bytes())
assert receipt['status']=='pass' and receipt['synthetic_fixture_only'] and receipt['temporary_fixture_deleted']
master=repo/'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
desktop=Path('C:/Users/gb/Desktop/document')/master.name
old=master.read_bytes();assert desktop.read_bytes()==old
assert hashlib.sha256(old).hexdigest()=='0ac81db224725faf3293eb910e9982228d7d69c054727fa26538c0ebe9f464b4'
now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
addition='\n\n### 20.126 小框头终点的原生初始化接续准备（'+now+'）\n\n'
addition+='实际接口差异已由源码确认：当前终点只有16项框头参数，原生加载器需要带module.前缀的完整model。新增独立导出工具，在正式Scan通过后使用E71补全1144项state，仅覆盖16项回归参数；头内12项BatchNorm缓冲继续保持E71。复用现有model_only_initialization，不修改主训练加载器、不继承旧optimizer/scheduler。工具重新核对原正式行记录、晋级决定、来源和权重SHA，再写唯一初始化文件；不是现在运行新数据集训练。\n\n'
addition+='原Python3.7/Torch1.10.2环境8项单元测试通过。使用真实E71和明确标注的合成参数差值，两个原生数据集模型工厂及加载器均实际执行：Nr/Sr各1144项逐值匹配重建模型，16项框参数正确覆盖，其余1128项保留；优化器状态0、调度器新建状态不变、start_epoch1。源618文件核对一致。该检查GPU前向0、真实训练步数0、正式行0，不是对未来训练终点的效果验证。\n\n'
addition+='CPU检查先遇到暂存scripts包遮蔽，再遇到chdir后相对__file__目录失效；均发生在fixture写入前，仅修正检查脚本的源码包路径与绝对目录，原错误日志保留。最终临时完整模型599063649 bytes，按路径和SHA核对后已删除，新测试权重留存0。当前固定训练/接续队列的源码、manifest和设置没有改变。\n\n'
addition+='正式通过后的真实终点导出、Nr/Sr实际加载、真实训练数据梯度与固定训练配置仍待接续；完整V99过线不自动证明原生网络达到V99。文档`docs/NATIVE_BOX_INITIALIZATION_2026-09-07.md`给出执行入口和边界。本轮CPU receipt SHA `'+hashlib.sha256((archive/'receipt.json').read_bytes()).hexdigest()+'`；导出脚本SHA `'+receipt['export_script_sha256']+'`。三数据集目标仍未完成，未触发新正式评估。\n'
new=old+addition.encode()
tracker=repo/'refine-logs/EXPERIMENT_TRACKER.md'
lines=tracker.read_text(encoding='utf-8').splitlines()
lines[2]='Updated: '+now+'. Section20.126: native box initializer8CPUtests+Nr/Sr actual model-loader fixture PASS;noGPU/actualendpoint/transfer result;fixedScantraining continues.'
lines.insert(6,'| Native box-head initialization export | 8CPUtests PASS;actual Nr/Sr native loader each1144state exact on explicit synthetic delta fixture;fresh optimizer/scheduler | Future true endpoint only after formalScan pass;temporary599MBfixture deleted;0 GPU/actual training/quality claim |')
tracker_bytes=('\n'.join(lines)+'\n').encode()
client=paramiko.SSHClient();client.load_system_host_keys();client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
sftp=client.open_sftp();runtime='/home/gb/new butd/butd_detr-main/MCLN-main/'
with sftp.open(runtime+'docs/'+master.name,'rb') as stream:
    stream.prefetch(file_size=len(old));assert stream.read()==old
for path,data in [('docs/'+master.name,new),('refine-logs/EXPERIMENT_TRACKER.md',tracker_bytes)]:
    with sftp.open(runtime+path,'wb') as stream:stream.set_pipelined(True);stream.write(data)
    with sftp.open(runtime+path,'rb') as stream:stream.prefetch(file_size=len(data));assert stream.read()==data
synced={}
for name in ['scripts/export_native_box_transfer_initialization.py','tests/test_native_box_transfer_initialization.py','docs/NATIVE_BOX_INITIALIZATION_2026-09-07.md']:
    data=(repo/name).read_bytes()
    with sftp.open(runtime+name,'wx') as stream:stream.write(data)
    with sftp.open(runtime+name,'rb') as stream:assert stream.read()==data
    synced[name]=hashlib.sha256(data).hexdigest()
master.write_bytes(new);desktop.write_bytes(new);tracker.write_bytes(tracker_bytes)
attribute=repo/'.gitattributes';rule='refine-logs/native_box_initialization_preparation_20260907_v1/** -text'
assert rule not in attribute.read_text(encoding='utf-8')
with attribute.open('a',encoding='utf-8',newline='\n') as stream:stream.write(rule+'\n')
proof={'section':'20.126','time_cst':now,'master_sha256':hashlib.sha256(new).hexdigest(),'master_bytes':len(new),
       'three_master_copies_equal':True,'runtime_new_files':synced,'actual_endpoint_used':False,'new_formal_rows':0,'goal_complete':False}
remote='/root/autodl-tmp/mcln_native_box_initialization_preparation_20260907_v1'
for name,data in [('handoff_sync.json',(json.dumps(proof,indent=2)+'\n').encode()),('document_from_local.py',Path(__file__).read_bytes())]:
    (archive/name).write_bytes(data)
    with sftp.open(remote+'/'+name,'wx') as stream:stream.write(data)
sftp.close();client.close();print(json.dumps(proof),flush=True)
