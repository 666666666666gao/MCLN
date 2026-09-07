import datetime,hashlib,json,os
from pathlib import Path
import paramiko
repo=Path('C:/Users/gb/.codex_mcln_g0_20260905')
local=repo/'refine-logs/scanrefer_native_box_transfer_posttraining_20260907_v1'
remote='/root/autodl-tmp/mcln_scanrefer_native_box_transfer_posttraining_20260907_v1'
runtime='/home/gb/new butd/butd_detr-main/MCLN-main/'
master=repo/'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md';desktop=Path('C:/Users/gb/Desktop/document')/master.name
old=master.read_bytes();assert hashlib.sha256(old).hexdigest()=='64aa074dfb6d251fed25089c385fd1b3d454ccc57a74cd8b0848a52c0c18e951'
assert desktop.read_bytes()==old
launch=json.loads((local/'launch.json').read_bytes())
spec=json.loads((local/'input_manifest.json').read_bytes())
now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
files=['scripts/audit_scanrefer_mesh_teacher_transfer.py','scripts/check_mesh_teacher_transfer_rows.py',
 'scripts/native_teacher_box_transfer.py','scripts/probe_scanrefer_native_box_transfer.py',
 'scripts/run_scanrefer_native_box_transfer_pair.py','scripts/audit_scanrefer_native_box_transfer_pair.py',
 'scripts/evaluate_scanrefer_native_box_transfer_official.py','scripts/audit_scanrefer_native_box_transfer_official.py',
 'scripts/queue_scanrefer_native_box_transfer_posttraining.py',
 'tests/test_native_teacher_box_transfer.py','tests/test_native_box_transfer_promotion.py','tests/test_native_box_transfer_queue.py']
c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
s=c.open_sftp()
existing={folder:set(s.listdir(runtime+folder)) for folder in ['scripts','tests']}
source_sync={}
for name in files:
 raw=(repo/name).read_bytes();folder,filename=name.split('/')
 if filename in existing[folder]:
  with s.open(runtime+name,'rb') as f:assert f.read()==raw,name
 else:
  with s.open(runtime+name,'wx') as f:f.set_pipelined(True);f.write(raw)
 with s.open(runtime+name,'rb') as f:assert f.read()==raw,name
 source_sync[name]=hashlib.sha256(raw).hexdigest()
with s.open(runtime+'docs/'+master.name,'rb') as f:
 f.prefetch(file_size=len(old));assert f.read()==old
addition='\n\n### 20.123 固定终态与正式评估自动接续已上线（'+now+'）\n\n'
addition+='接续队列13:25:11实际启动：screen58294.mcln_native_box_transfer_post_v1，Python58296；原训练58020/58023保持运行。队列绑定本轮训练manifest、独立formal-preparation文件SHA、固定候选gt_teacher_box及同一mesh数据。312份val superpoint再次逐文件核对一致，未执行新正式forward。第一次观察安排13:40 CST，此后后台每240秒验证具体screen进程并读取日志，不重启训练。\n\n'
addition+='训练控制器已串联CPU独立审计，队列只读取这份审计，不重复执行会以独占方式写文件的审计脚本。若固定候选在模块系统REC任一阈值低于起点或GT-only控制，写出未通过决定且不运行正式评估；若通过，按实际终点SHA生成唯一9508正式manifest，加载E71+16参数终点，评估保护/控制/候选三臂并独立重算指标。正式条件通过后明确转入Nr/Sr REC接续检查，当前不提前启动Nr/Sr，也不替换候选或重选epoch。\n\n'
addition+='准备阶段发现根目录queue.py遮蔽Python标准库queue，pytest的Dash/requests插件导入报AttributeError。该队列当时尚未启动；仅将入口改名posttraining_queue.py并更新controller，随后原Python3.7环境8项queue/promotion测试全部通过。没有重建环境、增加fallback或修改运行训练源码。\n\n'
addition+='13:23:50实际训练检查：预处理结束，baseline已记录1536/6887行，elapsed407.47秒，两个原进程存活。按该已记录吞吐估计完整起点约1827秒，约13:47附近结束，实际以日志为准；尚无完整baseline指标或本轮更新记录。接续queue manifest SHA `'+launch['manifest_sha256']+'`，脚本 SHA `'+spec['queue_script_sha256']+'`。当前仍无新的正式成绩。\n\n'
addition+='为保证远端项目入口可检查，本次把9份新增实验脚本和3份对应测试同步到远端MCLN-main的scripts/tests，逐文件字节核对；正在运行的614-file冻结来源和每轮overlay未改动。完整自动接续尚未走到终态，不能把队列启动或CPU测试视为晋级成功。\n'
new=old+addition.encode()
tracker=repo/'refine-logs/EXPERIMENT_TRACKER.md';lines=tracker.read_text(encoding='utf-8').splitlines()
lines[2]='Updated: '+now+'. Section20.123: trainingPID58023 baseline1536/6887;posttrainingqueuePID58296 live,firstcheck13:40 then240s;conditional singleformal9508+audit chained;no new formal.'
lines.insert(6,'| Native box transfer automatic continuation | Queue58294/Python58296 live;fixed source/data hashes verified;8 originalCPUtests PASS | Read controller audit once;only fixed moduleREC pass triggers9508 formal+audit;no candidate/epoch selection;Nr/Sr pendingScan |')
tracker_raw=('\n'.join(lines)+'\n').encode()
for rel,raw in [('docs/'+master.name,new),('refine-logs/EXPERIMENT_TRACKER.md',tracker_raw)]:
 with s.open(runtime+rel,'wb') as f:f.set_pipelined(True);f.write(raw)
 with s.open(runtime+rel,'rb') as f:f.prefetch(file_size=len(raw));assert f.read()==raw
master.write_bytes(new);desktop.write_bytes(new);tracker.write_bytes(tracker_raw)
proof={'time_cst':now,'section':'20.123','master_sha256':hashlib.sha256(new).hexdigest(),'master_bytes':len(new),
 'three_master_copies_equal':True,'runtime_source_files_synchronized':source_sync,'new_formal_rows':0,'goal_complete':False}
(local/'handoff_sync.json').write_bytes((json.dumps(proof,indent=2)+'\n').encode())
(local/'document_from_local.py').write_bytes(Path(__file__).read_bytes())
for name in ['handoff_sync.json','document_from_local.py']:
 with s.open(remote+'/'+name,'wx') as f:f.write((local/name).read_bytes())
with (repo/'.gitattributes').open('a',encoding='utf-8',newline='\n') as f:
 f.write('refine-logs/scanrefer_native_box_transfer_posttraining_20260907_v1/** -text\n')
s.close();c.close();print(json.dumps(proof),flush=True)
