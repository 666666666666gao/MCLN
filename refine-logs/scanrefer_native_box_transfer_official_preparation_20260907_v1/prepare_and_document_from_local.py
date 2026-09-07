import datetime,hashlib,json,os,shlex
from pathlib import Path
import paramiko
repo=Path('C:/Users/gb/.codex_mcln_g0_20260905')
train=repo/'refine-logs/scanrefer_native_box_transfer_pair_20260907_v1'
train_remote='/root/autodl-tmp/mcln_scanrefer_native_box_transfer_pair_20260907_v1'
prep=repo/'refine-logs/scanrefer_native_box_transfer_official_preparation_20260907_v1'
prep.mkdir(exist_ok=False);(prep/'scripts').mkdir();(prep/'tests').mkdir()
remote='/root/autodl-tmp/mcln_scanrefer_native_box_transfer_official_preparation_20260907_v1'
runtime='/home/gb/new butd/butd_detr-main/MCLN-main/'
source=json.loads((train/'input_manifest.json').read_bytes())['model_source']
names=['scripts/evaluate_scanrefer_native_box_transfer_official.py','scripts/audit_scanrefer_native_box_transfer_official.py',
 'scripts/audit_scanrefer_joint_readout_pair.py','scripts/scanrefer_data_contract.py','scripts/scanrefer_joint_readout.py',
 'scripts/scanrefer_rec_evaluation.py','tests/test_native_box_transfer_promotion.py']
files={}
for name in names:
 raw=(repo/name).read_bytes();(prep/name).write_bytes(raw);files[name]=hashlib.sha256(raw).hexdigest()
(prep/'scripts/__init__.py').write_bytes(b'');files['scripts/__init__.py']=hashlib.sha256(b'').hexdigest()
c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
s=c.open_sftp();s.mkdir(remote);s.mkdir(remote+'/scripts');s.mkdir(remote+'/tests')
for p in prep.rglob('*'):
 if p.is_file():s.put(str(p),remote+'/'+p.relative_to(prep).as_posix())
command='cd '+shlex.quote(source)+' && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='+shlex.quote(remote+':'+source)+' /root/miniconda3/envs/bdetr/bin/python -m pytest -q '+shlex.quote(remote+'/tests/test_native_box_transfer_promotion.py')
_,o,e=c.exec_command(command);test=o.read().decode()+e.read().decode();assert o.channel.recv_exit_status()==0,test
(prep/'cpu_tests.txt').write_bytes(test.encode());print(test,flush=True)
_,o,e=c.exec_command('ps -p 58020,58023 -o pid,ppid,comm,stat,etime,args\ntail -n 7 '+train_remote+'/run.log\nnvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader\ndf -B1 /root/autodl-tmp',timeout=30)
live=o.read().decode();assert '58023' in live and 'run_scanrefer_native_box_transfer_pair.py' in live,live
now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
observation={'time_cst':now,'read_only':True,'specific_training_pid':58023,'live':live,'training_files_not_modified':True}
raw=(json.dumps(observation,indent=2)+'\n').encode();(train/'launch_observation.json').write_bytes(raw)
with s.open(train_remote+'/launch_observation.json','wx') as f:f.write(raw)
print(json.dumps(observation),flush=True)
preparation={'schema':'mcln-native-box-transfer-official-preparation-v1','time_cst':now,'files':files,
 'native_checkpoint_restore':'protected E71 plus exact16 saved center/size parameter tensors',
 'candidate_predeclared':'gt_teacher_box_v99','control':'gt_only_v99','formal_rows_executed':0,
 'formal_manifest_not_yet_bound':'requires terminal training receipt,independent audit and both fixed endpoint SHA',
 'training_files_not_modified':True,'original_environment_promotion_tests':4,'automatic_formal_not_started':True}
(prep/'preparation.json').write_bytes((json.dumps(preparation,indent=2)+'\n').encode())
for name in ['preparation.json','cpu_tests.txt']:s.put(str(prep/name),remote+'/'+name)
master=repo/'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md';desktop=Path('C:/Users/gb/Desktop/document')/master.name
old=master.read_bytes();assert hashlib.sha256(old).hexdigest()=='66750ca9fa5246cab67f5ce346324bf541b5dde26981a89a402586447da69ae2'
assert desktop.read_bytes()==old
with s.open(runtime+'docs/'+master.name,'rb') as f:
 f.prefetch(file_size=len(old));assert f.read()==old
addition='\n\n### 20.122 原生教师框转移固定配对已启动；小终点正式入口准备（'+now+'）\n\n'
addition+='13:08:58 CST实际启动记录：screen58020.mcln_native_box_transfer_pair_v1，实际Python58023，来源GitHub main `3c4db12cca4027cbe600e6c8e510df4979709997`；运行manifest SHA `2e7b7fa65afd056c0ad98a9a92e974e563df6e104624abdb30cfaf4716e1e72a`。起动前A100仅1MiB、磁盘剩余9304215552 bytes（约8.665GiB）。控制器顺序执行固定配对及CPU独立审计，无需保持本地连接。当前再次实际ps确认58023存活，精确日志和GPU状态见launch_observation.json；不按启动标记代替进程证据。\n\n'
addition+='本轮仍先做6887条一致起点评估，再完成2482更新/臂及完整终态。初始排程估计起点评估约35分钟、全轮约4小时，实际以本轮吞吐更新；没有因估计到时就重启。后续按阶段预计结束时间检查，长任务观察间隔240秒；本次没有新正式成绩或Nr/Sr训练。\n\n'
addition+='提前准备`evaluate_scanrefer_native_box_transfer_official.py`及对应审计：读取完成且通过筛选的本轮receipt，加载E71再覆盖绑定的16项head参数，核对shape/dtype/完整state，然后使用原生9508loader和固定V99部署路径。原生REC单独记录，GT只在评估计数处使用。当前仅代码准备和原环境4项晋级规则CPU测试通过，未绑定未来checkpoint/receipt SHA，也未运行正式评估；不是宣称小终点已通过正式独立reload。训练目录及其已锁定代码没有改动。\n\n'
addition+='现行规则保持：候选正式Scan REC不少于5572/4797且不低于同次保护控制，Scan Mask不少于58.70/50.70/44.72；达到底线即接Nr/Sr REC，不等待59/51，也不加Nr/Sr Mask门。当前几何转移是性能衔接，不等于已解决Nr复杂语言泛化；三数据集整体目标仍未完成。\n'
new=old+addition.encode()
tracker=repo/'refine-logs/EXPERIMENT_TRACKER.md';lines=tracker.read_text(encoding='utf-8').splitlines()
lines[2]='Updated: '+now+'. Section20.122: native GT-only/teacher-box pair live PID58023,2482updates per arm planned;no terminal or formal yet;small-endpoint formal entry4CPUtests PASS.'
for i,line in enumerate(lines):
 if line.startswith('| Native teacher-box transfer |'):
  lines[i]='| Native teacher-box transfer | Actual16fit probe PASS;fixed pair started13:08 PID58023;only16box tensors/335814params | 2482/arm,29778fit/6887moduleholdout;CPU audit chained;formal entry prepared but not run;noNr/Sr |'
tracker_raw=('\n'.join(lines)+'\n').encode()
for rel,value in [('docs/'+master.name,new),('refine-logs/EXPERIMENT_TRACKER.md',tracker_raw)]:
 with s.open(runtime+rel,'wb') as f:f.set_pipelined(True);f.write(value)
 with s.open(runtime+rel,'rb') as f:f.prefetch(file_size=len(value));assert f.read()==value
master.write_bytes(new);desktop.write_bytes(new);tracker.write_bytes(tracker_raw)
proof={'time_cst':now,'section':'20.122','three_master_copies_equal':True,'master_sha256':hashlib.sha256(new).hexdigest(),
 'master_bytes':len(new),'specific_training_pid':58023,'active_goal_unmet':True}
(prep/'handoff_sync.json').write_bytes((json.dumps(proof,indent=2)+'\n').encode())
(prep/'prepare_and_document_from_local.py').write_bytes(Path(__file__).read_bytes())
with (repo/'.gitattributes').open('a',encoding='utf-8',newline='\n') as stream:
 stream.write('refine-logs/scanrefer_native_box_transfer_official_preparation_20260907_v1/** -text\n')
s.close();c.close();print(json.dumps(proof),flush=True)
