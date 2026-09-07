import datetime,hashlib,json,os,subprocess
from pathlib import Path
import paramiko

repo=Path('C:/Users/gb/.codex_mcln_g0_20260905')
archive='refine-logs/mask_geometry_initialization_preparation_20260907_v1'
record=json.loads((repo/archive/'receipt.json').read_bytes())
assert (repo/archive/'controller.exit').read_text().strip()=='0'
assert record['status']=='pass' and record['synthetic_fixture_only'] and record['temporary_fixture_deleted']
master=repo/'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md';desktop=Path('C:/Users/gb/Desktop/document')/master.name
old=master.read_bytes();assert desktop.read_bytes()==old
assert hashlib.sha256(old).hexdigest()=='78e06f79a5b38c4512dbdf29001cfb8c46178ab0fc22e17ba4b94b5509710fce'
now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
addition='''

### 20.136 当前84项终点的Nr/Sr原生初始化准备通过（TIME）

新增`export_mask_geometry_initialization.py`，仅在固定候选正式Scan晋级并重算审计后导出。以E71补全1144项，只覆盖当前84项core参数；明确包含注意力in_proj_weight/in_proj_bias，保留全部BN缓冲，不继承旧optimizer/scheduler。使用已有model_only_initialization，未改原生加载器或运行中的训练源。旧20.126的16项框头导出仍保留历史记录，当前不能使用。

原环境8项格式检查通过（1.21秒）。以真实E71及明确标注的84项合成差值，18:47:38完成Nr/Sr实际原生模型工厂、优化器工厂和checkpoint加载：各1144项state逐值一致，84项真实named_parameters覆盖、1060项冻结state保留，optimizer为空、scheduler不变、start_epoch1。检查native source618文件。加载阶段6.83秒；本检查GPU前向0、训练更新0、正式行0，没有使用尚未产生的真实终点。

临时599066465-byte完整fixture核对SHA及限定路径后已删除，剩余测试权重0。receipt SHA `73f5c9afb676035e4993d0077ea578b469e5507c3bf2962b4c3d3bb0ca5fc97a`，导出脚本SHA `048ca46542cc15fd020c46fc70c382428116dba163a4919d7b7a39e9f2f6c9a1`。归档`refine-logs/mask_geometry_initialization_preparation_20260907_v1`，入口与边界见`docs/MASK_GEOMETRY_NATIVE_INITIALIZATION_2026-09-07.md`。

这完成的是可执行的跨数据集初始化接口，不证明跨数据集效果。Mask几何辅助是训练损失，初始化文件不会自动启用该损失；正式Scan通过后仍须绑定真实终点、Nr/Sr原生数据批次和训练监督，实际GPU梯度核验后启动。当前Scan固定训练及接续队列不改设置，按20.135的19:10/240秒计划观察；原生与V99性能分开、既有REC/ScanMask晋级线不变。完整目标未完成。
'''.replace('TIME',now)
new=old+addition.encode()
tracker=repo/'refine-logs/EXPERIMENT_TRACKER.md';text=tracker.read_text(encoding='utf-8')
row='| Mask geometry84-core native initialization | 8CPU format tests and actual Nr/Sr native loaders PASS;1144state exact,84changed/1060preserved on synthetic fixture | Ready only for future formally promoted endpoint;GPU0/actualtraining0;temporary599MBfixture deleted |\n'
text=text.replace('|---|---|---|\n','|---|---|---|\n'+row,1);tracker.write_bytes(text.encode())
paths=['scripts/export_mask_geometry_initialization.py','tests/test_mask_geometry_initialization.py',
       'docs/MASK_GEOMETRY_NATIVE_INITIALIZATION_2026-09-07.md','docs/NATIVE_BOX_INITIALIZATION_2026-09-07.md',
       'refine-logs/EXPERIMENT_TRACKER.md']
c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
s=c.open_sftp();runtime='/home/gb/new butd/butd_detr-main/MCLN-main/'
with s.open(runtime+'docs/'+master.name,'rb') as f:f.prefetch(file_size=len(old));assert f.read()==old
training='/root/autodl-tmp/mcln_scanrefer_mask_geometry_pair_20260907_v1'
with s.open(training+'/input_manifest.json','rb') as f:raw=f.read()
assert hashlib.sha256(raw).hexdigest()=='15f46411069a7172a55373c5c13075b22bcb4146d39251e9fca2c37ed5867eb3'
for name,digest in json.loads(raw)['files'].items():
    with s.open(training+'/'+name,'rb') as f:assert hashlib.sha256(f.read()).hexdigest()==digest,name
for name in paths:
    raw=(repo/name).read_bytes()
    with s.open(runtime+name,'wb') as f:f.set_pipelined(True);f.write(raw)
    with s.open(runtime+name,'rb') as f:f.prefetch(file_size=len(raw));assert f.read()==raw
with s.open(runtime+'docs/'+master.name,'wb') as f:f.set_pipelined(True);f.write(new)
with s.open(runtime+'docs/'+master.name,'rb') as f:f.prefetch(file_size=len(new));assert f.read()==new
master.write_bytes(new);desktop.write_bytes(new)
proof={'time_cst':now,'master_sha256':hashlib.sha256(new).hexdigest(),'three_master_copies_equal':True,
       'training_source_and_manifest_unchanged':True,'initialization_tests':'pass','actual_endpoint_loaded':False,
       'new_formal_rows':0,'goal_complete':False}
(repo/archive/'handoff_sync.json').write_bytes((json.dumps(proof,indent=2)+'\n').encode())
(repo/archive/'publish_from_local.py').write_bytes(Path(__file__).read_bytes())
remote='/root/autodl-tmp/mcln_mask_geometry_initialization_preparation_20260907_v1'
for name in ['stage_from_local.py','collect_from_local.py','publish_from_local.py','handoff_sync.json','launch.json']:
    s.put(str(repo/archive/name),remote+'/'+name)
s.close();c.close()
with (repo/'.gitattributes').open('a',encoding='utf-8',newline='\n') as f:f.write(archive+'/** -text\n')
def git(*args):return subprocess.check_output(['git']+list(args),cwd=repo)
assert not git('diff','--cached','--name-only').strip()
paths+=['docs/'+master.name,'.gitattributes',archive]
git('add','--',*paths);git('add','--renormalize','--',archive)
names=git('diff','--cached','--name-only','-z').decode().strip('\0').split('\0')
for name in names:
    assert name in paths or name.startswith(archive+'/'),name
    raw=git('show',':'+name)
    assert os.environ['MCLN_SSH_PASSWORD'].encode() not in raw
    assert Path(name).suffix.lower() not in ['.pth','.pt','.pkl','.npz'] and len(raw)<10*1024**2,name
    if name.startswith(archive+'/'):assert raw==(repo/name).read_bytes(),name
git('-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol','diff','--cached','--check')
print(git('commit','-m','Prepare84-core initialization for native Nr3D and Sr3D training').decode(),flush=True)
print(git('push','origin','HEAD:main').decode(),flush=True)
head=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','origin','refs/heads/main').decode().split()[0]==head
assert not git('status','--porcelain').strip()
proof['commit']=head
Path('C:/Users/gb/.codex/tmp/mask_geometry_initialization_publication.json').write_bytes((json.dumps(proof,indent=2)+'\n').encode())
print(json.dumps(proof),flush=True)
