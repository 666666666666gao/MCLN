import datetime,gzip,hashlib,json,os,subprocess
from pathlib import Path
import paramiko

repo=Path('C:/Users/gb/.codex_mcln_g0_20260905')
archives=['refine-logs/pretrained_object_source_feasibility_20260907_v1','refine-logs/scanrefer_openshape_cache_20260908_v1','refine-logs/scanrefer_object_appearance_native_20260908_v1','refine-logs/scanrefer_object_appearance_pair_20260908_v1']
master='docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
tracker='refine-logs/EXPERIMENT_TRACKER.md'
docs=['docs/OPENSHAPE_RUNTIME_2026-09-07.md','docs/OPENSHAPE_VISUAL_SOURCE_PREFLIGHT_2026-09-07.md','docs/SCANREFER_PRETRAINED_OBJECT_INPUT_RESULT_2026-09-07.md','docs/SCANREFER_OBJECT_APPEARANCE_INTEGRATION_2026-09-08.md','docs/SCANREFER_OBJECT_APPEARANCE_PAIR_PLAN_2026-09-08.md']
code=['models/mcln.py','models/pretrained_object_appearance.py','scripts/export_scanrefer_pretrained_object_inputs.py','scripts/probe_openshape_pretrained_objects.py','scripts/analyze_openshape_object_probe.py','scripts/analyze_openshape_feature_separation.py','scripts/cache_scanrefer_openshape_objects.py','scripts/probe_pretrained_object_appearance_contract.py','scripts/probe_scanrefer_pretrained_object_memory.py','scripts/run_scanrefer_object_appearance_pair.py']
def git(*args):return subprocess.check_output(['git']+list(args),cwd=str(repo))
assert git('rev-parse','HEAD').decode().strip()=='38c46a38fb1465c187aacf161c4f097223f7f8da'
assert not git('diff','--cached','--name-only').strip()
old=(repo/master).read_bytes();assert hashlib.sha256(old).hexdigest()=='9bd64dd129c41b3174caa503e3eec26abd4fbf79c65fb56a05f4e7003b871542'
desktop=Path('C:/Users/gb/Desktop/document')/Path(master).name;assert desktop.read_bytes()==old
native=json.loads((repo/archives[2]/'receipt.json').read_bytes());assert native['status']=='pass' and native['original_state_tensors_preserved']==1144
cache=json.loads((repo/archives[1]/'receipt.json').read_bytes());assert cache['available_slots']==12392
now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
addition='''

### 20.143 冻结预训练对象外观：真实源检查、原生接入与ScanRefer配对启动（TIME）

20.142之后核对历史实现：SourceMoE、SACR/JQQ、Tier hard-query、Relation-CF已有质量或最终分数监督，因此不把普通IoU评分损失重复作为新方向；这些是分别存在的实现，非保护E71全部同时启用。当前对象记忆由128维框位置与160维类别组成，进入3层双向编码器及6层Decoder。新检查改为官方OpenShape G14 RGB冻结视觉证据，非重启O1末层轻量MLP或Source Selector。

固定前16条既有fit记录，10个场景、591次对象槽、314个去重槽；原生50000 XYZ/RGB SHA全部匹配真实E71 forward。GroupFree预测框裁剪，无GT框/实例mask清洗。409次/228个去重槽达到官方384 FPS点数要求；182次/86个槽不足。真实冻结编码全部完成，8.81793s、峰值321.08MiB。181个重复槽余弦中位数0.995989，同场景同预测类别不同槽304对中位数0.768863；重复槽检索181/181。此为槽分离和采样稳定性，非语义身份/REC准确率。床类有合理语义，部分柜子/椅子匹配仍偏离，不提前宣称预训练迁移有效。

官方权重SHA34949c162aca01b6fd3147ed7ccf34b448a34bdebc4a857605ea62412ad54fb9，官方推理源70dbc29、权重d771a99。保留原conda bdetr/Torch1.10.2cu111，新增DGL/redstone/networkx到独立prefix。首build缺networkx退出1原样保留；v2声明spec SHAfe8ac66b8f51ed0e179a279717d0d1c2213e9a6b463785ccaa6a6e5eaf2385ac，安装/真实CUDA witness/独立fresh Codex按文档执行均通过，原base包清单未变。review为同模型家族、provisional，仅证明runtime。局部分析从真实NPZ独立重算，和远端浮点差<1e-6。

随后完成全部562个ScanRefer训练场景的冻结缓存：16759槽，12392可编码，4367不足384点；实际编码310.18s。每场景/槽固定种子，同50000点、相同预测框，逐scene保存feature、available、box、point-count及SHA。未使用文本或GT标签。cache receipt SHA5bac70c4deb1166890fc80dd0c430f3fa76fda3e9ede17f83297b10a2bafc48b。缓存只供训练场景，目前未生成正式验证预测或Nr/Sr缓存。

新增默认关闭的PretrainedObjectAppearance：1280→160无偏置零初始化投影，共204800参数，加入原对象记忆语义部分；128位置维保持，无外观槽贡献为0。CPU身份/位置/缺失槽/置换/梯度检查通过。原生625文件隔离源码manifest SHA190d0011bc5bfefab4a3965d972606f4dc21a946f68b2382f5a90837b3a4c4ef；真实16条Scan fit、batch4逐输入点SHA/框/缓存对齐通过，初始框、Mask及V99 runtime均严格相同；GT投影梯度范数11.96—33.43。两次一次性试更新loss10.52475→9.90343，原1144状态张量全保持、无权重保存。此为工程通过，非性能增益；试更新权重已丢弃。

固定ScanRefer新配对已在00:40 CST启动screen70235.mcln_os_appearance_pair_v1。两组同E71、correct-mesh、29778 fit/6887模块留出、batch12、每臂2482更新/1完整遍历；同原生GT损失，cross_encoder/全部Decoder/预测头学习率1e-6，候选另训外观投影1e-4，AdamW wd0.0005/clip0.1。现有主干与语言编码、其他参数buffer、Parent/Geometry/V99保持；先做全选定参数batch12容量梯度检查，后起点评估、训练、终点评估及审计。对照忽略相同batch附带的外观字段。模块留出是主干已见训练场景，正式三数据集结果仍未更新。00:42:10实际PID70238存活，处于全量训练数据文本准备，尚无capacity或optimizer-step记录；不能将screen启动算完成训练。

当前研究目标及Scan过线再Nr/Sr REC顺序不变。新方案是采用已有预训练并改变原生信息入口，不宣称普通投影为原创强创新、不宣称去除V99后处理。下一步按已固定计划完成两臂，分别检查原生和完整系统REC修复/破坏与Scan Mask底线；通过后固定9508正式核验，随后尽快Nr/Sr。不得以两步loss下降或181/181槽检索替代REC成绩。

代码、计划、真实日志/receipt归档pretrained_object_source_feasibility_20260907_v1、scanrefer_openshape_cache_20260908_v1、scanrefer_object_appearance_native_20260908_v1、scanrefer_object_appearance_pair_20260908_v1。下载权重/依赖/特征NPZ不入Git；已删本地4个验证过的重复传输文件659353724B，远端运行权重、全部保护模型及证据保留。完整Goal仍未完成。
'''.replace('TIME',now)
new=old+addition.encode()
lines=(repo/tracker).read_text(encoding='utf-8').splitlines();lines[2]='Updated: '+now+'. Section20.143: frozen object source/cache/native probe complete; paired ScanRefer job launched, no new formal metric.'
lines.insert(6,'| Pretrained native object appearance | Frozen source409 forwards;562-scene cache complete;16-row native parity/gradient PASS;Scan paired job launched | Same E71,2482 updates/arm planned;all multimodal encoder/Decoder/prediction heads;no completed REC gain;see20.143 |')
new_tracker=('\n'.join(lines)+'\n').encode()
c=paramiko.SSHClient();c.load_system_host_keys();c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30);s=c.open_sftp();canonical='/home/gb/new butd/butd_detr-main/MCLN-main/'
for name in [master,tracker]:
    with s.open(canonical+name,'rb') as f:assert f.read()==git('show','HEAD:'+name),name
# Preserve any unrelated canonical model changes; apply only the reviewed hunk.
with s.open(canonical+'models/mcln.py','rb') as f:remote_model=f.read()
remote_text=remote_model.decode();newline='\r\n' if '\r\n' in remote_text else '\n'
normalized=remote_text.replace('\r\n','\n')
assert hashlib.sha256(remote_model).hexdigest()=='a9301a5fc9bac3b40e4450350a8d2eb4ba11c4763e734c5a09723a5232474db4'
canonical_patches=json.loads((repo/archives[2]/'patches.json').read_bytes())
canonical_patches[0]=('                 parent_relative_text_verifier_filter_non_gt_boxes=False):',
                      '                 parent_relative_text_verifier_filter_non_gt_boxes=False,\n                 use_pretrained_object_appearance=False):')
for before,after in canonical_patches:
    assert normalized.count(before)==1,(before,normalized.count(before));normalized=normalized.replace(before,after)
updated_model=normalized.replace('\n',newline).encode()
with s.open(canonical+'models/mcln.py','wb') as f:f.write(updated_model)
patch_proof=dict(before_sha256=hashlib.sha256(remote_model).hexdigest(),after_sha256=hashlib.sha256(updated_model).hexdigest(),scope='Only three appearance hunks; preserve unrelated canonical differences. Actual runtime is the separately hashed625-file source.')
(repo/archives[2]/'canonical_patch.json').write_bytes(json.dumps(patch_proof,indent=2).encode()+b'\n')
for name,content in [(master,new),(tracker,new_tracker)]+[(name,(repo/name).read_bytes()) for name in docs+code if name!='models/mcln.py']:
    with s.open(canonical+name,'wb') as f:f.set_pipelined(True);f.write(content)
    with s.open(canonical+name,'rb') as f:assert f.read()==content,name
(repo/master).write_bytes(new);desktop.write_bytes(new);(repo/tracker).write_bytes(new_tracker)
proof=dict(time_cst=now,three_master_copies_equal=True,master_sha256=hashlib.sha256(new).hexdigest(),goal_complete=False,paired_run_root='/root/autodl-tmp/mcln_scanrefer_object_appearance_pair_20260908_v1')
(repo/archives[3]/'handoff_sync.json').write_bytes(json.dumps(proof,indent=2).encode()+b'\n')
(repo/archives[3]/'publish_from_local.py').write_bytes(Path(__file__).read_bytes())
paths=[master,tracker,'.gitignore','.gitattributes']+docs+code+archives
git('add','--',*paths)
staged=git('diff','--cached','--name-only','-z').decode().strip('\0').split('\0')
for name in staged:
    assert name in paths or any(name.startswith(p+'/') for p in archives),name
    content=git('show',':'+name);inspected=gzip.decompress(content) if name.endswith('.gz') else content
    assert os.environ['MCLN_SSH_PASSWORD'].encode() not in inspected
    assert len(content)<10*1024**2 and Path(name).suffix not in ['.pt','.pth','.pkl','.npz','.whl'],name
    if any(name.startswith(p+'/') for p in archives):assert content==(repo/name).read_bytes(),name
# Mirror text evidence into the canonical repository, never over a live job log.
existing=set(s.listdir(canonical+'refine-logs'))
for ar in archives:
    if Path(ar).name not in existing:s.mkdir(canonical+ar)
    nested=set()
    for name in [v for v in staged if v.startswith(ar+'/')]:
        rel=Path(name).relative_to(ar)
        for parent in reversed(rel.parents):
            if str(parent)=='.':continue
            if str(parent) not in nested:s.mkdir(canonical+ar+'/'+parent.as_posix());nested.add(str(parent))
        s.put(str(repo/name),canonical+name)
s.close();c.close()
git('-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol','diff','--cached','--check')
print(git('commit','-m','Add pretrained object appearance and launch controlled ScanRefer adaptation').decode(),flush=True)
print(git('push','origin','HEAD:main').decode(),flush=True)
head=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','origin','refs/heads/main').decode().split()[0]==head
assert not git('status','--porcelain').strip()
proof['commit']=head;Path('C:/Users/gb/.codex/tmp/object_appearance_publication_20260908.json').write_bytes(json.dumps(proof,indent=2).encode())
print(json.dumps(proof),flush=True)
