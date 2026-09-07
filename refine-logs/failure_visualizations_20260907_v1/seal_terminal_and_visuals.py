import datetime,hashlib,json,os
from pathlib import Path
import paramiko

repo=Path('C:/Users/gb/.codex_mcln_g0_20260905')
root=Path('C:/Users/gb/Desktop/document/MCLN_3D_failure_visualizations_20260907')
master=repo/'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
desktop=root.parent/master.name
runtime='/home/gb/new butd/butd_detr-main/MCLN-main/'
evidence=repo/'refine-logs/scanrefer_frozen_readout_pair_20260907_v1'
decision=json.loads((repo/'refine-logs/scanrefer_frozen_readout_posttraining_20260907_v1/decision.json').read_bytes())
check=json.loads((evidence/'terminal_collection_check.json').read_bytes())
assert decision['status']=='module_rec_screen_not_passed' and check['integrity_pass']
assert not check['eligible_for_fixed_terminal_formal_evaluation']
old=master.read_bytes()
assert hashlib.sha256(old).hexdigest()=='e5a2b9fdcbc908bb35fb16349252e36e753274f86badd3b270c080f9bb413f37'
assert desktop.read_bytes()==old
cases=json.loads((root/'cases.json').read_bytes())
browser=json.loads((root/'browser_verification.json').read_bytes())
assert len(cases['cases'])==len(browser['pages'])==6 and not browser['page_errors']
now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y-%m-%d %H:%M CST')
addition='''

### 20.117 冻结读出兼容配对终态：未通过模块 REC 筛选（2026-09-07 11:54 CST 实际收取）

原训练与接续队列自然结束，两份controller.exit均为0，52529/52535已退出。两臂各2482步、29778条fit表达、6887条模块holdout，完整终态及独立审计已收取。11:50:39队列判定module_rec_screen_not_passed；本轮没有运行9508正式评估，没有Nr/Sr训练，也没有改变设置或重选终点。

| 6887条模块holdout | 系统REC@0.25/@0.50 hits | 原生REC hits | Mask hits@0.25/@0.50 | Mask mIoU |
|---|---:|---:|---:|---:|
| 相同起点 | 6684/6426 | 6572/5955 | 6511/6097 | 77.81086079% |
| native_only终点 | 6679/6449 | 6559/5994 | 6509/6096 | 77.79712060% |
| frozen_gt终点 | 6682/6442 | 6572/5987 | 6509/6099 | 77.82407767% |

固定候选系统REC相对起点−2/+16，相对native_only +3/−7；没有满足两个阈值相对两项控制均不退化。原生REC相对起点0/+32，相对native_only +13/−7。不能说所有单项都变差，也不能将相对起点的严格阈值收益全部归给兼容损失。系统相对起点修复/破坏：@0.25=5/7，@0.50=32/16；相对控制=6/3和10/17。实例身份尚未独立分解，IoU转移不自动等于跨实例切换。

独立审计PASS：冻结参数、三个旧读出及归一化元数据不变，66份optimizer状态均2482步，来源与数据一致。receipt SHA `d3f7053fd6ba6d8dcf110ce5c8ac6b23cb8c07a8cb7a2a4023a730143292a00c`；independent_audit SHA `0b1790d34d63207236a1292d2644008179dc50a283e40e56ddef7b77e4cd3c96`；terminal_rows SHA `a7a65a3d1302ec4253ff54df1fd4ef3f77ddd369094cccdeaa6a0745b9161e08`。这些是主干已见场景的模块筛选数据，不是新场景正式结果。历史最好不变，总体目标未完成；本版本封存，不继续同机制扫参。
'''
addition+='\n### 20.118 用户请求的真实失败场景本地可视化（'+now+'）\n\n'
addition+='用户要求在`C:\\Users\\gb\\Desktop\\document`渲染三个数据集的一些失败定位案例。当前已实际完成ScanRefer、Nr3D各3例，存于`MCLN_3D_failure_visualizations_20260907`，含6张2890×1700 PNG、6个离线交互HTML、总览、真实50k点NPZ、原文及来源记录。Sr3D历史权重和预测框缓存缺失，尚未完成；已向用户询问备份路径，没有用其他模型或人工框替代。\n\n'
addition+='ScanRefer使用受保护V99的逐级诊断stage_rows，不冒充精确历史正式逐行输出；3例实际输入point SHA与forward记录一致。Nr3D使用平均E57的root_only Default缓存（4275/7899口径），不是正式4475/7899逐行分解。GT按原数据实现从实例min/max转换为center/size；每例IoU重算一致，Nr3D全部有效缓存候选也核对一致。无新GPU forward或优化更新。\n\n'
addition+='| 数据集 | 场景 | 目标 | 类型 | 实际IoU | Top16 oracle IoU |\n|---|---|---|---|---:|---:|\n'
for row in cases['cases']:
    addition+=f'| {row["dataset"]} | {row["scene_id"]} | {row["target_name"]} | {row["category"]} | {row["recomputed_iou"]:.6f} | {row["top16_oracle_iou"]:.6f} |\n'
addition+='\nstrict_overlap只在0.50阈值失败；top16_coverage不推断Full256缺框。静态图明确记录高处剖开/裁切和显示降采样，交互图保留全部50k点；所有PNG已解码检查，Chrome断网逐页验证通过，并实看PNG与交互截图。案例不是总体性能估计。可视化代码和小型来源/验证记录归档于`refine-logs/failure_visualizations_20260907_v1`；点云及图像在用户指定本地目录，不加入Git大文件。\n'
new=old+addition.encode('utf-8')
tracker=repo/'refine-logs/EXPERIMENT_TRACKER.md'
lines=tracker.read_text(encoding='utf-8').splitlines()
lines[2]='Updated: '+now+'. Sections20.117-118: frozen_gt completed and failed module REC screen; no new formal; six real ScanRefer/Nr3D visualizations delivered locally, Sr3D predictions missing.'
for i,line in enumerate(lines):
    if line.startswith('| Frozen protected readout compatibility |'):
        lines[i]='| Frozen protected readout compatibility | Completed2482/arm;terminal6887 and independent integrity audit PASS | Module REC screen FAIL: system vs baseline -2/+16,vs native_only +3/-7;no9508 formal or Nr/Sr training |'
tracker_raw=('\n'.join(lines)+'\n').encode()
archive=repo/'refine-logs/failure_visualizations_20260907_v1'
archive.mkdir(exist_ok=True)
names=['cases.json','README.md','export_source.py','render_source.py','render_verification.json','browser_verification.json']
for name in names: (archive/name).write_bytes((root/name).read_bytes())
(archive/'verify_browser.py').write_bytes(Path('C:/Users/gb/.codex/tmp/verify_mcln_failure_html_20260907.py').read_bytes())
(archive/'seal_terminal_and_visuals.py').write_bytes(Path(__file__).read_bytes())
c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
s=c.open_sftp()
with s.open(runtime+'docs/'+master.name,'rb') as f:
    f.prefetch(file_size=len(old));assert f.read()==old
for rel,raw in [('docs/'+master.name,new),('refine-logs/EXPERIMENT_TRACKER.md',tracker_raw)]:
    with s.open(runtime+rel,'wb') as f: f.set_pipelined(True);f.write(raw)
    with s.open(runtime+rel,'rb') as f: f.prefetch(file_size=len(raw));assert f.read()==raw
master.write_bytes(new);desktop.write_bytes(new);tracker.write_bytes(tracker_raw)
for name in names+['verify_browser.py','seal_terminal_and_visuals.py']:
    s.put(str(archive/name),'/root/autodl-tmp/mcln_failure_visualizations_20260907_v1/'+name)
s.close();c.close()
proof={'time_cst':now,'section':'20.118','master_sha256':hashlib.sha256(new).hexdigest(),'master_bytes':len(new),'three_master_copies_equal':True,'scanrefer_cases':3,'nr3d_cases':3,'sr3d_cases':0,'goal_complete':False,'new_formal_results':False}
(archive/'handoff_sync.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8')
print(json.dumps(proof),flush=True)
