import hashlib
import html
import itertools
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
from PIL import Image
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

ROOT = Path('C:/Users/gb/Desktop/document/MCLN_3D_failure_visualizations_20260907')
manifest = json.loads((ROOT / 'cases.json').read_text(encoding='utf-8'))
font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family': ['Microsoft YaHei', 'DejaVu Sans'], 'axes.unicode_minus': False})
BG, INK, MUTED, GREEN, RED = '#f5f7fa', '#152c44', '#5b6a7a', '#00a875', '#ec4262'
CATEGORIES = {
    'selection': ('有好候选，但最终未选中', 'Top-16 中存在高 IoU 候选；当前预测与目标几乎不重叠。'),
    'strict_overlap': ('严格定位失败', '当前框通过 Acc@0.25，但未通过 Acc@0.50。'),
    'top16_coverage': ('Top-16 覆盖失败', '缓存前 16 个候选均未达到 IoU 0.25；不能据此判断完整 256 个候选缺框。'),
}

def corners(box):
    return np.asarray(box[:3]) + np.array(list(itertools.product([-.5, .5], repeat=3))) * box[3:]

EDGES = [(i, j) for i in range(8) for j in range(i + 1, 8) if bin(i ^ j).count('1') == 1]

def draw_box(ax, box, color):
    c = corners(box)
    ax.add_collection3d(Line3DCollection([(c[i], c[j]) for i, j in EDGES], colors=color, linewidths=2.3, zorder=10))

def sample_indices(indices, limit):
    if len(indices) > limit:
        return indices[np.linspace(0, len(indices) - 1, limit).astype(int)]
    return indices

def view(ax, xyz, rgb, indices, gt, pred, lo, hi, elev, azim, title, size):
    ii = sample_indices(indices, 30000)
    ax.scatter(*xyz[ii].T, c=np.clip(rgb[ii] * 256 / 255, 0, 1), s=size, alpha=.82, linewidths=0, depthshade=False, rasterized=True)
    draw_box(ax, gt, GREEN)
    draw_box(ax, pred, RED)
    limits = np.maximum(hi - lo, .15)
    ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), zlim=(lo[2], hi[2]))
    ax.set_box_aspect(limits)
    ax.view_init(elev=elev, azim=azim)
    ax.set_proj_type('ortho')
    ax.set_facecolor(BG)
    ax.set_title(title, loc='left', fontsize=13, color=INK, pad=8)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.set_pane_color((.96, .97, .98, 0))
        axis._axinfo['grid'].update({'color': '#dce2e9', 'linewidth': .45})
        axis.line.set_color('#b9c4cf')
    ax.tick_params(labelsize=7, colors=MUTED, pad=0)
    ax.set_xlabel('X (m)', fontsize=8, color=MUTED, labelpad=0)
    ax.set_ylabel('Y (m)', fontsize=8, color=MUTED, labelpad=0)
    ax.set_zlabel('Z (m)', fontsize=8, color=MUTED, labelpad=0)

def interactive_box(box, color, name):
    c = corners(box)
    coords = [[v for i, j in EDGES for v in (float(c[i, axis]), float(c[j, axis]), None)] for axis in range(3)]
    return go.Scatter3d(x=coords[0], y=coords[1], z=coords[2], mode='lines', line={'color':color,'width':7}, name=name, hoverinfo='name')

(ROOT / 'assets').mkdir(exist_ok=True)
(ROOT / 'assets/plotly.min.js').write_text(get_plotlyjs(), encoding='utf-8')
cards = []
verified = []
for number, case in enumerate(manifest['cases'], 1):
    dataset_dir = ROOT / case['dataset']
    dataset_dir.mkdir(exist_ok=True)
    arrays = np.load(ROOT / 'data' / case['npz_file'])
    xyz, rgb = arrays['xyz'], arrays['rgb']
    gt, pred = arrays['gt_box'], arrays['pred_box']
    assert len(xyz) == case['scene_points'] and int(arrays['target_mask'].sum()) == case['target_points']
    cg, cp = corners(gt), corners(pred)
    boxlo = np.minimum(cg.min(0), cp.min(0))
    boxhi = np.maximum(cg.max(0), cp.max(0))
    locallo = boxlo - np.array([.4, .4, .25])
    localhi = boxhi + np.array([.4, .4, .25])
    inside = np.all((xyz >= locallo) & (xyz <= localhi), axis=1)
    zcut = min(float(xyz[:,2].max()), float(boxhi[2] + .55))
    whole = np.flatnonzero(xyz[:,2] <= zcut)
    local = np.flatnonzero(inside)
    assert len(local) > 0
    scenelo = np.minimum(xyz.min(0), boxlo) - .1
    scenehi = np.maximum(xyz.max(0), boxhi) + .1
    scenehi[2] = max(zcut, boxhi[2]) + .1
    category, explanation = CATEGORIES[case['category']]
    iou = case['recomputed_iou']
    fig = plt.figure(figsize=(17, 10), dpi=170, facecolor=BG)
    fig.text(.04, .955, f'{case["dataset"]}  /  {case["scene_id"]}', fontsize=23, weight='bold', color=INK)
    fig.text(.96, .958, f'{number:02d}   {category}', fontsize=13, color=MUTED, ha='right')
    fig.text(.04, .908, f'Target: {case["target_name"]}   ·   instance {case["target_id"]}   ·   row {case["row_id"]}', fontsize=12, color=MUTED)
    wrapped = textwrap.fill(case['description'], width=123)
    fig.text(.04, .863, wrapped, fontsize=12, color=INK, linespacing=1.55, va='top')
    ax1 = fig.add_axes([.025,.285,.45,.475], projection='3d')
    ax2 = fig.add_axes([.50,.30,.46,.46], projection='3d')
    view(ax1,xyz,rgb,whole,gt,pred,scenelo,scenehi,27,-58,'01   场景概览 / 高处剖开以露出目标',1.25)
    view(ax2,xyz,rgb,local,gt,pred,locallo,localhi,24,-58,'02   预测框与目标框 / 局部放大',3.2)
    # A metrically equal XY view makes displaced or oversized boxes unambiguous.
    ax3 = fig.add_axes([.055,.077,.235,.205])
    ii = sample_indices(local, 14000)
    ax3.scatter(xyz[ii,0],xyz[ii,1],c=np.clip(rgb[ii]*256/255,0,1),s=1.1,alpha=.7,linewidths=0,rasterized=True)
    for b,col in [(gt,GREEN),(pred,RED)]:
        ax3.add_patch(plt.Rectangle(b[:2]-b[3:5]/2,b[3],b[4],fill=False,color=col,lw=2))
    ax3.set(xlim=(locallo[0],localhi[0]),ylim=(locallo[1],localhi[1]))
    ax3.set_aspect('equal',adjustable='box')
    ax3.set_facecolor(BG)
    ax3.set_title('03   俯视 / XY 等比例',loc='left',fontsize=10,color=INK)
    ax3.tick_params(labelsize=7,colors=MUTED)
    for spine in ax3.spines.values(): spine.set_color('#cdd6df')
    fig.text(.345,.262,'━  GT 标注框',fontsize=13,color=GREEN,weight='bold')
    fig.text(.52,.262,'━  实际预测框',fontsize=13,color=RED,weight='bold')
    fig.text(.345,.214,f'3D IoU  {iou:.4f}',fontsize=23,color=INK,weight='bold')
    status25='通过' if iou>=.25 else '失败'
    fig.text(.345,.177,f'Acc@0.25：{status25}    |    Acc@0.50：失败',fontsize=12,color=RED)
    fig.text(.345,.142,f'Top-16 oracle IoU  {case["top16_oracle_iou"]:.4f}    ·    目标采样点 {case["target_points"]:,}',fontsize=11,color=MUTED)
    fig.text(.345,.104,textwrap.fill(explanation,width=48),fontsize=10,color=MUTED,linespacing=1.5)
    source = 'ScanRefer: protected V99 / 逐级诊断输出' if case['dataset']=='ScanRefer' else 'Nr3D: averaged E57 / root_only Default 缓存诊断'
    fig.text(.04,.032,source+'   |   原始 RGB 点云；坐标单位 m；图例用于诊断，不代表总体准确率。',fontsize=9,color=MUTED)
    png = dataset_dir / (case['case_id']+'.png')
    fig.savefig(png,facecolor=BG,dpi=170)
    plt.close(fig)
    with Image.open(png) as im:
        im.verify()
    with Image.open(png) as im:
        assert im.size==(2890,1700)
    colors=['rgb(%d,%d,%d)' % tuple(c) for c in np.clip(np.round(rgb*256),0,255).astype(np.uint8)]
    fig3 = go.Figure()
    coordinates=np.round(xyz.astype(float),4)
    fig3.add_trace(go.Scatter3d(x=coordinates[:,0].tolist(),y=coordinates[:,1].tolist(),z=coordinates[:,2].tolist(),mode='markers',marker={'size':1.5,'color':colors,'opacity':.8},name='真实 RGB 点云（50,000 点）',hoverinfo='skip'))
    fig3.add_trace(interactive_box(gt,GREEN,'GT 标注框'))
    fig3.add_trace(interactive_box(pred,RED,'实际预测框'))
    target=coordinates[arrays['target_mask']]
    fig3.add_trace(go.Scatter3d(x=target[:,0].tolist(),y=target[:,1].tolist(),z=target[:,2].tolist(),mode='markers',marker={'size':3,'color':GREEN},name='GT 实例点（点击显示）',visible='legendonly',hoverinfo='skip'))
    fig3.update_layout(template='plotly_white',margin={'l':0,'r':0,'t':15,'b':0},scene={'aspectmode':'data','xaxis_title':'X (m)','yaxis_title':'Y (m)','zaxis_title':'Z (m)','camera':{'eye':{'x':1.3,'y':-1.6,'z':1.0}}},legend={'orientation':'h','y':1.03},height=760,paper_bgcolor=BG)
    plot=fig3.to_html(full_html=False,include_plotlyjs='../assets/plotly.min.js',config={'responsive':True,'displaylogo':False,'toImageButtonOptions':{'filename':case['case_id'],'format':'png','scale':2}})
    casehtml=dataset_dir / (case['case_id']+'.html')
    casehtml.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+html.escape(case['case_id'])+'</title><style>body{margin:24px auto;max-width:1400px;background:#f5f7fa;color:#152c44;font:16px/1.65 "Microsoft YaHei",sans-serif}a{color:#126c96}.meta{color:#5b6a7a}h1{font-size:26px}blockquote{margin:16px 0;padding:16px 24px;background:white;border-left:4px solid #00a875}</style><a href="../index.html">← 返回案例总览</a><h1>'+html.escape(case['dataset']+' / '+case['scene_id']+' — '+category)+'</h1><blockquote>'+html.escape(case['description'])+'</blockquote><p>绿色：GT；红色：真实预测。3D IoU <b>'+f'{iou:.4f}'+'</b>；Top-16 oracle IoU '+f'{case["top16_oracle_iou"]:.4f}'+'。拖动旋转、滚轮缩放；点击图例可隐藏点云查看框，或显示 GT 实例点。</p>'+plot+'<p class="meta">交互视图保留全部 50,000 个输入点。静态图为了可见性，概览仅显示 Z ≤ '+f'{zcut:.3f}'+' m 的点，局部图裁切到两个框的联合邻域；以上处理不改变预测、GT 或 IoU。坐标为原始轴对齐场景坐标。</p><p class="meta">来源：'+html.escape(case['prediction_version'])+'。此记录用于错误诊断，不能等同于历史正式 evaluator 的逐行结果。GT 实例点仅用于可视化标注。</p><a href="'+png.name+'">查看高清 PNG</a></html>',encoding='utf-8')
    relative=png.relative_to(ROOT).as_posix()
    interactive=casehtml.relative_to(ROOT).as_posix()
    cards.append('<article><a href="'+interactive+'"><img src="'+relative+'" alt="'+html.escape(case['case_id'])+'"></a><div class="body"><span class="tag">'+case['dataset']+'</span><h2>'+html.escape(case['scene_id']+' · '+case['target_name'])+'</h2><p>'+category+' · IoU <b>'+f'{iou:.4f}'+'</b></p><p class="description">'+html.escape(case['description'])+'</p><a class="button" href="'+interactive+'">旋转查看 3D 场景</a> <a href="'+relative+'">高清 PNG</a></div></article>')
    case['rendering']={'overview_z_cut_m':zcut,'overview_displayed_points':min(len(whole),30000),'local_displayed_points':min(len(local),30000),'point_selection_for_display':'uniform index subsampling after spatial crop; no coordinate deformation','interactive_scene_points':len(xyz),'png':relative,'interactive_html':interactive,'local_bounds':[locallo.tolist(),localhi.tolist()]}
    verified.append({'file':relative,'bytes':png.stat().st_size,'sha256':hashlib.sha256(png.read_bytes()).hexdigest(),'dimensions':[2890,1700]})
    print('RENDERED',relative,flush=True)

page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MCLN · 真实 3D 定位失败案例</title><style>*{box-sizing:border-box}body{margin:0;background:#edf2f7;color:#152c44;font:16px/1.7 "Microsoft YaHei",sans-serif}main{max-width:1450px;margin:auto;padding:48px 28px}header{margin-bottom:30px}h1{font-size:36px;letter-spacing:-1px;margin:5px 0}.kicker{font-size:13px;letter-spacing:2px;color:#648097}.lead{max-width:1080px;color:#4e6478}.status{background:#fff4d9;padding:18px 22px;border-radius:12px;margin:24px 0}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px}article{background:white;border-radius:16px;overflow:hidden;box-shadow:0 5px 25px #152c4408}article img{width:100%;display:block}.body{padding:22px}h2{font-size:21px;margin:9px 0}.tag{color:#126c96;font-size:13px;font-weight:bold}.description{color:#607183;min-height:55px}a{color:#126c96}.button{display:inline-block;background:#153953;color:white;padding:8px 14px;border-radius:7px;text-decoration:none;margin-right:15px}footer{margin:32px 0;color:#5b6a7a}@media(max-width:850px){.grid{grid-template-columns:1fr}main{padding:24px 14px}h1{font-size:27px}}</style><main><header><div class="kicker">MCLN / REAL SCENE DIAGNOSTICS / 2026.09.07</div><h1>真实 3D 定位失败案例</h1><p class="lead">ScanRefer 与 Nr3D 各 3 个不同场景。绿色框为目标 GT，红色框为已有实验的真实预测。每例包含高清静态图、可离线旋转的完整点云、原始描述与可核对的数据记录。</p><div class="status"><b>Sr3D 尚未补齐：</b>当前服务器与已检查的备份中没有找到历史 Sr3D 权重或带预测框的缓存。为保证案例真实，未用其他模型输出替代。待取回权重或预测记录后补充。</div><p class="lead">ScanRefer 使用受保护 V99 的逐级诊断记录；Nr3D 使用平均 E57 的 root_only Default 候选缓存。两者均为诊断案例，不能用于拼接正式成绩。严格定位案例的 IoU 在 0.25 与 0.50 之间，仅在 0.50 阈值下失败。</p></header><div class="grid">'''+''.join(cards)+'''</div><footer>所有图都来自真实输入点云与已有预测，不使用生成式图片。静态概览采用剖开与裁切以露出目标；交互页保留完整 50,000 点。六个案例不代表全体错误的比例。<br>数据与来源：<a href="cases.json">cases.json</a> · <a href="README.md">说明与复核</a></footer></main></html>'''
(ROOT/'index.html').write_text(page,encoding='utf-8')
(ROOT/'cases.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(ROOT/'render_verification.json').write_text(json.dumps({'decoded_pngs':verified,'cases':len(verified),'synthetic_predictions':False,'sr3d_complete':False},indent=2)+'\n',encoding='utf-8')
table=['| 数据集 | 场景 | 目标 | 错误类型 | IoU | Top-16 oracle IoU |','|---|---|---|---|---:|---:|']
for c in manifest['cases']:
    table.append(f'| {c["dataset"]} | {c["scene_id"]} | {c["target_name"]} | {CATEGORIES[c["category"]][0]} | {c["recomputed_iou"]:.4f} | {c["top16_oracle_iou"]:.4f} |')
readme='''# MCLN 真实定位失败案例

双击 `index.html` 查看总览。进入案例 HTML 可旋转、缩放点云；点击图例可隐藏点云或显示目标实例点。所有网页依赖均保存在本目录，离线可用。

- ScanRefer：3 个不同场景，使用受保护 E71 + Parent + Geometry + V99 的 2026-09-07 逐级诊断输出。
- Nr3D：3 个不同场景，使用受保护平均 E57 的历史 root_only Default 候选缓存；该缓存总体为 4275/7899，不等同于正式 4475/7899 输出。
- Sr3D：权重及预测框缓存暂缺，尚未完成。没有使用其他模型或手工框冒充预测。

绿色：GT；红色：真实预测。静态图每例 2890×1700，包括场景概览、局部 3D 与 XY 俯视。概览剖开高处点云，局部图裁切两个框的联合邻域；按点索引均匀降采样仅用于显示。交互页保留完整 50,000 个真实输入点。坐标单位米，三轴按真实比例展示。

'''+ '\n'.join(table)+'''

## 真实性与边界

预测框均来自留存实验输出；GT 从对应采样场景的目标实例重建，按数据集实现由 min/max 转成中心和尺寸。所有案例重算 IoU 与缓存一致。ScanRefer 的网络输入点云 SHA-256 与当时 forward 记录逐例一致；Nr3D 缓存没有保存当时点云哈希，所以通过全部有效候选的 IoU 重算核对几何对应，并未声称有原始输入哈希证明。

`selection` 仅说明 Top-16 有好候选但未选中，不自动断言候选实例身份。`strict_overlap` 通过 0.25、未过 0.50。`top16_coverage` 仅表示前 16 个候选未覆盖，不能推断完整 256 个 Query 都没有好框。GT 目标点仅作为可视化标注，没有用于生成模型预测。

文件：各数据集子目录内为 PNG/HTML；`data/*.npz` 保存实际 XYZ/RGB、GT 实例点标签与两个框；`cases.json` 保存原始描述、行号、IoU、数据来源与 SHA-256；`render_verification.json` 保存高清图解码与文件校验结果。`export_source.py`、`render_source.py` 保存使用的脚本。
'''
(ROOT/'README.md').write_text(readme,encoding='utf-8')
(ROOT/'render_source.py').write_bytes(Path(__file__).read_bytes())
# A compact contact sheet provides a single file that can be opened immediately.
thumbs=[]
for entry in verified:
    with Image.open(ROOT/entry['file']) as im:
        thumb=im.convert('RGB');thumb.thumbnail((1156,680));thumbs.append(thumb.copy())
sheet=Image.new('RGB',(2312,2040),BG)
for i,im in enumerate(thumbs): sheet.paste(im,((i%2)*1156,(i//2)*680))
sheet.save(ROOT/'全部案例总览.jpg',quality=94)
print('RENDER_COMPLETE',len(verified),flush=True)
