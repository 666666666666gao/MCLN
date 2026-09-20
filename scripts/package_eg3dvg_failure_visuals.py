"""Package and verify the local dense scene render artifacts."""
import argparse
import hashlib
import html
import json
from pathlib import Path
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, required=True)
args = parser.parse_args()
root = args.root
out = root / 'paper_style_dense'
manifest = json.loads((root / 'cases.json').read_text(encoding='utf-8'))
render = json.loads((out / 'dense_render_verification.json').read_text(encoding='utf-8'))
datasets = list(dict.fromkeys(c['dataset'] for c in manifest['cases']))
case_count = len(manifest['cases'])
assert len(render['cases']) == case_count == 3 * len(datasets)
combined_pdf = '_'.join(datasets) + '_GT_EG_dense.pdf'
assert (out / combined_pdf).is_file()
labels = {'selection': '有合格候选，但未选中', 'strict_overlap': '通过 0.25，未通过 0.50', 'full256_coverage': '全部 256 框均未达到 0.50'}
parts = ['''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>EG-3DVG 真实失败场景</title><style>body{margin:32px auto;max-width:1250px;padding:0 20px;font:17px/1.7 system-ui;background:#f5f6f8;color:#18202a}h1{font-size:30px}section,article{background:white;border-radius:12px;padding:24px;margin:20px 0}img{max-width:100%;height:auto}a{color:#175eb1}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:20px}.tag{font-weight:650;color:#9d2727}small{color:#56616e}</style>
<h1>EG-3DVG：真实失败场景的稠密网格对照</h1>
<p>上排绿色框为 GT，下排红色框为 EG 预测。使用 ScanNet 原始彩色三角网格，两排保持同一场景、相机与剖切范围。点击图片查看原尺寸。</p>
<p><strong>模型来源：</strong>作者 ScanRefer epoch69 完整预训练权重，零更新评估，原生 last/bbs 输出。Nr3D／Sr3D 案例属于该 ScanRefer 权重的跨数据集起点，不是对应数据集训练终点，也不是历史保护模型的预测。</p>
<p>本页只展示来源记录中已完成正式评估的数据集。案例用于解释失败类型，不代表随机样本或整体准确率。Oracle 只用于离线诊断。</p>''']
verified = []
for dataset in datasets:
    name = dataset + '_GT_EG_dense.png'
    href = 'paper_style_dense/' + name
    parts.append('<section><h2>' + dataset + '</h2><a href="' + href + '"><img src="' + href + '" alt="' + dataset + ' GT 与 EG 稠密网格对照"></a><p><a href="paper_style_dense/' + dataset + '_GT_EG_dense.pdf">下载本页 PDF</a></p></section>')
parts.append('<div class="grid">')
for case in manifest['cases']:
    href = 'paper_style_dense/' + case['case_id'] + '_GT_EG.png'
    description = case['description'].rsplit(' . not mentioned', 1)[0]
    parts.append('<article><h3>' + html.escape(case['dataset'] + ' / ' + case['scene_id']) + '</h3><p class="tag">' + labels[case['category']] + '</p><p>目标：' + html.escape(case['target_name']) + '<br>预测 IoU：{:.4f}<br>256 框上界 IoU：{:.4f}</p>'.format(case['recomputed_iou'], case['full256_native_oracle_iou']) + '<p>' + html.escape(description) + '</p><a href="' + href + '"><img src="' + href + '" alt="单例 GT 与 EG 框对照"></a></article>')
parts.append('</div><p><a href="paper_style_dense/' + combined_pdf + '">下载完整 PDF</a> · <a href="cases.json">预测与输入来源记录</a></p><small>图中文字仅去掉模型追加的末尾 “. not mentioned”；完整实际输入保存在 cases.json。房间水平剖切仅用于显示，原始网格、GT 和预测框保持不变。</small></html>')
(root / 'index.html').write_text('\n'.join(parts), encoding='utf-8')
for path in sorted(out.glob('*.png')):
    with Image.open(path) as image:
        size = image.size
        image.verify()
    verified.append({'file': str(path.relative_to(root)), 'size': size, 'bytes': path.stat().st_size,
                     'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
assert len(verified) == 4 * case_count + len(datasets)
for case, record in zip(manifest['cases'], render['cases']):
    assert case['case_id'] == record['case_id']
    assert record['alignment_max_nearest_point_error_m'] < 2e-6
    assert record['prediction_and_gt_boxes_unchanged'] and record['gt_prediction_background_identical']
result = {'status': 'complete', 'cases': case_count, 'datasets': datasets,
          'sr3d_complete': 'Sr3D' in datasets, 'images': verified, 'new_model_forwards': 0,
          'caption_transform': 'Remove only final appended . not mentioned for display; full input retained.',
          'browser_tested': False, 'manual_visual_inspection': 'pending'}
(root / 'delivery_verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
(root / 'README.md').write_text('''# EG-3DVG 稠密失败场景

打开 index.html 查看本目录来源记录中的各数据集，每个数据集三个不同场景。使用原始彩色三角网格，GT/EG 两排共享相机与裁切，绿色 GT、红色真实预测，白色描边便于区分。PNG 和完整 PDF 位于 paper_style_dense。

这是作者 ScanRefer epoch69 原权重的完整零更新评估输出；Nr3D／Sr3D 案例属于跨数据集起点，不是各自适配终点。不以其他模型替代或虚构案例，历史保护模型可视化目录未修改。

所有案例均核对实际模型输入点云 SHA、场景实例 GT、预测框和 IoU；mesh 按作者坐标对齐后与输入采样点最大最近距离小于 2e-6 米。完整来源、选择规则、框、哈希见 cases.json；渲染相机、剖切高度和像素位置见 paper_style_dense/dense_render_verification.json。候选上界只作 GT 离线诊断。案例经过显示筛选，不是总体指标估计。

图片标题 EG 指完整作者基线，并不表示本项目新增方法。未调整模型预测、评分或正式协议。当前 HTML 尚未进行浏览器交互测试。
''', encoding='utf-8')
print('EG_VISUAL_PACKAGE_VERIFIED ' + str(root))
