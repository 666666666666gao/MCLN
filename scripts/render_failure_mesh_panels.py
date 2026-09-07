"""Render existing, verified predictions on original ScanNet colored triangles."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pyvista as pv
from scipy.spatial import cKDTree

parser=argparse.ArgumentParser()
parser.add_argument('--limit',type=int,default=6)
options=parser.parse_args()
root=Path('C:/Users/gb/Desktop/document/MCLN_3D_failure_visualizations_20260907')
out=root/'paper_style_dense'
manifest=json.loads((root/'cases.json').read_bytes())
alignments=json.loads((out/'scans_axis_alignment_matrices.json').read_bytes())
sources=json.loads((out/'mesh_sources.json').read_bytes())
for item in sources['files']:
    assert hashlib.sha256((out/item['local_file']).read_bytes()).hexdigest()==item['sha256']
font_path='C:/Windows/Fonts/arial.ttf'
bold_path='C:/Windows/Fonts/arialbd.ttf'
cn_path='C:/Windows/Fonts/msyh.ttc'
def font(size,bold=False):return ImageFont.truetype(bold_path if bold else font_path,size)
GREEN=(0,181,79);RED=(235,32,39);INK=(17,22,28)
W,H=1120,1050
edges=[(i,j) for i in range(8) for j in range(i+1,8) if bin(i^j).count('1')==1]
def corners(box):
    return np.asarray(box[:3])+np.asarray(list(itertools.product([-.5,.5],repeat=3)))*box[3:]

def projected(renderer,points):
    result=[]
    for p in points:
        renderer.SetWorldPoint(float(p[0]),float(p[1]),float(p[2]),1)
        renderer.WorldToDisplay();x,y,z=renderer.GetDisplayPoint()
        result.append((x,H-y))
    return result

def overlay(background,pts,color):
    image=background.copy();draw=ImageDraw.Draw(image)
    for i,j in edges:
        draw.line([pts[i],pts[j]],fill='white',width=10)
    for i,j in edges:
        draw.line([pts[i],pts[j]],fill=color,width=6)
    return image

def draw_wrapped(draw,text,xy,width,font_value,fill=INK,line_height=40):
    x,y=xy;line=''
    for word in text.split():
        trial=(line+' '+word).strip()
        if draw.textlength(trial,font=font_value)>width and line:
            draw.text((x,y),line,font=font_value,fill=fill);y+=line_height;line=word
        else:line=trial
    draw.text((x,y),line,font=font_value,fill=fill)
    return y+line_height

rows=[]
for number,case in enumerate(manifest['cases'][:options.limit]):
    name=case['scene_id']
    mesh=pv.read(out/'meshes'/(name+'_vh_clean_2.ply'))
    raw=np.asarray(mesh.points).copy()
    matrix=np.asarray(alignments[name]).reshape(4,4)
    xyz=np.c_[raw,np.ones(len(raw))]@matrix.T
    xyz=xyz[:,:3]
    data=np.load(root/'data'/case['npz_file'])
    distances,_=cKDTree(xyz).query(data['xyz'],workers=2)
    assert distances.max()<2e-6,(name,float(distances.max()))
    mesh.points=xyz
    gt=np.asarray(case['gt_box']);pred=np.asarray(case['pred_box'])
    support=np.vstack([corners(gt),corners(pred)])
    # A horizontal cut opens the room for viewing; retain every triangle below it.
    # Full uncut source mesh remains in meshes/ and every box is unchanged.
    zcut=min(xyz[:,2].max(),max(support[:,2].max()+.35,xyz[:,2].min()+1.45))
    visible=mesh.clip(normal=(0,0,1),origin=(0,0,zcut),invert=True)
    colors=[k for k in visible.point_data if visible.point_data[k].ndim==2 and visible.point_data[k].shape[1] in (3,4)]
    assert colors,visible.point_data.keys()
    color_name='RGB' if 'RGB' in colors else colors[0]
    if visible.point_data[color_name].shape[1]==4:
        visible.point_data['display_rgb']=visible.point_data[color_name][:,:3];color_name='display_rgb'
    # Fixed physical orthographic camera: principal room direction vertical,
    # near-overhead angle exposes furniture while preserving height cues.
    cloud=np.vstack([visible.points,support]);center=(cloud.min(0)+cloud.max(0))/2
    covariance=np.cov(xyz[:,:2].T)
    _,basis=np.linalg.eigh(covariance);long_axis=basis[:,-1]
    if long_axis[1]<0:long_axis=-long_axis
    horizontal=np.array([long_axis[1],-long_axis[0],0.])
    altitude=np.deg2rad(68)
    direction=horizontal*np.cos(altitude)+np.array([0,0,np.sin(altitude)])
    up=np.array([long_axis[0],long_axis[1],0.])
    right=np.cross(-direction,up);right/=np.linalg.norm(right)
    true_up=np.cross(right,-direction)
    relative=cloud-center
    scale=max(np.ptp(relative@true_up)/2,np.ptp(relative@right)/2*H/W)*1.035
    plot=pv.Plotter(off_screen=True,window_size=(W,H))
    plot.set_background('white')
    plot.add_mesh(visible,scalars=color_name,rgb=True,lighting=False,show_scalar_bar=False)
    plot.camera.position=center+direction*max(np.ptp(cloud,axis=0))*4
    plot.camera.focal_point=center
    plot.camera.up=up
    plot.camera.parallel_projection=True
    plot.camera.parallel_scale=float(scale)
    plot.enable_anti_aliasing('ssaa')
    plot.render()
    background=Image.fromarray(plot.screenshot(return_img=True)).convert('RGB')
    gt_xy=projected(plot.renderer,corners(gt));pred_xy=projected(plot.renderer,corners(pred))
    assert all(0<=x<W and 0<=y<H for x,y in gt_xy+pred_xy)
    gt_image=overlay(background,gt_xy,GREEN)
    pred_image=overlay(background,pred_xy,RED)
    panel=Image.new('RGB',(W+170,H*2+650),'white')
    panel.paste(gt_image,(150,70));panel.paste(pred_image,(150,H+90))
    draw=ImageDraw.Draw(panel)
    draw.text((24,520),'GT',font=font(45,True),fill=GREEN)
    draw.text((24,H+540),'Ours',font=font(40,True),fill=RED)
    draw.text((150,16),case['dataset']+' / '+name+' / '+case['target_name'],font=font(26,True),fill=INK)
    draw.text((150,H*2+145),'3D IoU = {:.4f}'.format(case['recomputed_iou']),font=font(34,True),fill=RED)
    last_y=draw_wrapped(draw,case['description'],(150,H*2+205),W-40,font(29),line_height=40)
    panel=panel.crop((0,0,panel.width,last_y+40))
    panel.save(out/(case['case_id']+'_GT_Ours.png'),dpi=(300,300))
    gt_image.save(out/(case['case_id']+'_GT.png'))
    pred_image.save(out/(case['case_id']+'_Ours.png'))
    background.save(out/(case['case_id']+'_scene.png'))
    rows.append({'case_id':case['case_id'],'dataset':case['dataset'],'scene_id':name,
        'vertices':mesh.n_points,'triangles':mesh.n_cells,'rendered_cells_after_cut':visible.n_cells,
        'alignment_max_nearest_point_error_m':float(distances.max()),'z_cut_m':float(zcut),
        'camera_position':list(plot.camera.position),'camera_focus':list(plot.camera.focal_point),
        'camera_up':list(plot.camera.up),'parallel_scale':float(scale),
        'gt_projected_pixels':gt_xy,'prediction_projected_pixels':pred_xy,
        'prediction_and_gt_boxes_unchanged':True,'gt_prediction_background_identical':True})
    plot.close();print('DENSE_RENDER',case['case_id'],mesh.n_points,mesh.n_cells,flush=True)

if options.limit==6:
    composites=[]
    for dataset in ('ScanRefer','Nr3D'):
        selected=[c for c in manifest['cases'] if c['dataset']==dataset]
        canvas=Image.new('RGB',(W*3+200,H*2+680),'white');draw=ImageDraw.Draw(canvas)
        draw.text((155,25),dataset+'  |  Grounding failure cases',font=font(42,True),fill=INK)
        draw.text((24,590),'GT',font=font(45,True),fill=GREEN)
        draw.text((24,H+610),'Ours',font=font(39,True),fill=RED)
        for i,case in enumerate(selected):
            x=155+i*W
            draw.text((x+35,100),'({})  {}'.format('abc'[i],case['scene_id']),font=font(29,True),fill=INK)
            canvas.paste(Image.open(out/(case['case_id']+'_GT.png')),(x,150))
            canvas.paste(Image.open(out/(case['case_id']+'_Ours.png')),(x,H+170))
            y=H*2+190
            draw.text((x+35,y),'({})  {}   |   IoU {:.4f}'.format('abc'[i],case['target_name'],case['recomputed_iou']),font=font(30,True),fill=RED)
            end=draw_wrapped(draw,case['description'],(x+35,y+57),W-75,font(31),line_height=42)
            assert end<canvas.height-80,(case['case_id'],end)
        draw.text((155,canvas.height-62),'Green: ground truth   |   Red: saved model prediction   |   Same scene and camera in both rows',font=font(25),fill=(75,75,75))
        filename=dataset+'_GT_Ours_dense.png'
        canvas.save(out/filename,dpi=(300,300));canvas.save(out/(dataset+'_GT_Ours_dense.pdf'),resolution=300)
        composites.append(canvas)
    composites[0].save(out/'ScanRefer_Nr3D_GT_Ours_dense.pdf',save_all=True,append_images=composites[1:],resolution=300)

(out/'dense_render_verification.json').write_text(json.dumps({'cases':rows,'style':'GT row above Ours row; exact shared mesh, crop and orthographic camera; no axes; colored triangle surfaces; white-halo box lines.',
    'mesh_interpolation':'ScanNet original triangle connectivity; no generated surfaces or duplicated points.',
    'new_predictions':False,'rendering_only_uses_dense_mesh':True,'sr3d_complete':False},indent=2)+'\n',encoding='utf-8')
