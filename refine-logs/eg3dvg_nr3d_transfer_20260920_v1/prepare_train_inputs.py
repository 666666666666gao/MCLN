import datetime,hashlib,json,os,pickle,random,re,sys,time
from pathlib import Path
r=Path(__file__).resolve().parent;source=r/'source';data=Path('/root/autodl-tmp/DATA_ROOT_mcln_meshsp');parent=Path('/root/autodl-tmp/mcln_eg3dvg_acceptance_20260920_v1');out=r/'train_input_cache';out.mkdir()
os.chdir(str(source));sys.path.insert(0,str(source));sys.path.insert(1,str(source/'pointnet2'))
import numpy as np
from src.joint_det_dataset import Joint3DDataset,unpickle_data,read_label_mapping
random.seed(2027);np.random.seed(2027)
obj=Joint3DDataset.__new__(Joint3DDataset);obj.split='train';obj.overfit=False;obj.data_path=str(parent/'referit_input_cache/data_view')+'/'
obj.scans=list(unpickle_data(str(data/'train_v3scans.pkl')))[0]
obj.label_map=read_label_mapping('data/meta_data/scannetv2-labels.combined.tsv',label_from='raw_category',label_to='id')
report={'status':'complete','model_forwards':0,'training_steps':0,'scope':'unchanged author annotation loaders plus train input checks; not training','datasets':{}}
for name in ['nr3d','scannet']:
 t=time.time();print('TRAIN_CACHE_BEGIN '+name,flush=True);annos=obj.load_annos(name);path=out/(name+'_annotations.pkl')
 with path.open('xb') as f:pickle.dump(annos,f,protocol=4)
 scenes=sorted({a['scan_id'] for a in annos});assert all(s in obj.scans for s in scenes)
 assert all((data/'superpoints/train'/(s+'_superpoint.pth')).is_file() for s in scenes)
 record={'rows':len(annos),'scenes':len(scenes),'seconds':time.time()-t,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
 if name=='nr3d':
  assert all(0<=a['target_id']<len(obj.scans[a['scan_id']].three_d_objects) for a in annos)
  view_words={'front','behind','back','left','right','facing','leftmost','rightmost','looking','across'}
  disagreement=[]
  for i,a in enumerate(annos):
   old=Joint3DDataset._augment_nr3d(a['utterance']);normalized=not bool(set(re.findall('[a-z]+',a['utterance'].lower()))&view_words)
   if old!=normalized:disagreement.append({'row':i,'scan_id':a['scan_id'],'utterance':a['utterance'],'author_rotate_allowed':old,'normalized_same_words_allow':normalized})
  record['view_word_tokenization_disagreements']=len(disagreement);record['view_examples']=disagreement[:8]
  (out/'view_word_disagreements.json').write_text(json.dumps(disagreement,indent=2))
 report['datasets'][name]=record;print('TRAIN_CACHE_COMPLETE '+json.dumps(record),flush=True)
report['time_cst']=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
(out/'receipt.json').write_text(json.dumps(report,indent=2));print('EG_TRAIN_INPUTS_COMPLETE '+json.dumps(report),flush=True)
