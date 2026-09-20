import ast,hashlib,json,os,pickle,random,sys,time
from pathlib import Path
r=Path(__file__).resolve().parent;source=r/'source';data=Path('/root/autodl-tmp/DATA_ROOT_mcln_meshsp');out=r/'referit_input_cache';out.mkdir();view=out/'data_view';view.mkdir();(view/'ReferIt3D').symlink_to(data/'refer_it_3d',target_is_directory=True)
os.chdir(str(source));sys.path.insert(0,str(source));sys.path.insert(1,str(source/'pointnet2'))
import numpy as np
from src.joint_det_dataset import Joint3DDataset,unpickle_data
random.seed(2027);np.random.seed(2027)
scans=list(unpickle_data(str(data/'val_v3scans.pkl')))[0]
obj=Joint3DDataset.__new__(Joint3DDataset);obj.split='val';obj.overfit=False;obj.data_path=str(view)+'/';obj.scans=scans
report={'status':'complete','model_forwards':0,'optimizer_steps':0,'scope':'unchanged author annotation loaders and Scene_graph_parse only; not full Dataset/model acceptance','seed':2027,'datasets':{}}
for name,n in [('nr3d',7899),('sr3d',17726)]:
 t=time.time();print('PREPARE_BEGIN '+name,flush=True);annos=obj.load_annos(name);assert len(annos)==n
 assert all(a['dataset']==name for a in annos)
 path=out/(name+'_annotations.pkl')
 with path.open('xb') as f:pickle.dump(annos,f,protocol=4)
 report['datasets'][name]={'rows':len(annos),'scenes':len({a['scan_id'] for a in annos}),'seconds':time.time()-t,'cache_path':str(path),'cache_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'annotation_fields':sorted(annos[0])}
 print('PREPARE_COMPLETE '+json.dumps(report['datasets'][name]),flush=True)
(out/'receipt.json').write_text(json.dumps(report,indent=2));print('REFERIT_ANNOTATIONS_COMPLETE '+json.dumps(report),flush=True)
