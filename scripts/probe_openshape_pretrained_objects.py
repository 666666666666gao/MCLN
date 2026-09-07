"""Run the pinned official OpenShape encoder on uncleaned ScanRefer detector crops."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):
            h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True,type=Path)
    p.add_argument('--witness-only',action='store_true')
    args=p.parse_args()
    root=args.root
    import numpy as np
    import pkg_resources
    pkg_resources.require(['dgl-cu111==0.9.1.post1', 'torch.redstone==0.0.6',
                           'einops==0.6.1', 'huggingface-hub==0.4.0'])
    import torch
    import torch.nn.functional as F
    import dgl
    from openshape import G14
    torch.manual_seed(0);torch.cuda.manual_seed_all(0);np.random.seed(0)
    torch.set_num_threads(4)
    assert torch.__version__=='1.10.2+cu111'
    assert sha(root/'assets/model.pt')=='34949c162aca01b6fd3147ed7ccf34b448a34bdebc4a857605ea62412ad54fb9'
    model=G14(torch.load(root/'assets/model.pt',map_location='cpu')).cuda().eval().requires_grad_(False)
    x=torch.randn(1,6,1024,device='cuda')
    x[:,:3]/=x[:,:3].norm(dim=1).amax(dim=-1)[:,None,None]
    x[:,3:]=x[:,3:].sigmoid()
    torch.manual_seed(11)
    with torch.no_grad(): y=model(x)
    torch.cuda.synchronize()
    assert y.shape==(1,1280) and torch.isfinite(y).all() and y.norm()>0
    witness=dict(shape=list(y.shape),norm=float(y.norm()),device=torch.cuda.get_device_name(),torch=torch.__version__,dgl=dgl.__version__,parameters=sum(v.numel() for v in model.parameters()))
    print('OPENSHAPE_WITNESS',json.dumps(witness),flush=True)
    if args.witness_only:
        return
    export=json.loads((root/'inputs/export_receipt.json').read_text())
    categories=json.loads((root/'assets/lvis_categories.json').read_text())
    cats=F.normalize(torch.load(root/'assets/lvis_cats.pt',map_location='cpu').float(),dim=-1).cuda()
    assert cats.shape[1]==1280 and len(categories)<=len(cats)
    rows=[];embeddings=[];started=time.time()
    torch.cuda.reset_peak_memory_stats()
    for row in export['rows']:
        path=root/'inputs'/row['file'];assert sha(path)==row['file_sha256']
        data=np.load(str(path))
        for obj in row['objects']:
            entry=dict(row_id=row['row_id'],scan_id=row['scan_id'],**obj)
            # G14 requests 384 distinct FPS positions. This initial source test
            # records undersized crops as unavailable; it never fabricates points.
            if obj['points']<384:
                entry['encoded']=False;rows.append(entry);continue
            pc=data['object_%03d'%obj['slot']].copy()
            rng=np.random.RandomState(row['row_id']*256+obj['slot'])
            if len(pc)>10000:
                pc=pc[rng.choice(len(pc),10000,replace=False)]
            pc[:,:3]-=pc[:,:3].mean(0)
            radius=np.linalg.norm(pc[:,:3],axis=1).max()
            assert radius>0
            pc[:,:3]/=radius
            tensor=torch.from_numpy(pc.T.copy())[None].cuda()
            torch.manual_seed(row['row_id']*256+obj['slot'])
            with torch.no_grad():
                feature=model(tensor)
                similarity=F.normalize(feature,dim=-1)@cats[:len(categories)].T
            assert torch.isfinite(feature).all() and feature.norm()>0
            top=similarity[0].topk(5)
            entry.update(encoded=True,input_points=len(pc),feature_norm=float(feature.norm()),
                         top5_categories=[categories[i] for i in top.indices.tolist()],
                         top5_cosines=top.values.tolist())
            embeddings.append(feature[0].cpu().numpy());entry['embedding_index']=len(embeddings)-1
            rows.append(entry)
        print('OBJECT_SOURCE_ROW',row['row_id'],sum(r.get('encoded',False) for r in rows),flush=True)
    torch.cuda.synchronize()
    np.savez_compressed(str(root/'object_embeddings.npz'),features=np.asarray(embeddings,dtype=np.float32))
    receipt=dict(status='complete',witness=witness,objects=rows,encoded_objects=len(embeddings),
                 total_objects=len(rows),elapsed_seconds=time.time()-started,
                 gpu_peak_bytes=torch.cuda.max_memory_allocated(),export_receipt_sha256=sha(root/'inputs/export_receipt.json'),
                 embedding_file_sha256=sha(root/'object_embeddings.npz'),
                 formal_rows=0,optimizer_steps=0,new_rec_metrics=False,
                 note='LVIS top5 descriptions are diagnostics, not ScanRefer classification or REC accuracy.')
    with (root/'probe_receipt.json').open('x') as f:json.dump(receipt,f,indent=2,allow_nan=False)
    print('OBJECT_SOURCE_COMPLETE',len(embeddings),len(rows),receipt['elapsed_seconds'],flush=True)


if __name__=='__main__':
    main()
