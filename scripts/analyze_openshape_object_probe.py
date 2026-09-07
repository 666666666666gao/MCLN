"""Describe source coverage and repeated-scene feature stability, not REC accuracy."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    args=p.parse_args()
    import numpy as np
    r=json.loads((args.root/'probe_receipt.json').read_text())
    x=np.load(str(args.root/'object_embeddings.npz'))['features']
    assert x.shape==(r['encoded_objects'],1280) and np.isfinite(x).all()
    seen={};pairs=[];classes={}
    for row in r['objects']:
        key=(row['scan_id'],row['slot'])
        group=classes.setdefault(row['detector_class'],dict(occurrences=0,encoded=0))
        group['occurrences']+=1;group['encoded']+=int(row['encoded'])
        if row['encoded']:
            feature=x[row['embedding_index']]
            if key in seen:
                reference=seen[key]
                cosine=float(feature@reference/(np.linalg.norm(feature)*np.linalg.norm(reference)))
                pairs.append(dict(scan_id=key[0],slot=key[1],cosine=cosine))
            else:
                seen[key]=feature
    v=np.array([row['cosine'] for row in pairs])
    result=dict(occurrences=len(r['objects']),encoded_occurrences=len(x),
                unique_encoded_objects=len(seen),repeated_object_pairs=len(pairs),
                repeated_object_cosine_min=float(v.min()),repeated_object_cosine_median=float(np.median(v)),
                repeated_object_cosine_mean=float(v.mean()),by_detector_class=classes,
                elapsed_seconds=r['elapsed_seconds'],gpu_peak_bytes=r['gpu_peak_bytes'],
                interpretation='Repeated objects use different fixed row/slot sampling seeds. Cosines measure sampling stability, not classification or REC accuracy.',
                new_rec_metrics=False)
    (args.root/'probe_analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='by_detector_class'}))


if __name__=='__main__':
    main()
