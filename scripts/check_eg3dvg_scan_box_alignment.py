"""Check actual GroupFree augmentation against a separate center/extent transform."""
import json
import os
from pathlib import Path
import pickle
import sys


def main():
    root = Path(__file__).resolve().parent
    source = root / 'source'
    os.chdir(source)
    sys.path.insert(0, str(source))
    sys.path.insert(1, str(source / 'pointnet2'))
    import numpy as np
    from src.joint_det_dataset import Joint3DDataset, read_label_mapping, rot_x, rot_y, rot_z
    with (root / 'scanrefer_annotations.pkl').open('rb') as f:
        scenes = sorted({a['scan_id'] for a in pickle.load(f)})[:16]
    loader = Joint3DDataset.__new__(Joint3DDataset)
    loader.data_path = '/root/autodl-tmp/DATA_ROOT_mcln_meshsp/'
    loader.butd, loader.butd_cls = True, False
    loader.split, loader.augment_det = 'train', False
    loader.label_map = read_label_mapping('data/meta_data/scannetv2-labels.combined.tsv', label_from='raw_category', label_to='id')
    maximum = 0.
    old_displacement = 0.
    count = 0
    for index, scene in enumerate(scenes):
        a = dict(yz_flip=index % 2 == 0, xz_flip=index % 3 == 0,
                 theta_z=31., theta_x=-1.3, theta_y=2.1, shift=np.array([[.17,-.23,.09]]), scale=1.01)
        loader.augment = False
        original, valid, _, _ = loader._get_detected_objects('train', scene, a)
        loader.augment = True
        actual, valid2, _, _ = loader._get_detected_objects('train', scene, a)
        basis = np.eye(3)
        if a['yz_flip']: basis[:,0] *= -1
        if a['xz_flip']: basis[:,1] *= -1
        basis = rot_y(rot_x(rot_z(basis,a['theta_z']),a['theta_x']),a['theta_y'])
        expected = np.concatenate(((original[:,:3] @ basis + a['shift']) * a['scale'],
                                    original[:,3:] @ np.abs(basis) * a['scale']),axis=1)
        maximum = max(maximum,float(np.abs(actual[valid]-expected[valid]).max()))
        old = rot_y(rot_x(rot_z(original[:,:3],a['theta_z']),a['theta_x']),a['theta_y'])
        if a['yz_flip']: old[:,0] *= -1
        if a['xz_flip']: old[:,1] *= -1
        old = (old + a['shift']) * a['scale']
        old_displacement = max(old_displacement,float(np.linalg.norm(old[valid]-expected[valid,:3],axis=1).max()))
        assert np.array_equal(valid,valid2)
        count += int(valid.sum())
    assert maximum < 1e-5 and old_displacement > .01
    report=dict(status='pass',scenes=len(scenes),boxes=count,maximum_corrected_component_error=maximum,
                old_order_maximum_center_displacement=old_displacement,model_forwards=0,optimizer_steps=0,
                annotation='Same corrected input path is used for both task and native training.')
    (root/'box_alignment.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
