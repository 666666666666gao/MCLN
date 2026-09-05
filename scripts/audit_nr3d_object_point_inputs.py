"""CPU-only availability audit for object appearance from existing inputs."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time


def file_sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    options = parser.parse_args()
    addon = options.manifest.parent
    manifest = json.loads(options.manifest.read_text())
    source = Path(manifest['model_source'])

    def verify_inputs():
        assert file_sha(source / 'g0_source_manifest.json') == manifest['source_manifest_sha256']
        for name, digest in json.loads((source / 'g0_source_manifest.json').read_text())['files'].items():
            assert file_sha(source / name) == digest, name
        for name, digest in manifest['files'].items():
            assert file_sha(addon / name) == digest, name
        for name, value in manifest['data_files'].items():
            assert file_sha(Path(name)) == value['sha256'], name
        assert file_sha(Path(manifest['m4_receipt'])) == manifest['m4_receipt_sha256']
        assert file_sha(Path(manifest['m3_receipt'])) == manifest['m3_receipt_sha256']

    verify_inputs()
    os.chdir(str(source))
    sys.path.insert(0, str(source))
    import numpy as np
    import torch
    from src.joint_det_dataset import Joint3DDataset
    from scripts.run_nr3d_view_pair_role import read_train_rows

    assert not torch.cuda.is_available()
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    rows = read_train_rows(Path('/root/autodl-tmp/DATA_ROOT'))
    row_ids = manifest['fit_row_ids']
    m3 = json.loads(Path(manifest['m3_receipt']).read_text())
    m4 = json.loads(Path(manifest['m4_receipt']).read_text())
    prior = {row['fit_row_id']:row for batch in m3['batches'] for row in batch['rows']}
    neighborhoods = {row['fit_row_id']:row for batch in m4['batches'] for row in batch['neighborhoods']}
    assert row_ids == m3['fit_row_ids'] == m4['fit_row_ids']

    class FixedFit(Joint3DDataset):
        def _scene_graph_parse(self, annos):
            annos[:] = [annos[i] for i in row_ids]
            super()._scene_graph_parse(annos)

    dataset = FixedFit(dataset_dict={'nr3d':1},test_dataset='nr3d',split='train',
                       data_path='/root/autodl-tmp/DATA_ROOT/',use_color=True,detect_intermediate=True,
                       butd_cls=True,skip_missing_superpoints=True)
    dataset.augment = False
    results = []
    started = time.time()
    for index,row_id in enumerate(row_ids):
        item = dataset[index]
        assert item['scan_ids'] == rows[row_id]['scan_id'] and item['target_id'] == int(rows[row_id]['target_id'])
        points = item['point_clouds']
        assert hashlib.sha256(points.tobytes()).hexdigest() == prior[row_id]['input_point_cloud_sha256']
        seed_indices = np.asarray(neighborhoods[row_id]['seed_input_indices'])
        seed_xyz = np.asarray(neighborhoods[row_id]['seed_xyz'],dtype=np.float32)
        assert np.array_equal(points[seed_indices,:3],seed_xyz)
        boxes = item['all_detected_boxes']
        valid = item['all_detected_bbox_label_mask'].astype(bool)
        assert np.array_equal(boxes,item['all_bboxes'])
        assert np.array_equal(valid,item['all_bbox_label_mask'])
        scan = dataset.scans[item['scan_ids']]
        objects = []
        for object_id in np.flatnonzero(valid):
            box = boxes[object_id]
            # Exact current float32 box bounds; no enlargement or margin tuning.
            inside = (np.abs(points[:,:3]-box[:3]) <= box[3:]/2).all(-1)
            crop_count = int(inside.sum())
            instance_points = np.asarray(scan.three_d_objects[object_id]['points'])
            member = np.zeros(len(points),dtype=bool)
            member[instance_points] = True
            object_entry = {'object_id':int(object_id),'box':box.tolist(),
                            'predicted_class':int(item['all_detected_class_ids'][object_id]),
                            'gt_class_audit_only':int(item['all_class_ids'][object_id]),
                            'instance_points_audit_only':int(len(instance_points)),
                            'crop_points':crop_count,'crop_instance_points_audit_only':int(inside[instance_points].sum()),
                            'seed_centers_in_box':int(inside[seed_indices].sum()),
                            'seed_centers_in_instance_audit_only':int(member[seed_indices].sum()),
                            'nonpositive_box_axes':int((box[3:] <= 0).sum())}
            if crop_count:
                color = points[inside,3:6].astype(np.float64)
                object_entry['native_input_rgb_mean'] = color.mean(0).tolist()
                object_entry['native_input_rgb_std'] = color.std(0).tolist()
                object_entry['crop_purity_audit_only'] = float(inside[instance_points].sum()/crop_count)
            objects.append(object_entry)
        results.append({'fit_row_id':row_id,'scan_id':item['scan_ids'],'target_id':int(item['target_id']),
                        'input_point_cloud_sha256':prior[row_id]['input_point_cloud_sha256'],'objects':objects})
        print('OBJECT POINT INPUT',json.dumps({'fit_row_id':row_id,'objects':len(objects),
              'empty_crops':sum(o['crop_points']==0 for o in objects),
              'empty_seed_rois':sum(o['seed_centers_in_box']==0 for o in objects)}),flush=True)
    all_objects = [obj for row in results for obj in row['objects']]
    first_four = [obj for row in results[:4] for obj in row['objects']]
    assert len(first_four)==190 and sum(o['predicted_class']==o['gt_class_audit_only'] for o in first_four)==155
    summary = {'rows':16,'scenes':16,'objects':len(all_objects),
               'empty_raw_crops':sum(o['crop_points']==0 for o in all_objects),
               'empty_seed_rois':sum(o['seed_centers_in_box']==0 for o in all_objects),
               'empty_seed_rois_with_at_least32_raw_points':sum(o['seed_centers_in_box']==0 and o['crop_points']>=32 for o in all_objects),
               'objects_with_nonpositive_box_axis':sum(o['nonpositive_box_axes']>0 for o in all_objects),
               'crop_purity_median_audit_only':float(np.median([o['crop_purity_audit_only'] for o in all_objects if o['crop_points']])),
               'crops_with_less_than_half_instance_points':sum(o['crop_points']>0 and o['crop_purity_audit_only']<.5 for o in all_objects)}
    verify_inputs()
    result = {'schema':'mcln-object-point-input-audit-v1','status':'complete','rows':results,'summary':summary,
              'gpu_forwards':0,'optimizer_steps':0,'heldout_rows':0,'formal_rows':0,
              'fixed_m3_m4_input_and_seed_identity':True,'existing_butd_cls_boxes_only':True,
              'instance_memberships_used_for_diagnostics_only':True,'no_segmented_object_input_added':True,
              'new_model_or_performance_claim':False,'source_and_data_unchanged':True,
              'manifest_sha256':file_sha(options.manifest),'elapsed_seconds_excluding_dataset_init':time.time()-started}
    with (addon/'receipt.json').open('x') as stream:
        json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    print('OBJECT POINT INPUT COMPLETE',json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
