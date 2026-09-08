"""Observe one frozen batch, reusing the pinned three-forward replay unchanged."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_bytes())
    os.chdir(spec['model_source'])
    sys.path.insert(0, spec['model_source'])
    import numpy as np
    import torch
    from models.pv_ground import PVGround
    from pcdet.ops.pointnet2.pointnet2_stack import pointnet2_utils
    from pvground_support_observation import observe_support, stack_indices_to_global

    packets = []
    original_forward = PVGround.forward

    def first_forward(model, inputs):
        # Later replays use the unwrapped class method, checking observation effects.
        PVGround.forward = original_forward
        vsa = model.backbone_net.vsa
        assert 'get_sampled_points' not in vsa.__dict__
        assert 'aggregate_keypoint_features_from_one_source' not in vsa.__dict__
        names = {id(vsa.SA_rawpoints): 'raw_points'}
        names.update({id(layer): name for layer, name in zip(vsa.SA_layers, vsa.SA_layer_names)})
        original_points = vsa.get_sampled_points
        original_aggregate = vsa.aggregate_keypoint_features_from_one_source
        original_ball = pointnet2_utils.ball_query
        context = {}

        def sampled_points(batch_dict):
            points = original_points(batch_dict)
            context['query_batch'] = points[:, 0].detach().cpu().numpy().astype(np.int64)
            return points

        def aggregate(**kwargs):
            assert not kwargs['filter_neighbors_with_roi']
            context['source'] = names[id(kwargs['aggregate_func'])]
            labels = kwargs['xyz_bs_idxs'].detach().cpu().numpy()
            context['support_batch'] = labels.astype(np.int64)
            assert np.array_equal(labels, context['support_batch'])
            result = original_aggregate(**kwargs)
            del context['source']
            return result

        def ball(radius, nsample, xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt):
            indices, empty = original_ball(radius, nsample, xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt)
            source = context['source']
            supports = xyz_batch_cnt.detach().cpu().numpy()
            queries = new_xyz_batch_cnt.detach().cpu().numpy()
            actual_batch = context['query_batch']
            assert np.array_equal(actual_batch, np.repeat(np.arange(len(queries)), queries))
            assert np.array_equal(context['support_batch'], np.repeat(np.arange(len(supports)), supports))
            global_indices, valid = stack_indices_to_global(
                indices.detach().cpu().numpy(), empty.detach().cpu().numpy(), supports, queries)
            # Fixed index stride, never GT/score/coverage-based selection.
            selected = np.concatenate([np.flatnonzero(actual_batch == b)[::32] for b in range(len(queries))])
            packet = dict(support_xyz=xyz.detach().cpu().numpy().copy(),
                          support_batch=context['support_batch'].copy(),
                          query_xyz=new_xyz.detach().cpu().numpy()[selected].copy(),
                          query_batch=actual_batch[selected].copy(),
                          selected_global_indices=global_indices[selected].copy(),
                          selected_valid=valid[selected].copy(), query_original_indices=selected,
                          support_batch_counts=supports, query_batch_counts=queries,
                          all_query_operator_valid=valid)
            packets.append((source, float(radius), int(nsample), packet))
            return indices, empty

        vsa.get_sampled_points = sampled_points
        vsa.aggregate_keypoint_features_from_one_source = aggregate
        pointnet2_utils.ball_query = ball
        # Restore shared callable bindings even when an observed forward fails.
        try:
            return original_forward(model, inputs)
        finally:
            del vsa.get_sampled_points
            del vsa.aggregate_keypoint_features_from_one_source
            pointnet2_utils.ball_query = original_ball

    PVGround.forward = first_forward
    sys.argv = [str(args.spec.parent/'replay.py'), '--spec', str(args.spec), '--output', str(args.output)]
    runpy.run_path(str(args.spec.parent/'replay.py'), run_name='__main__')
    replay = json.loads((args.output/'receipt.json').read_bytes())
    assert replay['repeated_seed_exact'], 'Observed first forward differs from unobserved reset-seed replay'
    assert len(packets) == 10 and {p[0] for p in packets} == {'raw_points','x_conv1','x_conv2','x_conv3','x_conv4'}
    captures = args.output/'support'
    captures.mkdir()
    records = []
    for index, (source, radius, cap, packet) in enumerate(packets):
        assert len(packet['query_xyz']) == 256 and packet['selected_global_indices'].shape == (256, cap)
        path = captures/('%02d_%s.npz' % (index, source))
        np.savez_compressed(str(path), **packet)
        rows = observe_support(*(packet[key] for key in ['support_xyz','support_batch','query_xyz','query_batch',
                                                        'selected_global_indices','selected_valid']), radius=radius)
        assert all(r['selected_wrong_batch_slots'] == 0 for r in rows)
        report = dict(source=source, radius_m=radius, cap=cap, query_location='vsa_keypoint',
                      packet_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      all_query_count=int(len(packet['all_query_operator_valid'])),
                      all_query_operator_empty=int((~packet['all_query_operator_valid']).sum()), rows=rows)
        report_path = path.with_suffix('.json')
        report_path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        records.append(dict(source=source, radius_m=radius, cap=cap, packet=path.name,
                            packet_sha256=report['packet_sha256'], report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
                            all_query_count=report['all_query_count'],all_query_operator_empty=report['all_query_operator_empty'],
                            diagnostic_query_count=len(rows), geometry_empty=sum(not r['geometry_nonempty'] for r in rows),
                            max_selected_unique_rows=max(r['selected_unique_rows'] for r in rows)))
    result = dict(status='pass',records=records,full_forwards=3,optimizer_steps=0,formal_rows=0,
                  observed_vs_unobserved_seed_repeat_exact=True,input_sha256=replay['input_sha256'],
                  checkpoint_sha256=replay['checkpoint_sha256'],seed=spec['seed'],
                  scope='one pinned eight-row batch, parent weights; sampled keypoint geometry is not candidate-box coverage or dataset rate')
    (args.output/'support_receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    print('SUPPORT_CAPTURE_COMPLETE '+json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
