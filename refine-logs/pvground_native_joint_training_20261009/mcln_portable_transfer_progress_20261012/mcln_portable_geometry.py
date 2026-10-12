"""Current best portable support/range mechanisms inside native MCLN forward."""
import numpy as np
import torch

from whole_mask_range import member_statistics
from native_mask_geometry import native_mask_geometry


def finish_portable_geometry(model, decoder_query, mask_query, supports, inputs, predictions):
    assert model.portable_geometry_arm in ('mask_reference', 'full_transfer')
    assert model.source_choice_selector is None and model.source_moe is None
    assert model.cs_structure is None and model.cs_context_reader is None and model.cs_box_refiner is None
    points = inputs['point_clouds']
    assert points.shape[1:] == (50000, 6)
    predictions['support_geometry_query'] = decoder_query
    predictions['support_query_features'] = mask_query
    predictions['support_super_features'] = supports
    if model.portable_geometry_arm == 'full_transfer':
        predictions['support_parent_query_masks'] = predictions['sp_last_pred_masks']
        corrected, geometries = model.candidate_support_corrector(
            mask_query, supports, points, predictions['last_center'],
            predictions['last_pred_size'], predictions)
        predictions['sp_last_pred_masks'] = corrected
    else:
        xyz = points[..., :3].detach().cpu().numpy().astype(np.float64)
        geometries = [member_statistics(xyz[bid], predictions['superpoints'][bid].detach().cpu().numpy(), bins=32)
                      for bid in range(len(points))]
    predictions['support_member_geometry'] = geometries
    native_mask_geometry(predictions, geometries)
    if model.portable_geometry_arm == 'full_transfer':
        center, size, evidence = model.candidate_span_mixer(decoder_query, supports, predictions, geometries)
        predictions['span_mask_center'] = predictions['last_center']
        predictions['span_mask_size'] = predictions['last_pred_size']
        predictions['last_center'], predictions['last_pred_size'] = center, size
        predictions['span_axis_gate'] = torch.stack([row['axis_gate'] for row in evidence])
        predictions['span_raw_axis_gate'] = torch.stack([row['raw_axis_gate'] for row in evidence])
        predictions['span_source_fraction'] = torch.stack([row['source_fraction'] for row in evidence])

