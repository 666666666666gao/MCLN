"""Discrete prediction-only Mask reference inside native model.forward."""
import torch


def extent_reference(coarse_center, coarse_size, text_logits, query_logits, alpha, geometry):
    """One reference per Query; absent/degenerate support retains the native prior.

    The completed9508 diagnostic has39 selected invalid extents. This validity
    state is not an IoU/identity/quality gate and never reads GT.
    """
    assert text_logits.shape==query_logits.shape and query_logits.ndim==2 and alpha.ndim==0
    slots=torch.as_tensor(geometry['native_ids'],device=query_logits.device,dtype=torch.long)
    fused=alpha.detach()*text_logits.detach()+(1-alpha.detach())*query_logits.detach()
    active=fused[:,slots].sigmoid()>.5
    lower=query_logits.new_tensor(geometry['origin']+geometry['lower']*geometry['span'])
    upper=query_logits.new_tensor(geometry['origin']+geometry['upper']*geometry['span'])
    low=lower[None].expand(len(query_logits),-1,-1).masked_fill(~active[...,None],float('inf')).min(1).values
    high=upper[None].expand(len(query_logits),-1,-1).masked_fill(~active[...,None],float('-inf')).max(1).values
    valid=active.any(1)&(high>low).all(-1)
    center=torch.where(valid[:,None],(low+high)*.5,coarse_center)
    size=torch.where(valid[:,None],high-low,coarse_size)
    assert torch.isfinite(center).all() and torch.isfinite(size).all()
    return center,size,valid


def native_mask_geometry(predictions, geometries):
    native_center = predictions['last_center']
    native_size = predictions['last_pred_size']
    centers, sizes, validities = [], [], []
    for bid, geometry in enumerate(geometries):
        center, size, valid = extent_reference(native_center[bid], native_size[bid],
            predictions['last_pred_masks'][bid][0], predictions['sp_last_pred_masks'][bid],
            predictions['adaptive_weights'][bid], geometry)
        centers.append(center)
        sizes.append(size.clamp_min(1e-6))
        validities.append(valid)
    predictions['native_coarse_center'] = native_center
    predictions['native_coarse_size'] = native_size
    predictions['mask_reference_valid'] = torch.stack(validities)
    predictions['mask_reference_center'] = torch.stack(centers)
    predictions['mask_reference_size'] = torch.stack(sizes)
    predictions['last_center'] = predictions['mask_reference_center']
    predictions['last_pred_size'] = predictions['mask_reference_size']
