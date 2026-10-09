"""Whole predicted Mask supplies the reference for the learned native box head."""
import numpy as np
import torch
from torch import nn

from pvground_boundary_box_refiner import BoundaryBoxRefiner
from whole_mask_range import member_statistics


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


class MaskReferenceBoxRefiner(BoundaryBoxRefiner):
    def __init__(self,reference_mode):
        assert reference_mode in ('native','fused_mask')
        super().__init__('distribution')
        self.reference_mode=reference_mode

    def whole_range(self,raw_points,reference_center,reference_size,end_points):
        # Keep the actual original regressor coordinates as prior features.
        return super().whole_range(raw_points,end_points['native_coarse_center'],
                                   end_points['native_coarse_size'],end_points)

    def forward(self,query,raw_points,coarse_center,coarse_size,end_points):
        assert query.shape[1:]==(256,288) and raw_points.shape==(len(query),50000,6)
        references=[];validities=[]
        xyz=raw_points[...,:3].detach().cpu().numpy().astype(np.float64)
        for bid in range(len(query)):
            geometry=member_statistics(xyz[bid],end_points['superpoints'][bid].detach().cpu().numpy(),bins=32)
            center,size,valid=extent_reference(coarse_center[bid],coarse_size[bid],
                end_points['last_pred_masks'][bid][0],end_points['sp_last_pred_masks'][bid],
                end_points['adaptive_weights'][bid],geometry)
            references.append((center,size));validities.append(valid)
        mask_center=torch.stack([value[0] for value in references])
        mask_size=torch.stack([value[1] for value in references])
        valid=torch.stack(validities)
        end_points['native_coarse_center']=coarse_center
        end_points['native_coarse_size']=coarse_size
        end_points['mask_reference_center']=mask_center
        end_points['mask_reference_size']=mask_size
        end_points['mask_reference_valid']=valid
        if self.reference_mode=='native':
            center,size=coarse_center,coarse_size
        else:
            center,size=mask_center,mask_size
        end_points['geometry_reference_center']=center
        end_points['geometry_reference_size']=size
        return super().forward(query,raw_points,center,size,end_points)


def install_mask_reference(model,reference_mode):
    old=model.candidate_box_refiner
    assert isinstance(old,BoundaryBoxRefiner) and old.boundary_mode=='distribution'
    replacement=MaskReferenceBoxRefiner(reference_mode)
    replacement.load_state_dict(old.state_dict(),strict=True)
    assert all(torch.equal(value,replacement.state_dict()[name]) for name,value in old.state_dict().items())
    # Both arms adapt the same hidden8 tensors; neutral distributions initialize
    # their changed coordinate reference, instead of carrying coarse offsets.
    nn.init.zeros_(replacement.output.weight);nn.init.zeros_(replacement.output.bias)
    assert sum(parameter.numel() for parameter in replacement.parameters())==456102
    assert len(replacement.state_dict())==10
    model.candidate_box_refiner=replacement


def reference_bounds_witness(predictions,batch):
    """M0 only: independently recompute all256 references from actual raw points."""
    max_error=0.0
    invalid=0
    for bid in range(len(batch['point_clouds'])):
        xyz=batch['point_clouds'][bid,:,:3].detach().cpu().numpy()
        ids=predictions['superpoints'][bid].detach().cpu().numpy()
        text=predictions['last_pred_masks'][bid][0].detach()
        own=predictions['sp_last_pred_masks'][bid].detach()
        alpha=predictions['adaptive_weights'][bid].detach()
        active=(alpha*text+(1-alpha)*own).sigmoid().gt(.5).cpu().numpy()
        center=predictions['mask_reference_center'][bid].detach().cpu().numpy()
        size=predictions['mask_reference_size'][bid].detach().cpu().numpy()
        valid=predictions['mask_reference_valid'][bid].detach().cpu().numpy()
        prior_center=predictions['native_coarse_center'][bid].detach().cpu().numpy()
        prior_size=predictions['native_coarse_size'][bid].detach().cpu().numpy()
        for query in range(256):
            support=xyz[active[query,ids]]
            good=bool(len(support))
            if good:
                low,high=support.min(0),support.max(0)
                good=bool((high>low).all())
            assert good==bool(valid[query])
            if good:
                expected_center=(low+high)*.5
                expected_size=high-low
            else:
                expected_center,expected_size=prior_center[query],prior_size[query]
                invalid+=1
            np.testing.assert_allclose(center[query],expected_center,rtol=0,atol=1e-6)
            np.testing.assert_allclose(size[query],expected_size,rtol=0,atol=1e-6)
            max_error=max(max_error,float(np.abs(center[query]-expected_center).max()),
                          float(np.abs(size[query]-expected_size).max()))
    return dict(actual_all256_raw_member_extent_verified=True,reference_bound_max_error=max_error,
                observed_invalid_reference_candidates=invalid,reference_uses_gt=False)
