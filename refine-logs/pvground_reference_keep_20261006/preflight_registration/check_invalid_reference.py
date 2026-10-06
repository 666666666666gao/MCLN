"""Execute39 actual empty-support rows through the reviewed reference function."""
from pathlib import Path
import numpy as np
import torch
from mask_reference import extent_reference

def check_invalid_reference(root):
    rows=0
    with np.load(Path(root)/'invalid_reference_fixture.npz',allow_pickle=False) as fixture:
        for index in range(39):
            prefix='r%d__'%index
            own=torch.from_numpy(fixture[prefix+'own_logits']).cuda()[None]
            text=torch.from_numpy(fixture[prefix+'text_logits']).cuda()[None]
            alpha=torch.from_numpy(fixture[prefix+'alpha']).cuda()
            prior=torch.from_numpy(fixture[prefix+'prior_box']).cuda()[None]
            geometry=dict(native_ids=np.arange(own.shape[1]),origin=np.zeros(3),span=np.ones(3),
                lower=fixture[prefix+'lower'],upper=fixture[prefix+'upper'])
            center,size,valid=extent_reference(prior[:,:3],prior[:,3:],text,own,alpha,geometry)
            assert not valid.any()
            assert torch.equal(center,prior[:,:3]) and torch.equal(size,prior[:,3:])
            rows+=1
    return dict(actual_empty_support_rows_verified=rows,actual_prior_retained_exact=True,no_gt_fields=True)
