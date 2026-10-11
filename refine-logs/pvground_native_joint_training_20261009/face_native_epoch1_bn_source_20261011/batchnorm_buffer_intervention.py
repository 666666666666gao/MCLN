"""Read-only-evaluation intervention: replace only actual BatchNorm buffers."""
import torch
from torch import nn


def batchnorm_buffer_keys(model):
    return tuple(name+'.'+field
        for name,module in model.named_modules()
        if isinstance(module,nn.modules.batchnorm._BatchNorm)
        for field in ('running_mean','running_var','num_batches_tracked'))


def capture_batchnorm_buffers(model):
    state=model.state_dict()
    return {key:state[key].detach().clone() for key in batchnorm_buffer_keys(model)}


def install_batchnorm_buffers(model,buffers):
    keys=batchnorm_buffer_keys(model)
    assert set(keys)==set(buffers)
    state=model.state_dict()
    with torch.no_grad():
        for key in keys:
            value=buffers[key]
            assert value.shape==state[key].shape and value.dtype==state[key].dtype
            state[key].copy_(value.to(device=state[key].device))
