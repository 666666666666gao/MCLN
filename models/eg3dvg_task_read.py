"""Task-conditioned requests to EG's existing GMA and SWA visual memories.

The visual reading/tail operations follow Gwan9Wook/EG3DVG commit
174e34894aea6513442da6b5dfa9b3e2bf8a1efa, models/encoder_decoder_layers.py.
Retain that project's CC-BY-NC-SA license and upstream attribution when using
this adaptation. This module does not add PV-Ground's six-source memory.
"""
import torch
from torch import nn
from torch.nn import functional as F


def install_task_read(model):
    """Call after strictly loading the author checkpoint, before optimization."""
    layer = model.decoder[-1]
    assert layer.task_queries is None
    assert layer.norm_v.normalized_shape == (288,)
    layer.task_queries = nn.Parameter(torch.zeros(2, 288, 288))


def read_one(layer, query, request, query_pos, vis_feats, spatial_feature,
             padding_mask, lang_feats, super_features, attn_mask_list):
    """Keep the native two memories and shared tail; change the read request."""
    visual = layer.cross_v(
        q=(request + query_pos).transpose(0, 1), k=vis_feats, v=vis_feats,
        pairwise_locs=spatial_feature, key_padding_mask=padding_mask,
        txt_embeds=torch.max(lang_feats, dim=1)[0])[0]
    visual = visual.transpose(0, 1).contiguous()
    query_v = layer.norm_v(query + layer.dropout_v(visual))
    super_queries = []
    for bs in range(query.shape[1]):
        value, _, _ = layer.cross_s(
            super_features[bs].unsqueeze(0).transpose(1, 2).transpose(0, 1),
            request[:, bs].unsqueeze(1), attn_mask=attn_mask_list[bs])
        super_queries.append(value)
    query_s = torch.cat(super_queries)
    result = layer.norm_sp(query_v + query_s.transpose(0, 1))
    return layer.norm2(result + layer.ffn(result))


def read_tasks(layer, query, query_pos, vis_feats, spatial_feature,
               padding_mask, lang_feats, super_features, attn_mask_list):
    semantic_request = query + F.linear(query, layer.task_queries[0])
    geometry_request = query + F.linear(query, layer.task_queries[1])
    args = (query_pos, vis_feats, spatial_feature, padding_mask, lang_feats,
            super_features, attn_mask_list)
    # Both tasks must see the same stochastic masks. Advance global RNG by one
    # read, as in the native control, including on subsequent negative forwards.
    if layer.training:
        cpu_before = torch.get_rng_state()
        devices = [query.device] if query.is_cuda else []
        cuda_before = torch.cuda.get_rng_state(query.device) if query.is_cuda else None
        semantic = read_one(layer, query, semantic_request, *args)
        with torch.random.fork_rng(devices=devices):
            torch.set_rng_state(cpu_before)
            if query.is_cuda:
                torch.cuda.set_rng_state(cuda_before, query.device)
            geometry = read_one(layer, query, geometry_request, *args)
    else:
        semantic = read_one(layer, query, semantic_request, *args)
        geometry = read_one(layer, query, geometry_request, *args)
    return geometry, semantic.transpose(0, 1).contiguous()
