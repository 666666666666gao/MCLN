"""Task-specific queries read shared observed memories for one candidate identity.

The semantic and geometric queries share content/state Key/Value projections,
attention parameters, output projection and decoder tail parameters. Only the
query transforms differ. The shared dropout masks keep task differences from
being created by unrelated random masks.
"""
import torch
from torch import nn
from torch.nn import functional as F

from pvground_observation_query import ObservationQueryRead, install_observation_query_read


class TaskObservationQueryRead(ObservationQueryRead):
    def __init__(self, reader):
        nn.Module.__init__(self)
        assert type(reader) is ObservationQueryRead
        self.enabled = reader.enabled
        for name, module in reader.named_children():
            self.add_module(name, module)
        # Reuse C's actual initialized parameters; these zeros consume no RNG.
        self.task_queries = nn.Parameter(torch.zeros(2, 288, 288))

    def forward(self, query, source_features, source_position, observations):
        semantic = query + F.linear(query, self.task_queries[0])
        geometry = query + F.linear(query, self.task_queries[1])
        evidence = super().forward(torch.cat([semantic, geometry], dim=0),
                                   source_features, source_position, observations)
        return evidence.chunk(2, dim=0)


def shared_dropout(dropout, first, second):
    mask = dropout(torch.ones_like(first))
    return first * mask, second * mask


def finish_task_queries(layer, query, visual, residuals):
    """Finish the last decoder with separate values and shared parameters/masks."""
    semantic, geometry = shared_dropout(layer.dropout_v,
                                        visual + residuals[0], visual + residuals[1])
    semantic = layer.norm_v(query + semantic)
    geometry = layer.norm_v(query + geometry)
    assert len(layer.ffn) == 5
    sem_hidden = layer.ffn[1](layer.ffn[0](semantic))
    geo_hidden = layer.ffn[1](layer.ffn[0](geometry))
    sem_hidden, geo_hidden = shared_dropout(layer.ffn[2], sem_hidden, geo_hidden)
    sem_hidden = layer.ffn[3](sem_hidden)
    geo_hidden = layer.ffn[3](geo_hidden)
    sem_hidden, geo_hidden = shared_dropout(layer.ffn[4], sem_hidden, geo_hidden)
    semantic = layer.norm2(semantic + sem_hidden)
    geometry = layer.norm2(geometry + geo_hidden)
    return semantic.transpose(0, 1).contiguous(), geometry.transpose(0, 1).contiguous()


def install_task_observation_query_read(model):
    install_observation_query_read(model)
    last = model.decoder[-1]
    assert not last.task_read
    last.source_query_read = TaskObservationQueryRead(last.source_query_read)
    last.source_query_read.train(last.training)
    last.task_read = True
