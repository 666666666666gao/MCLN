"""Required real-batch runtime witnesses for the paired tail experiment."""
from collections import Counter
import torch


MASK_LOSS_NAMES = (
    'loss_mask','loss_dice','sp_loss_mask','sp_loss_dice',
    'corresponding_loss_mask','corresponding_loss_dice',
    'adaptive_weight_loss_mask','adaptive_weight_loss_dice'
)


def native_mask_loss_routes(predictions, parameters):
    """Check each native Mask loss separately, avoiding cancellation in a sum."""
    result = {}
    for name in MASK_LOSS_NAMES:
        gradients = torch.autograd.grad(predictions[name], parameters,
            retain_graph=True, allow_unused=True)
        assert all(g is None or bool((g == 0).all()) for g in gradients), name
        result[name] = sum(float(g.norm()) for g in gradients if g is not None)
    return result


def observed_forward(model, inputs):
    """Count native methods/modules without adding tensor operations or RNG use."""
    events = []
    modules = [('proposal',model.proposal_head),('query_projection',model.x_query)]
    modules += [('head'+str(i),head) for i,head in enumerate(model.prediction_heads)]
    modules += [('swa'+str(i),layer) for i,layer in enumerate(model.swa_layers)]
    modules += [('swa_ffn'+str(i),layer) for i,layer in enumerate(model.swa_ffn_layers)]
    if model.candidate_box_refiner is not None:
        modules += [('refiner',model.candidate_box_refiner)]
    handles = [module.register_forward_pre_hook(
        lambda module,args,label=label: events.append(label)) for label,module in modules]
    # Native Text/Query Mask calculations are Python methods, not nn.Modules.
    # These two temporary diagnostic intercepts only record and call originals.
    text_mask = model.prediction_head
    query_mask = model._seg_seeds_prediction

    def record_text(*args, **kwargs):
        events.append('text_mask')
        return text_mask(*args, **kwargs)

    def record_query(*args, **kwargs):
        events.append('query_mask')
        return query_mask(*args, **kwargs)

    model.prediction_head = record_text
    model._seg_seeds_prediction = record_query
    predictions = model(inputs)
    model.prediction_head = text_mask
    model._seg_seeds_prediction = query_mask
    for handle in handles:
        handle.remove()
    expected = ['proposal']+['head'+str(i) for i in range(6)]+['query_projection']
    for bid in range(inputs['batch_size']):
        expected.append('text_mask')
        for i in range(3):
            expected += ['swa'+str(i),'swa_ffn'+str(i),'text_mask']
        expected += ['text_mask','query_mask']
    if model.candidate_box_refiner is not None:
        expected.append('refiner')
    assert events == expected, (events, expected)
    return predictions, dict(order=events,counts=dict(Counter(events)),order_matches_native=True,
        refiner_enabled=model.candidate_box_refiner is not None,batch_size=inputs['batch_size'])
