"""CUDA routing/gradient checks for D; synthetic data, no accuracy claim."""
import copy, json, sys
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from pvground_observation_query import ObservationQueryRead, OBSERVATION_WIDTHS
from pvground_task_observation_query import TaskObservationQueryRead, finish_task_queries

torch.manual_seed(2027)
torch.cuda.manual_seed_all(2027)
fusion = nn.Linear(1024, 288, bias=False)
attention = nn.MultiheadAttention(288, 8, dropout=.1)
control = ObservationQueryRead(fusion, attention)
state = torch.get_rng_state()
candidate = TaskObservationQueryRead(copy.deepcopy(control))
assert torch.equal(state, torch.get_rng_state())
assert len(candidate.state_dict()) == 37
assert sum(p.numel() for p in candidate.parameters()) == 923616
for name, value in control.state_dict().items():
    assert torch.equal(value, candidate.state_dict()[name]), name
control.cuda(); candidate.cuda()
query = torch.randn(7, 2, 288, device='cuda', requires_grad=True)
features = torch.randn(2, 11, 1024, device='cuda', requires_grad=True)
position = torch.randn(2, 11, 288, device='cuda')
observations = [torch.randn(2, 11, w, device='cuda') for w in OBSERVATION_WIDTHS]
with torch.no_grad():
    control.output.weight.copy_(torch.eye(288, device='cuda'))
    candidate.output.weight.copy_(control.output.weight)
reference = control(query, features, position, observations)
semantic, geometry = candidate(query, features, position, observations)
zero_error = max(float((semantic-reference).abs().max()), float((geometry-reference).abs().max()))
assert torch.allclose(semantic, reference, atol=1e-6, rtol=1e-5)
assert torch.equal(semantic, geometry)
with torch.no_grad():
    candidate.task_queries[0].copy_(torch.eye(288, device='cuda')*.2)
sem_changed, geo_same = candidate(query, features, position, observations)
assert torch.equal(geometry, geo_same)
assert float((sem_changed-semantic).abs().max()) > 1e-5
with torch.no_grad():
    candidate.task_queries[1].copy_(torch.eye(288, device='cuda')*-.15)
sem_same, geo_changed = candidate(query, features, position, observations)
assert torch.equal(sem_changed, sem_same)
assert float((geo_changed-geo_same).abs().max()) > 1e-5
sem_grad = torch.autograd.grad(sem_same.square().mean(), candidate.task_queries, retain_graph=True)[0]
geo_grad = torch.autograd.grad(geo_changed.square().mean(), candidate.task_queries, retain_graph=True)[0]
assert sem_grad[0].norm()>0 and sem_grad[1].count_nonzero()==0
assert geo_grad[1].norm()>0 and geo_grad[0].count_nonzero()==0
(sem_same.square().mean()+geo_changed.square().mean()).backward()
assert features.grad.norm()>0 and query.grad.norm()>0

layer = SimpleNamespace(dropout_v=nn.Dropout(.1).cuda(), norm_v=nn.LayerNorm(288).cuda(),
    ffn=nn.Sequential(nn.Linear(288,256),nn.ReLU(),nn.Dropout(.1),nn.Linear(256,288),nn.Dropout(.1)).cuda(),
    norm2=nn.LayerNorm(288).cuda())
visual = torch.randn_like(query); residual = torch.randn_like(query)
tail_checks = []
for training in [False, True]:
    for module in vars(layer).values(): module.train(training)
    before = torch.cuda.get_rng_state()
    original = layer.norm_v(query+layer.dropout_v(visual+residual))
    original = layer.norm2(original+layer.ffn(original)).transpose(0,1).contiguous()
    end = torch.cuda.get_rng_state()
    torch.cuda.set_rng_state(before)
    first, second = finish_task_queries(layer, query, visual, (residual,residual))
    assert torch.equal(first, second)
    assert torch.equal(first, original)
    assert torch.equal(end, torch.cuda.get_rng_state())
    tail_checks.append({'training':training,'output_exact':True,'rng_exact':True})

sys.path.insert(0, '/root/autodl-tmp/mcln_pvground_task_observation_source_20260917_v1/PV-Ground')
from models.modules import ClsAgnosticPredictHead
head = ClsAgnosticPredictHead(256,1,7,288,objectness=False,heading=False).cuda().eval()
sem = sem_same.transpose(0,1).transpose(1,2).detach().requires_grad_()
geo = geo_changed.transpose(0,1).transpose(1,2).detach().requires_grad_()
out = {}
head(sem, torch.zeros(2,7,3,device='cuda'), out, prefix='last_', geometry_features=geo)
sem_paths = torch.autograd.grad(out['last_sem_cls_scores'].square().mean(), (sem,geo), allow_unused=True, retain_graph=True)
geo_paths = torch.autograd.grad(out['last_center'].square().mean()+out['last_pred_size'].square().mean(), (sem,geo), allow_unused=True)
assert sem_paths[0].norm()>0 and sem_paths[1] is None
assert geo_paths[0] is None and geo_paths[1].norm()>0
record = dict(status='pass', module_parameters=923616, extra_parameters=165888, added_tensors=37,
    zero_task_vs_control_max_abs=zero_error, semantic_geometry_direct_gradient_separation=True,
    shared_tail=tail_checks, prediction_head_direct_routing=True, optimizer_steps=0,
    full_model_forwards=0, formal_rows=0, metric_claim=False)
(Path(__file__).parent/'unit.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
