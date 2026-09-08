"""Synthetic operator/unit checks, not scene evaluation or model accuracy."""
from pathlib import Path
from types import SimpleNamespace
import json,hashlib
import torch
from torch import nn
from pcdet.ops.pointnet2.pointnet2_stack import pointnet2_utils
from pvground_source_query import SourceQueryRead
from pvground_observation_query import ObservedQueryAndGroup, bev_observation, ObservationQueryRead, OBSERVATION_WIDTHS

torch.manual_seed(2027);torch.cuda.manual_seed_all(2027)
xyz=torch.tensor([[0,0,0],[.03125,0,0],[.0625,0,0],[.5,0,0],
                  [10,0,0],[10.1875,0,0],[10.3125,0,0]],device='cuda')
queries=torch.tensor([[0,0,0],[1,0,0],[.5,0,0],[10,0,0],[10.09375,0,0],[20,0,0]],device='cuda')
counts=torch.tensor([4,3],dtype=torch.int32,device='cuda')
qcounts=torch.tensor([3,3],dtype=torch.int32,device='cuda')
features=torch.arange(21,dtype=torch.float32,device='cuda').view(7,3)
original=pointnet2_utils.QueryAndGroup(.125,2,use_xyz=True)
observed=ObservedQueryAndGroup(original)
with torch.no_grad():
    before=original(xyz,counts,queries,qcounts,features)
    after=observed(xyz,counts,queries,qcounts,features)
assert all(torch.equal(a,b) for a,b in zip(before,after))
assert observed.observation[:,0].tolist()==[1,0,1,1,1,0]
assert observed.observation[:,1].tolist()==[1,0,.5,.5,1,0]
assert observed.observation[:,5].tolist()==[1,0,0,0,1,0]
assert torch.equal(observed.observation[[1,5]],torch.zeros((2,6),device='cuda'))
# Independent brute-force selected-neighbor oracle, respecting strict radius and cap.
for row in range(6):
    batch=row//3;start=0 if batch==0 else 4;stop=4 if batch==0 else 7
    distances=(xyz[start:stop]-queries[row]).square().sum(-1)
    selected=torch.nonzero(distances < .125**2).view(-1)[:2]
    if len(selected):
        distance=distances[selected].sqrt()/.125
        expected=torch.stack([distance.min(),distance.mean(),distance.max()])
        assert torch.equal(observed.observation[row,2:5],expected)
        assert int(observed.observation[row,1]*2)==len(selected)

indices=torch.tensor([[0,0,0,0],[0,1,1,1],[1,0,2,2]],device='cuda',dtype=torch.int32)
batch=dict(encoded_spconv_tensor=SimpleNamespace(indices=indices,spatial_shape=(2,3,4)),
           spatial_features=torch.zeros((2,128,3,4),device='cuda'),spatial_features_stride=1)
keypoints=torch.tensor([[0,.25,.5,.5],[0,.25,.5,3],[1,2,2,.5]],device='cuda')
state=bev_observation(keypoints,batch,[1,1,1],[0,0,0,4,3,2])
assert state.shape==(3,10) and torch.isfinite(state).all()
assert state[0].tolist()==[1,1,1,0,0,1,.375,.375,.125,.125]
assert state[1,0].item()==0 and torch.equal(state[0,1:],state[1,1:])
assert state[2,1].item()==0  # edge stencil is clamped, not an interior stencil

fusion=nn.Linear(1024,288,bias=False);attention=nn.MultiheadAttention(288,8,dropout=.1)
saved=torch.get_rng_state();control=SourceQueryRead(fusion,attention);after_control=torch.get_rng_state()
torch.set_rng_state(saved);candidate=ObservationQueryRead(fusion,attention)
assert torch.equal(torch.get_rng_state(),after_control)
for key,value in control.state_dict().items():assert torch.equal(value,candidate.state_dict()[key]),key
extra=sum(p.numel() for p in candidate.parameters())-sum(p.numel() for p in control.parameters())
assert extra==43200 and sum(p.numel() for p in candidate.parameters())==757728
assert len(candidate.state_dict())==36 and len(control.state_dict())==24
# Nonzero output projection exposes zero-state equivalence beyond a zero residual.
with torch.no_grad():
    control.output.weight.copy_(torch.eye(288));candidate.output.weight.copy_(control.output.weight)
control.cuda();candidate.cuda()
features=torch.randn(2,4,1024,device='cuda');position=torch.randn(2,4,288,device='cuda')
query=torch.randn(3,2,288,device='cuda')
states=[torch.randn(2,4,width,device='cuda') for width in OBSERVATION_WIDTHS]
old=control(query,features,position);new=candidate(query,features,position,states)
assert torch.equal(old,new)
new.square().mean().backward()
for parameters in [candidate.observation_keys,candidate.observation_values]:
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0 for p in parameters)
result=dict(status='pass',synthetic=True,full_model_forwards=0,optimizer_steps=0,new_checkpoints=0,formal_rows=0,
            cuda_group_values_exact=True,distinct_selected_support_oracle=True,empty_and_batch_cases=True,
            bev_stencil_state_checked=True,initial_parameter_and_rng_match_B=True,nonzero_output_zero_state_exact=True,
            observation_key_value_gradients=True,added_parameters_over_B=extra,total_reader_parameters=757728,reader_state_tensors=36)
root=Path(__file__).parent
result['module_sha256']=hashlib.sha256((root/'pvground_observation_query.py').read_bytes()).hexdigest()
(root/'unit_receipt.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
