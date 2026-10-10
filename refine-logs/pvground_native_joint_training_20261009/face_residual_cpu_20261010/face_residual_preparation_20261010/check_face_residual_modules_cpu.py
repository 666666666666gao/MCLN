"""Synthetic CPU checks of prepared face residual; no native training/REC."""
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys

assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
import numpy as np
import torch
from torch import nn


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


root = Path(__file__).resolve().parent
bundle = json.loads((root/'CPU_BUNDLE.json').read_bytes())
original = Path(bundle['original_source'])
assert not (root/'CPU_MODULE_RESULT.json').exists()
for name, digest in bundle['original_sha256'].items():
    assert hashlib.sha256((original/name).read_bytes()).hexdigest() == digest
for name, digest in bundle['prepared_sha256'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest
torch.set_num_threads(1)
torch.manual_seed(2027)
np.random.seed(2027)
old_A = load_module('face_check_A', original/'mask_support_corrector.py')
old_B = load_module('face_check_B', original/'extremal_span_mixer.py')
geometry_module = load_module('face_check_geometry', original/'native_mask_geometry.py')
prepared = load_module('prepared_face_residual', root/'face_residual_span_mixer.py')
for role in ('support','span'):
    path = Path(bundle[role+'_checkpoint'])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == bundle[role+'_checkpoint_sha256']
support_payload = torch.load(bundle['support_checkpoint'],map_location='cpu')
span_payload = torch.load(bundle['span_checkpoint'],map_location='cpu')
assert support_payload['arm'] == 'selected_query' and support_payload['geometry_encoding'] == 'zero'
assert span_payload['source_mode'] == 'extremal_support'
prefix = 'candidate_support_corrector.'
a_state = {name[len(prefix):]:value for name,value in support_payload['state_delta'].items()}
assert len(a_state) == 10 and len(span_payload['mixer_state']) == 14
a = old_A.CandidateMaskSupportCorrector(False)
a.load_state_dict(a_state,strict=True)
prior = old_B.ExtremalSpanMixer('extremal_support')
prior.load_state_dict(span_payload['mixer_state'],strict=True)
source_visible = prepared.FaceResidualSpanMixer(copy.deepcopy(prior),True)
source_hidden = prepared.FaceResidualSpanMixer(copy.deepcopy(prior),False)
source_hidden.load_state_dict(source_visible.state_dict(),strict=True)
assert sum(p.numel() for p in source_visible.parameters()) == 53218
assert sum(p.numel() for p in source_visible.face_residual.parameters()) == 23425
assert len(source_visible.state_dict()) == 20
assert len(source_visible.face_residual.state_dict()) == 6
assert not torch.count_nonzero(source_visible.face_residual[-1].weight)
assert not torch.count_nonzero(source_visible.face_residual[-1].bias)

query = torch.randn(1,256,288)
supports = [torch.randn(288,8)]
points = torch.randn(1,50000,6)
native_center = torch.randn(1,256,3)
native_size = torch.rand(1,256,3)+.5
own = torch.randn(256,8)
text = torch.randn(256,8)
prediction = dict(superpoints=[torch.arange(50000)%8],sp_last_pred_masks=[own],
    last_pred_masks=[[text]],adaptive_weights=[torch.tensor(.4)],
    last_center=native_center,last_pred_size=native_size)
with torch.no_grad():
    corrected,geometries = a(query,supports,points,native_center,native_size,prediction)
    prediction['sp_last_pred_masks'] = corrected
    geometry_module.native_mask_geometry(prediction,geometries)
    pc,ps,pe = prior(query,supports,prediction,geometries)
    vc,vs,ve = source_visible(query,supports,prediction,geometries)
captures = []
hook = source_hidden.face_residual[0].register_forward_pre_hook(
    lambda module,args: captures.append(args[0].detach().clone()))
with torch.no_grad():
    hc,hs,he = source_hidden(query,supports,prediction,geometries)
hook.remove()
assert torch.equal(vc,pc) and torch.equal(vs,ps)
assert torch.equal(hc,pc) and torch.equal(hs,ps)
assert len(captures) == 1 and captures[0].shape == (256,6,332)
assert not torch.count_nonzero(captures[0][...,:32])
assert not torch.count_nonzero(captures[0][...,324])
assert torch.equal(ve[0]['face_gate'],pe[0]['axis_gate'][:,:,None].expand(-1,-1,2).reshape(256,6))

# An explicitly separate gradient fixture uses interior gates so that the
# new output's chain rule is exercised rather than blocked by gate saturation.
gradient_module = prepared.FaceResidualSpanMixer(old_B.ExtremalSpanMixer('extremal_support'),True)
nc = torch.full((256,3),.3)
ns = torch.full((256,3),1.5)
mc = torch.zeros_like(nc)
ms = torch.ones_like(ns)
prior_center = .5*(nc+mc)
prior_size = .5*(ns+ms)
evidence = dict(face_tokens=torch.randn(256,6,32),axis_gate=torch.full((256,3),.5),
    source_fraction=torch.full((256,3,2),.125),reference_valid=torch.ones(256,dtype=torch.bool))
gq = torch.randn(256,288)
optimizer = torch.optim.AdamW(gradient_module.face_residual.parameters(),lr=1e-3,weight_decay=5e-4)
gradients = []
for step in range(2):
    optimizer.zero_grad()
    gc,gs,ge = gradient_module.refine_one(gq,nc,ns,mc,ms,prior_center,prior_size,evidence)
    loss = ((gc-.2)**2).mean()+((gs-1.1)**2).mean()
    loss.backward()
    output_grad = gradient_module.face_residual[-1].weight.grad
    encoder_grad = gradient_module.face_residual[0].weight.grad
    assert output_grad is not None and torch.count_nonzero(output_grad)>0
    assert encoder_grad is not None
    if step == 0:
        assert not torch.count_nonzero(encoder_grad)
    else:
        assert torch.count_nonzero(encoder_grad)>0
    assert torch.isfinite(gc).all() and torch.isfinite(gs).all() and (gs>0).all()
    gradients.append(dict(step=step,output_gradient_norm=float(output_grad.norm()),
        encoder_gradient_norm=float(encoder_grad.norm()),loss=float(loss)))
    optimizer.step()

# Supply a known residual to exercise the decoder's real inverted-face path.
# This fixture does not claim that the learned head predicted these values.
class SuppliedFaceResidual(nn.Module):
    def forward(self, inputs):
        values = inputs.new_zeros((256,6,1))
        values[:,0,0] = .5
        values[:,1,0] = -.5
        return values

ordering_module = copy.deepcopy(gradient_module)
ordering_module.face_residual = SuppliedFaceResidual()
disjoint_nc = torch.zeros_like(nc)
disjoint_nc[:,0] = 4
unit_size = torch.ones_like(ns)
ordered_center,ordered_size,ordered_evidence = ordering_module.refine_one(
    gq,disjoint_nc,unit_size,mc,unit_size,disjoint_nc*.5,unit_size,evidence)
assert torch.equal(ordered_evidence['raw_face_size'][:,0],torch.full((256,),-3.))
assert torch.equal(ordered_size[:,0],torch.full((256,),3.))
assert torch.isfinite(ordered_center).all() and (ordered_size>0).all()

# Module state only, in memory. This is not full PV/optimizer recovery.
gradient_module.eval()
with torch.no_grad():
    before = gradient_module.refine_one(gq,nc,ns,mc,ms,prior_center,prior_size,evidence)
buffer = io.BytesIO()
torch.save(gradient_module.state_dict(),buffer)
buffer.seek(0)
reloaded = prepared.FaceResidualSpanMixer(old_B.ExtremalSpanMixer('extremal_support'),True)
reloaded.load_state_dict(torch.load(buffer,map_location='cpu'),strict=True)
reloaded.eval()
with torch.no_grad():
    after = reloaded.refine_one(gq,nc,ns,mc,ms,prior_center,prior_size,evidence)
assert torch.equal(before[0],after[0]) and torch.equal(before[1],after[1])
assert not torch.cuda.is_initialized()
result = dict(status='ACTUAL_FACE_RESIDUAL_SYNTHETIC_CPU_MODULE_PASS',torch_version=torch.__version__,
    fixture=dict(seed=2027,batch=1,queries=256,superpoints=8,points=50000),
    prior_parameters=29793,new_face_parameters=23425,combined_parameters=53218,
    combined_module_states=20,new_face_module_states=6,
    zero_output_boxes_bitwise_equal_loaded_prior_both_modes=True,
    hidden_source_token_and_fraction_columns_zero=True,synthetic_gradient_steps=gradients,
    supplied_inverted_face_decoding_checked=True,module_state_in_memory_roundtrip_bitwise_equal=True,
    CUDA_initialized=False,GPU_calls=0,real_loader_rows=0,current_training_queries=0,
    full_PV_factory_checked=False,native_criterion_checked=False,full_optimizer_recovery_checked=False,
    learned_inverted_face_prediction_proven=False,new_weight_files=0,formal_accuracy=None,
    GPU_training_admission=False,full_goal_complete=False)
(root/'CPU_MODULE_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
