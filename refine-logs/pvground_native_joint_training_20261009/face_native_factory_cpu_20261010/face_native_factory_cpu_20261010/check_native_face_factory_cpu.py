"""Two real PV native factory constructions; CPU only, no neural forward."""
import copy
import gc
import inspect
import json
import os
from pathlib import Path
import sys
import time

root = Path(__file__).resolve().parent
port = json.loads((root/'NATIVE_SOURCE_PORT.json').read_bytes())
warm = Path(port['model_source'])
source = root/'source'
assert os.environ['CUDA_VISIBLE_DEVICES']==''
os.chdir(str(warm))
sys.path.insert(0,str(source))
sys.path.insert(1,str(warm))
import torch
torch.set_num_threads(1)
assert not torch.cuda.is_initialized()
from main_utils import parse_option
from train_dist_mod import TrainTester
from native_face_model_initialization import configure_native_face_model
from native_model_initialization import configure_native_model
from pcdet.config import cfg,cfg_from_yaml_file
assert Path(inspect.getfile(TrainTester)).resolve()==source/'train_dist_mod.py'
assert Path(inspect.getfile(configure_native_face_model)).resolve()==source/'native_face_model_initialization.py'
assert Path(inspect.getfile(configure_native_model)).resolve()==warm/'native_model_initialization.py'
assert Path(inspect.getfile(parse_option)).resolve()==warm/'main_utils.py'
cfg_from_yaml_file('wandb_config.yaml',cfg)
protocol=json.loads((root/'NORMAL_NATIVE_RUN_PROTOCOL.json').read_bytes())
identity=json.loads((root/'NORMAL_E0_IDENTITY.json').read_bytes())
payload=torch.load(identity['path'],map_location='cpu')
assert payload['epoch']==0 and payload['retained_metrics']['rows']==9508
assert payload['retained_metrics']['hits025']==5677 and payload['retained_metrics']['hits050']==4920
assert len(payload['model'])==1295
assert all(name.startswith('module.') for name in payload['model'])
expected={name[7:]:value for name,value in payload['model'].items()}
first_states=None
cases={}
for mode in ('source_conditioned','without_additional_source'):
    started=time.monotonic()
    spec=root/(mode+'.json')
    sys.argv=[str(source/'train_dist_mod.py')]+protocol['common_arguments']+['--native_init_spec',str(spec)]
    args=parse_option()
    assert args.dataset==['scanrefer'] and args.rng_seed==2027 and args.checkpoint_path is None
    torch.manual_seed(2027)
    tester=TrainTester.__new__(TrainTester)
    tester.model_cfg=copy.deepcopy(cfg)
    model=tester.get_model(args)
    states=model.state_dict()
    assert len(states)==1301
    migrated={}
    for name,value in expected.items():
        mapped=name
        if name.startswith('candidate_span_mixer.'):
            mapped='candidate_span_mixer.axis_prior.'+name[len('candidate_span_mixer.'):]
        migrated[mapped]=value
        assert states[mapped].shape==value.shape and states[mapped].dtype==value.dtype
        assert states[mapped].device.type=='cpu' and torch.equal(states[mapped],value)
    added=set(states)-set(migrated)
    assert len(added)==6 and all(name.startswith('candidate_span_mixer.face_residual.') for name in added)
    face=model.candidate_span_mixer.face_residual
    assert sum(p.numel() for p in face.parameters())==23425
    assert sum(p.numel() for p in model.candidate_span_mixer.parameters())==53218
    assert all(torch.count_nonzero(v)==0 for v in face[-1].state_dict().values())
    assert all(p.requires_grad for p in face.parameters())
    assert all(p.requires_grad for p in model.candidate_span_mixer.axis_prior.parameters())
    assert not any(p.requires_grad for p in model.text_encoder.parameters())
    assert model.native_training_architecture['full_state_tensors']==1301
    assert model.native_training_architecture['face_residual_mode']==mode
    assert model.candidate_span_mixer.use_source_evidence==(mode=='source_conditioned')
    if first_states is None:
        first_states={name:value.clone() for name,value in states.items()}
    else:
        assert set(first_states)==set(states)
        assert all(torch.equal(first_states[name],value) for name,value in states.items())
    assert not torch.cuda.is_initialized()
    cases[mode]=dict(full_model_states=1301,retained_E0_states_equal_after_explicit_prior_rename=1295,
        new_face_states=6,new_face_parameters=23425,combined_geometry_parameters=53218,
        face_output_zero=True,text_native_freeze_preserved=True,axis_and_face_trainable=True,
        architecture=model.native_training_architecture,elapsed_seconds=time.monotonic()-started)
    del states,model,tester,migrated
    gc.collect()
result=dict(status='ACTUAL_NATIVE_FACE_FACTORY_CPU_INITIALIZATION_PASS',cases=cases,
    initial_complete1301_states_equal_between_modes=True,
    torch_version=torch.__version__,CUDA_initialized=torch.cuda.is_initialized(),
    real_PV_factory=True,real_loader_rows=0,forwards=0,criterion_calls=0,optimizer_steps=0,
    saved_weight_files=0,full_checkpoint_recovery_verified=False,GPU_calls=0,
    current_training_queries=0,formal_accuracy=None,GPU_training_admission=False,
    full_goal_complete=False)
(root/'NATIVE_FACE_FACTORY_CPU_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
