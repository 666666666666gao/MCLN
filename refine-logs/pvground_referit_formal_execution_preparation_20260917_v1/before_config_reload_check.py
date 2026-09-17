"""Check REC boundaries and real CPU parent reconstruction; not terminal or GPU evaluation."""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time
import unittest


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):digest.update(block)
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--spec',type=Path,required=True)
    args=parser.parse_args();root=args.spec.parent;spec=json.loads(args.spec.read_bytes());start=time.time()
    for name,digest in spec['files'].items():assert sha(root/name)==digest,name
    for name in spec['files']:
        if name.endswith('.py'):compile((root/name).read_bytes(),name,'exec')
    module_spec=importlib.util.spec_from_file_location('referit_formal',root/'evaluate.py')
    evaluator=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(evaluator)
    boundaries=[]
    for name,a,b in [('nr3d',4726,4059),('sr3d',12139,10335)]:
        for hit25,hit50,expected in [(a,b,True),(a-1,b,False),(a,b-1,False),(a+1,b+1,True)]:
            value=dict(rec_hits25=hit25,rec_hits50=hit50,mask_hits25=0,mask_hits50=0,mask_miou=0.)
            result=evaluator.promotion_check(value,name)
            assert result['rec_target_pass']==expected and result['mask_gate'] is False
            boundaries.append(dict(dataset=name,hits=[hit25,hit50],pass_expected=expected))
    runtime=Path(spec['runtime']);source=Path(spec['model_source'])
    environment=json.loads((runtime/'env_spec.json').read_bytes())
    assert hashlib.sha256(json.dumps(environment,sort_keys=True,separators=(',',':')).encode()).hexdigest()==spec['env_spec_sha256']
    assert sha(spec['source_port'])==spec['source_port_sha256']
    for name,digest in json.loads(Path(spec['source_port']).read_bytes())['files'].items():assert sha(source/name)==digest,name
    os.chdir(str(source));sys.path[:0]=[str(root),str(source)]
    import numpy as np
    import torch
    from models.pv_ground import PVGround
    from pcdet.config import cfg,cfg_from_yaml_file
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    cfg_from_yaml_file(str(runtime/'PV-Ground/wandb_config.yaml'),cfg)
    checks={}
    for dataset,entry in spec['datasets'].items():
        random.seed(2027);np.random.seed(2027);torch.manual_seed(2027)
        assert sha(entry['parent'])==entry['parent_sha256']
        saved=torch.load(entry['parent'],map_location='cpu');config=saved['config']
        assert not config.butd and config.butd_cls and not config.butd_gt
        assert all(name.startswith('module.') for name in saved['model'])
        parent={name[7:]:value for name,value in saved['model'].items()};del saved
        model=PVGround(cfg,num_class=256,num_queries=256,num_decoder_layers=6,
            self_position_embedding=config.self_position_embedding,contrastive_align_loss=True,butd=True,
            pointnet_ckpt=None,data_path=spec['data_root'],self_attend=config.self_attend)
        initial=evaluator.expanded_parent_state(model,parent)
        assert len(initial)==1272
        # Exercise actual strict restore helper with an in-memory initial-state delta.
        # This is not an actual trained terminal checkpoint.
        names={name for name,p in model.named_parameters() if p.requires_grad}
        names|={name for name,_ in model.named_buffers() if name in initial}
        fixture={name:initial[name] for name in names}
        restored=evaluator.terminal_state(model,initial,fixture)
        model.load_state_dict(restored,strict=True)
        assert all(torch.equal(value,initial[name]) for name,value in model.state_dict().items())
        missing=dict(fixture);missing.pop('decoder.5.source_query_read.task_queries')
        with unittest.TestCase().assertRaises(AssertionError):
            evaluator.terminal_state(model,initial,missing)
        assert not torch.cuda.is_initialized()
        checks[dataset]=dict(parent_sha256=entry['parent_sha256'],state_tensors=1272,
            actual_cpu_parent_restore=True,initial_state_fixture_restore=True,missing_task_state_rejected=True,
            actual_terminal_restore=False,model_forwards=0)
        del model,parent,initial,fixture,restored,missing
    receipt=dict(status='cpu_preparation_pass',time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        datasets=checks,boundary_cases=boundaries,formal_rows=0,model_forwards=0,optimizer_steps=0,
        new_checkpoints=0,torch_cuda_initialized=False,script_sha256=sha(__file__),
        evaluator_sha256=sha(root/'evaluate.py'),elapsed_seconds=time.time()-start,
        scope='actual CPU parents plus initial-state fixture and REC boundary checks; no trained terminal or formal inference')
    (root/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
