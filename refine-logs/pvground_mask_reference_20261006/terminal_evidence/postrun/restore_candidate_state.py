"""After actual closure, rebuild one declared candidate on CPU and restore Adam."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import sys


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--arm', choices=['native_reference', 'fused_mask_reference'], required=True)
    parser.add_argument('--stage', choices=['initial_formal', 'formal'], required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    assert str(root) == '/root/autodl-tmp/pvground_mask_reference_20261006'
    assert (root/'fit_controller.exit').read_text().strip() == '0'
    status = json.loads((root/'fit_status.json').read_bytes())
    assert status['status']=='complete' and status['protected_parents_exact']
    spec_path = root/(args.arm+'_spec.json')
    spec = json.loads(spec_path.read_bytes())
    runtime = Path(spec['runtime'])
    env = json.loads((runtime/'env_spec.json').read_bytes())
    assert hashlib.sha256(json.dumps(env,sort_keys=True,separators=(',', ':')).encode()).hexdigest()==spec['env_spec_sha256']
    official = env['weight_dirs']['scanrefer']
    assert sha(official['path'])==official['sha256']==spec['checkpoint_sha256']
    assert sha(spec['base_terminal'])==spec['base_terminal_sha256']
    assert sha(spec['geometry_terminal'])==spec['geometry_terminal_sha256']
    assert sha(root/'mask_reference.py')==spec['mask_reference_sha256']
    for name, digest in spec['runner_files'].items():
        assert sha(Path(spec['helper_root'])/name)==digest
    port = json.loads(Path(spec['source_port']).read_bytes())
    assert sha(spec['source_port'])==spec['source_port_sha256']
    model_source = Path(spec['model_source'])
    for name, digest in port['files'].items():
        assert sha(model_source/name)==digest
    manifest = json.loads(Path(spec['input_manifest']).read_bytes())
    path = root/args.arm/('initial.pth' if args.stage=='initial_formal' else 'terminal.pth')
    formal = json.loads((root/args.arm/args.stage/'receipt.json').read_bytes())
    assert formal['status']=='pass' and formal['rows']==formal['formal_rows']==9508
    destination = root/'postrun_restoration'/(args.arm+'_'+args.stage+'.json')
    assert not destination.exists()
    sys.path.insert(0,str(root/'postrun_restoration'))
    sys.path.insert(0,spec['helper_root'])
    sys.path.insert(0,str(root))
    sys.path.insert(0,str(model_source))
    os.chdir(str(model_source))
    import numpy as np
    import torch
    from pcdet.config import cfg, cfg_from_yaml_file
    from readback_model_factory import build_readback_model
    from mask_reference import install_mask_reference
    from selected_mask_reference_factory import build_selected_mask_reference_model
    from whole_model_preflight_checks import optimizer_restore_exact
    official_payload = torch.load(official['path'],map_location='cpu')
    g_payload = torch.load(spec['base_terminal'],map_location='cpu')
    prior_payload = torch.load(spec['geometry_terminal'],map_location='cpu')
    selected = torch.load(str(path),map_location='cpu')
    step = 0 if args.stage=='initial_formal' else 3723
    assert selected['step']==step and selected['spec_sha256']==sha(spec_path)
    assert selected['reference_mode']==spec['reference_mode']
    assert selected['common_output_reset']==['output.weight','output.bias']
    assert selected['retained_hidden_prior_updates']==11169
    for key in ('checkpoint_sha256','base_terminal_sha256','geometry_terminal_sha256','source_port_sha256','mask_reference_sha256'):
        assert selected[key]==spec[key]
    if args.stage=='initial_formal':
        assert selected['zero_update_architecture'] and selected['reset_output_total_updates']==0
        assert not selected['optimizer']['state']
        assert torch.count_nonzero(selected['state_delta']['candidate_box_refiner.output.weight'])==0
        assert torch.count_nonzero(selected['state_delta']['candidate_box_refiner.output.bias'])==0
        assert all(torch.equal(value,prior_payload['state_delta'][name])
            for name,value in selected['state_delta'].items() if not name.startswith('candidate_box_refiner.output.'))
    else:
        assert selected['head_only'] and selected['retained_hidden_total_updates']==14892
        assert selected['reset_output_total_updates']==3723
        assert selected['boundary_mode']=='distribution' and selected['support_arm']=='whole_range'
        assert selected['use_whole_range'] and selected['boundary_loss_weight']==1./7
        assert selected['extra_geometry_weight']==1. and selected['auxiliary_target_mode']=='native_gt'
        fit = json.loads((root/args.arm/'receipt.json').read_bytes())
        assert sha(path)==fit['terminal_sha256']
        assert len(selected['row_ids'])==len(set(selected['row_ids']))==29778
        assert len(selected['optimizer']['state'])==10
        assert all(int(value['step'])==3723 for value in selected['optimizer']['state'].values())
    cfg_from_yaml_file(str(runtime/'PV-Ground/wandb_config.yaml'),cfg)

    def seed():
        random.seed(2027)
        np.random.seed(2027)
        torch.manual_seed(2027)

    seed()
    direct, config, construction = build_selected_mask_reference_model(cfg,official_payload,g_payload,selected,manifest['data_root'])
    seed()
    historical, _, _, _ = build_readback_model(cfg,official_payload,g_payload,prior_payload,manifest['data_root'],True)
    install_mask_reference(historical,spec['reference_mode'])
    state = historical.state_dict()
    state.update(selected['state_delta'])
    historical.load_state_dict(state,strict=True)
    assert set(direct.state_dict())==set(historical.state_dict())
    assert all(torch.equal(value,historical.state_dict()[name]) for name,value in direct.state_dict().items())
    assert set(selected['state_delta'])=={name for name, parameter in direct.named_parameters() if parameter.requires_grad}
    parameters = tuple(parameter for parameter in direct.parameters() if parameter.requires_grad)
    optimizer = torch.optim.AdamW(parameters,lr=spec['lr'],weight_decay=spec['weight_decay'])
    optimizer.load_state_dict(selected['optimizer'])
    optimizer_check = optimizer_restore_exact(optimizer,selected['optimizer'])
    assert len(optimizer.state)==(0 if step==0 else 10)
    result=dict(status='pass',time_cst=datetime.datetime.now().astimezone().isoformat(),
        arm=args.arm,stage=args.stage,actual_step=step,checkpoint=str(path),checkpoint_sha256=sha(path),
        checkpoint_bytes=path.stat().st_size,zero_update_architecture=step==0,
        construction=construction,strict_CPU_geometry_restore=True,
        all_full_model_states_equal_to_original_construction=True,
        optimizer=optimizer_check,optimizer_steps=step,old_geometry_checkpoint_needed_by_direct_factory=False,
        old_geometry_checkpoint_used_only_for_this_comparison_witness=True,
        formal_metrics=formal['metrics'],GPU_forward_replayed=False,
        raw_Mask_support_recomputed=False,weights_created=0,weights_deleted=0,
        spec_sha256=sha(spec_path),factory_sha256=sha(root/'postrun_restoration/selected_mask_reference_factory.py'),
        checker_sha256=sha(__file__))
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
