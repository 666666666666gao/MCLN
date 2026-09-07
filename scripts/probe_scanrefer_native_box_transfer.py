"""Actual fit-input gradient and two-step checks for native box target transfer."""
import argparse
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import sys


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    option = parser.parse_args()
    directory = option.manifest.parent
    manifest = json.loads(option.manifest.read_text())
    assert manifest['schema'] == 'mcln-native-box-transfer-probe-v1'
    source = Path(manifest['model_source'])
    assert sha(source / 'local_visual_source_manifest.json') == manifest['source_manifest_sha256']
    for name, digest in json.loads((source / 'local_visual_source_manifest.json').read_text())['files'].items():
        assert sha(source / name) == digest, name
    for name, digest in manifest['files'].items():
        assert sha(directory / name) == digest, name
    for name, item in manifest['artifacts'].items():
        assert sha(item['path']) == item['sha256'], name
    assert sha(manifest['teacher_audit_receipt']) == manifest['teacher_audit_receipt_sha256']
    teacher_audit = json.loads(Path(manifest['teacher_audit_receipt']).read_text())
    assert teacher_audit['data_root'] == manifest['data_root'] and teacher_audit['status'] == 'pass'
    assert sha(manifest['split_protocol']) == manifest['split_protocol_sha256']
    split = json.loads(Path(manifest['split_protocol']).read_text())
    selected_ids = split['selected_ids']
    assert len(selected_ids) == 16 and set(selected_ids).issubset(split['row_ids']['fit'])
    assert manifest['learning_rate'] == 1e-6 and manifest['auxiliary_weight'] == 1.
    os.chdir(str(source))
    sys.path.insert(0, str(source))
    import numpy as np
    import torch
    import scripts
    scripts.__path__ = [str(directory / 'scripts'), str(source / 'scripts')]
    from main_utils import parse_option
    from train_dist_mod import TrainTester
    from src.joint_det_dataset import Joint3DDataset
    from scripts.run_frozen_v99_pareto_contextual_official import build_authoritative_command
    from scripts.scanrefer_data_contract import set_scanrefer_data_root, verify_scanrefer_superpoints
    from scripts.scanrefer_joint_readout import JointRecReadout
    from scripts.native_teacher_box_transfer import BOX_PARAMETER_PREFIXES, native_teacher_box_loss

    random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    command = set_scanrefer_data_root(build_authoritative_command(directory / 'unused_output'), manifest['data_root'])
    verified_data = verify_scanrefer_superpoints(manifest['data_root'], 'train', manifest['train_superpoint_files'])
    sys.argv = [sys.argv[0]] + command[command.index('train_dist_mod.py') + 1:]
    args = parse_option()
    assert args.dataset == ['scanrefer'] and args.butd and not args.butd_cls and not args.butd_gt
    initial = {name[7:]: value for name, value in torch.load(args.checkpoint_path, map_location='cpu')['model'].items()}
    model = TrainTester.get_model(args).cuda().eval()
    model.load_state_dict(initial, strict=True)
    assert model.decoder[-1].local_visual is None and model.query_mask_fusion_calibrator is None
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(BOX_PARAMETER_PREFIXES))
    parameters = {name: value for name, value in model.named_parameters() if value.requires_grad}
    assert len(parameters) == 16
    teacher = copy.deepcopy(model).requires_grad_(False)
    artifacts = {name: torch.load(item['path'], map_location='cpu') for name, item in manifest['artifacts'].items() if name != 'backbone'}
    readout = JointRecReadout(artifacts).cuda().eval().requires_grad_(False)
    readout_state = {name: value.detach().cpu().clone() for name, value in readout.state_dict().items()}
    criterion, set_criterion = TrainTester.get_criterion(args)

    class ProbeDataset(Joint3DDataset):
        def _scene_graph_parse(self, annos):
            assert len(annos) == 36665
            for index, row in enumerate(annos):
                row['_box_transfer_id'] = index
            selected_scenes = {annos[i]['scan_id'] for i in selected_ids}
            annos[:] = [row for row in annos if row['scan_id'] in selected_scenes]
            super()._scene_graph_parse(annos)

    dataset = ProbeDataset(dataset_dict={'scanrefer': 1}, test_dataset='scanrefer', split='train',
        data_path=args.data_root, use_color=args.use_color, use_height=args.use_height,
        use_multiview=args.use_multiview, detect_intermediate=args.detect_intermediate,
        butd=args.butd, butd_gt=args.butd_gt, butd_cls=args.butd_cls,
        augment_det=False, skip_missing_superpoints=args.skip_missing_superpoints)
    by_id = {row['_box_transfer_id']: row for row in dataset.annos}
    dataset.annos = [by_id[i] for i in selected_ids]
    dataset.augment = False
    loader = torch.utils.data.DataLoader(dataset, batch_size=12, shuffle=False, num_workers=0,
                                        generator=torch.Generator().manual_seed(0))
    unchanged_keys = ['last_sem_cls_scores', 'last_proj_queries', 'proj_tokens', 'last_pred_masks', 'sp_last_pred_masks']

    def assert_outputs_same(first, second):
        for key in unchanged_keys:
            a, b = first[key], second[key]
            if torch.is_tensor(a):
                assert torch.equal(a, b), key
            else:
                assert len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b)), key

    def losses(candidate, inputs, batch, teacher_boxes):
        outputs = candidate(inputs)
        boxes = torch.cat([outputs['last_center'], outputs['last_pred_size']], dim=-1)
        roots = torch.cat([batch['center_label'][:, 0], batch['size_gts'][:, 0]], dim=-1)
        captured = []
        def capture(module, arguments, result):
            if torch.equal(arguments[0]['pred_boxes'], boxes):
                assert all(torch.equal(target['boxes'][0], roots[i]) for i, target in enumerate(arguments[1]))
                captured.append(result)
        handle = set_criterion.matcher.register_forward_hook(capture)
        outputs.update(batch)
        native, outputs = TrainTester._compute_loss(outputs, criterion, set_criterion, args)
        handle.remove()
        assert len(captured) == 1
        auxiliary, stats = native_teacher_box_loss(boxes, teacher_boxes, roots, captured[0])
        assert torch.isfinite(native) and torch.isfinite(auxiliary)
        return outputs, native, auxiliary, stats

    observations = []
    for batch_index, raw in enumerate(loader):
        batch = TrainTester._to_gpu(raw)
        inputs = TrainTester._get_inputs(batch); inputs['train'] = False
        with torch.no_grad():
            teacher_outputs = teacher(inputs)
            runtime = readout(teacher_outputs, inputs)['runtime']
            selected = runtime['rec_geometry_scores'].masked_fill(~runtime['rec_geometry_valid_mask'], -float('inf')).argmax(dim=1)
            teacher_boxes = runtime['rec_geometry_boxes'][torch.arange(len(selected), device=selected.device), selected].detach()
        outputs, native, auxiliary, stats = losses(model, inputs, batch, teacher_boxes)
        assert_outputs_same(outputs, teacher_outputs)
        assert torch.equal(outputs['last_center'], teacher_outputs['last_center'])
        assert torch.equal(outputs['last_pred_size'], teacher_outputs['last_pred_size'])
        first = torch.autograd.grad(native, tuple(parameters.values()), retain_graph=True)
        second = torch.autograd.grad(auxiliary, tuple(parameters.values()))
        assert all(torch.isfinite(x).all() for x in first + second)
        n1 = sum(float(x.square().sum()) for x in first)
        n2 = sum(float(x.square().sum()) for x in second)
        dot = sum(float((x*y).sum()) for x,y in zip(first,second))
        record = {'batch':batch_index,'row_ids':selected_ids[batch_index*12:batch_index*12+len(raw['scan_ids'])],
            'scan_ids':raw['scan_ids'],'point_sha256':[hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest() for x in inputs['point_clouds']],
            'native_loss':float(native),'teacher_box_loss':float(auxiliary),'native_gradient_norm':n1**.5,'teacher_gradient_norm':n2**.5,
            'gradient_dot':dot,'teacher_selected_flat':selected.cpu().tolist(),
            'target_stats':{name:value.cpu().tolist() for name,value in stats.items()}}
        observations.append(record)
        print('NATIVE BOX TRANSFER GRADIENT',json.dumps({k:record[k] for k in ['batch','native_loss','teacher_box_loss','native_gradient_norm','teacher_gradient_norm']}),flush=True)
        del outputs,native,auxiliary,stats,first,second,runtime
    assert len(observations)==2 and sum(sum(r['target_stats']['eligible']) for r in observations)>0
    assert all(torch.equal(value.cpu(),initial[name]) for name,value in model.state_dict().items())
    updates = {}
    for arm in ['gt_only','gt_teacher_box']:
        student = copy.deepcopy(model)
        trainable = [value for value in student.parameters() if value.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=manifest['learning_rate'], weight_decay=.0005)
        step_rows = []
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            outputs,native,auxiliary,stats = losses(student,inputs,batch,teacher_boxes)
            assert_outputs_same(outputs,teacher_outputs)
            objective = native + (auxiliary if arm=='gt_teacher_box' else auxiliary*0.)
            objective.backward()
            assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in trainable)
            norm = torch.nn.utils.clip_grad_norm_(trainable,.1)
            optimizer.step()
            step_rows.append({'step':step+1,'native_loss':float(native),'auxiliary_loss':float(auxiliary),'gradient_norm':float(norm),'eligible':int(stats['eligible'].sum()),'root_query_indices':stats['student_query_indices'].cpu().tolist()})
            del outputs,native,auxiliary,stats,objective
        with torch.no_grad():
            after=student(inputs)
            assert_outputs_same(after,teacher_outputs)
            center_change=float((after['last_center']-teacher_outputs['last_center']).abs().max())
            size_change=float((after['last_pred_size']-teacher_outputs['last_pred_size']).abs().max())
            assert center_change>0 and size_change>0
        changed=[]
        for name,value in student.state_dict().items():
            same=torch.equal(value.cpu(),initial[name])
            if name not in parameters:assert same,(arm,name)
            elif not same:changed.append(name)
        assert changed and all(float(state['step'])==2 for state in optimizer.state.values())
        updates[arm]={'steps':step_rows,'changed_tensors':changed,'max_center_change':center_change,'max_size_change':size_change,'frozen_outputs_equal':unchanged_keys}
        del student,optimizer,trainable,after
    assert all(torch.equal(value.cpu(),initial[name]) for name,value in teacher.state_dict().items())
    assert all(torch.equal(value.cpu(),readout_state[name]) for name,value in readout.state_dict().items())
    receipt={'schema':'mcln-native-box-transfer-probe-result-v1','status':'pass','time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        'input_manifest_sha256':sha(option.manifest),'rows':16,'observations':observations,'updates':updates,
        'trainable_tensors':list(parameters),'parameter_count':sum(p.numel() for p in parameters.values()),
        'teacher_and_readouts_unchanged':True,'disposable_optimizer_steps_per_arm':2,'checkpoint_writes':0,'formal_rows':0,
        'data_inputs':verified_data,'quality_result':False,'max_gpu_mib':torch.cuda.max_memory_allocated()/1024**2}
    write(directory/'receipt.json',receipt)
    print('NATIVE BOX TRANSFER PROBE COMPLETE',json.dumps({'status':'pass','trainable_tensors':len(parameters),'rows':16,'quality_result':False}),flush=True)


if __name__=='__main__':
    main()
