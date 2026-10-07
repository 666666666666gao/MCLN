"""One frozen parent, independent face-center versus face-region geometry learning."""
import argparse
from collections import Counter
import importlib.util
import math
import copy
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import random
import sys
import time


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def main():
    begin = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--mode', choices=['preflight', 'train', 'initial_formal', 'formal'], required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_bytes())
    output = Path(spec['root'])
    runtime = Path(spec['runtime'])
    assert spec['batch_size'] == 8 and spec['seed'] == 2027
    assert spec['lr'] == 1e-5 and spec['weight_decay'] == .0005 and spec['clip_norm'] == .1
    assert spec['starting_hits']==[5598,4848] and spec['head_only']
    assert spec['use_geometry_evidence'] is True and spec['extra_geometry_weight']==1.0
    assert spec['auxiliary_target_mode']=='native_gt'
    assert spec['reference_mode']=='fused_mask' and spec['reference_keep_weight']==0.0
    assert spec['sampling_modes']==['face_center','face_region'] and spec['shared_frozen_parent_forward']
    assert spec['common_output_reset']==['output.weight','output.bias']
    assert sha(output/'mask_reference.py')==spec['mask_reference_sha256']
    assert sha(output/'selected_mask_reference_factory.py')==spec['selected_factory_sha256']
    assert sha(output/'face_region_box_refiner.py')==spec['sampler_sha256']
    assert sha(output/'face_support_model_factory.py')==spec['face_factory_sha256']
    assert sha(output/'paired_geometry_loop.py')==spec['paired_loop_sha256']
    for name, digest in spec['runner_files'].items():
        assert sha(Path(spec['helper_root']) / name)==digest,name
    manifest = json.loads(Path(spec['input_manifest']).read_bytes())
    dataset_source = Path(manifest['model_source'])
    assert sha(dataset_source / 'appearance_source_manifest.json') == manifest['source_manifest_sha256']
    for name, digest in json.loads((dataset_source / 'appearance_source_manifest.json').read_bytes())['files'].items():
        assert sha(dataset_source / name) == digest, name
    assert sha(manifest['split_protocol']) == manifest['split_protocol_sha256']
    partitions = json.loads(Path(manifest['split_protocol']).read_bytes())['row_ids']
    assert len(partitions['fit']) == 29778 and len(partitions['holdout']) == 6887
    assert not set(partitions['fit']).intersection(partitions['holdout'])
    env = json.loads((runtime / 'env_spec.json').read_bytes())
    env_sha = hashlib.sha256(json.dumps(env, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    assert env_sha == spec['env_spec_sha256']
    interface = json.loads(Path(spec['training_interface_receipt']).read_bytes())
    assert interface['status'] == 'pass' and interface['optimizer_steps'] == 2
    assert interface['env_spec_sha256'] == env_sha and interface['strict_cpu_restore']
    assert interface['direct_routing_verified'] and interface['added_state_tensors'] == 37
    assert interface['new_parameters'] == 923616
    assert interface['source_port_sha256'] == sha(spec['parent_source_port'])
    assert interface['module_sha256'] == spec['runner_files']['pvground_task_observation_query.py']
    model_source = Path(spec['model_source'])
    assert sha(spec['source_port']) == spec['source_port_sha256']
    port = json.loads(Path(spec['source_port']).read_bytes())
    assert port['boundary_evidence_readback'] and port['native_semantic_head_deferred']
    assert port['call_position'] == 'after native Mask generation'
    for name, digest in port['files'].items():
        assert sha(model_source / name) == digest, name
    official_weight = env['weight_dirs']['scanrefer']
    assert sha(official_weight['path']) == official_weight['sha256'] == spec['checkpoint_sha256']
    assert sha(spec['base_terminal']) == spec['base_terminal_sha256']
    assert sha(spec['selected_terminal'])==spec['selected_terminal_sha256']

    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset
    torch.cuda.reset_peak_memory_stats()

    def reset_rng():
        random.seed(spec['seed'])
        np.random.seed(spec['seed'])
        torch.manual_seed(spec['seed'])
        torch.cuda.manual_seed_all(spec['seed'])

    reset_rng()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.chdir(str(dataset_source))
    sys.path.insert(0, str(dataset_source))
    from src.joint_det_dataset import Joint3DDataset
    from scripts.scanrefer_data_contract import verify_scanrefer_superpoints
    assert Path(sys.modules['src.joint_det_dataset'].__file__).resolve() == dataset_source / 'src/joint_det_dataset.py'
    assert 'models' not in sys.modules
    os.chdir(str(model_source))
    sys.path.insert(0,spec['helper_root'])
    sys.path.insert(0,str(output))
    sys.path.insert(0,str(model_source))
    from main_utils import BaseTrainTester
    from prepare_data import DataProcessor
    from pcdet.config import cfg, cfg_from_yaml_file
    from face_support_model_factory import build_face_support_model
    from paired_geometry_loop import PairedGeometryRun
    from readback_preflight_checks import observed_readback_forward
    from pvground_semantic_assignment import semantic_assignment_correction
    from pvground_boundary_box_refiner import distribution_loss, BoundaryBoxRefiner
    from query_supported_geometry import query_supported_geometry_loss
    from whole_model_preflight_checks import optimizer_restore_exact
    imported = {name: str(Path(sys.modules[name].__file__).resolve()) for name in
        ('src.joint_det_dataset', 'models.pv_ground', 'models.losses', 'main_utils', 'prepare_data')}
    for name in ('models.pv_ground', 'models.losses', 'main_utils', 'prepare_data'):
        assert model_source in Path(imported[name]).parents
    write_json(output / 'imports.json', dict(files=imported, sha256={k: sha(v) for k, v in imported.items()}))
    cfg_from_yaml_file(str(runtime / 'PV-Ground/wandb_config.yaml'), cfg)
    selected_payload=torch.load(spec['selected_terminal'],map_location='cpu')
    assert selected_payload['step']==0 and selected_payload['zero_update_architecture']
    assert selected_payload['reference_mode']=='fused_mask' and not selected_payload['optimizer']['state']
    official_payload=torch.load(official_weight['path'],map_location='cpu')
    g_payload=torch.load(spec['base_terminal'],map_location='cpu')
    start_arm=dict(sampling_mode='face_center',step=0,state_delta=selected_payload['state_delta'])
    model,config,load=build_face_support_model(cfg,official_payload,g_payload,selected_payload,
        start_arm,manifest['data_root'])
    assert load['full_state_tensors']==1304 and not load['old_geometry_checkpoint_required']
    assert torch.count_nonzero(model.candidate_box_refiner.output.weight)==0
    assert torch.count_nonzero(model.candidate_box_refiner.output.bias)==0
    load.update(status='pass',time_cst=datetime.datetime.now().astimezone().isoformat(),
        spec_sha256=sha(args.spec),selected_terminal=spec['selected_terminal'],
        selected_terminal_sha256=spec['selected_terminal_sha256'],reference_keep_weight=spec['reference_keep_weight'],
        common_output_reset=spec['common_output_reset'],retained_hidden_state_tensors=8,reset_output_state_tensors=2)
    write_json(output/'load.json',load)
    model.cuda()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    geometry_head=model.candidate_box_refiner
    for parameter in geometry_head.parameters():
        parameter.requires_grad_(True)
    initial={name:value.detach().cpu().clone() for name,value in model.state_dict().items()}
    model.eval()
    load.update(frozen_geometry_provider=False,frozen_parent_and_R=True,geometry_head_only=True,
        fresh_readback_optimizer_required=False,fresh_geometry_optimizer_required=True)
    write_json(output/'load.json',load)
    training = copy.copy(config)
    training.frozen = False
    training.small_lr = False
    training.lr = spec['lr']
    training.lr_backbone = spec['lr']
    assert training.weight_decay == spec['weight_decay'] and training.clip_norm == spec['clip_norm']
    criterion, set_criterion = BaseTrainTester.get_criterion(training)
    processors = {mode: DataProcessor(cfg.DATA_PROCESSOR, np.asarray(cfg.DATA_CONFIG.POINT_CLOUD_RANGE), mode == 'train', 6)
                  for mode in ('train', 'eval')}
    trainable = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    expected_states=10
    expected_parameters=456102
    assert len(trainable)==expected_states and all(name.startswith('candidate_box_refiner.') for name in trainable)
    assert sum(parameter.numel() for parameter in trainable.values())==expected_parameters
    core_names=set(initial)-set(trainable)
    formal = args.mode in ('initial_formal','formal')
    assert spec['fit_passes'] == 1 and spec['updates'] == 3723
    assert spec['primary_mode'] == 'bbs' and spec['primary_threshold'] == .5
    verify_scanrefer_superpoints(manifest['data_root'], 'val' if formal else 'train',
                               manifest['superpoint_files']['val' if formal else 'train'])

    class FitDataset(Joint3DDataset):
        def _scene_graph_parse(self, annos):
            assert len(annos) == 36665
            actual = {'fit': [], 'holdout': []}
            for index, row in enumerate(annos):
                row['_local_training_id'] = index
                code = (manifest['split_salt'] + '\0' + row['scan_id'].split('_')[0]).encode()
                fold = int(hashlib.sha256(code).hexdigest()[:8], 16) % 5
                actual['holdout' if fold == 0 else 'fit'].append(index)
            assert actual == partitions
            super()._scene_graph_parse(annos)

        def _get_target_boxes(self, anno, scan):
            target = anno['target_id'][0] if isinstance(anno['target_id'], list) else anno['target_id']
            corners = scan.get_object_bbox(target).reshape(6).astype(np.float64)
            self.pre_jitter_root = np.concatenate(((corners[:3]+corners[3:])*.5,
                corners[3:]-corners[:3])).astype(np.float32)
            return super()._get_target_boxes(anno, scan)

        def __getitem__(self, index):
            result = super().__getitem__(index)
            result['pre_jitter_root_box'] = self.pre_jitter_root.copy()
            result['local_training_id'] = self.annos[index]['_local_training_id']
            assert np.isin(result['gt_masks'], [0, 1]).all()
            result['gt_masks'] = result['gt_masks'].astype(np.bool_)
            return result

    os.chdir(str(dataset_source))
    if formal:
        class FormalDataset(Joint3DDataset):
            def _scene_graph_parse(self, annos):
                assert len(annos) == 9508
                for index, row in enumerate(annos):
                    row['_local_training_id'] = index
                super()._scene_graph_parse(annos)

            def __getitem__(self, index):
                result = super().__getitem__(index)
                result['local_training_id'] = self.annos[index]['_local_training_id']
                assert np.isin(result['gt_masks'], [0, 1]).all()
                result['gt_masks'] = result['gt_masks'].astype(np.bool_)
                return result
        dataset = FormalDataset(dataset_dict={'scanrefer': 1}, test_dataset='scanrefer', split='val',
            data_path=manifest['data_root'], use_color=True, use_height=False, use_multiview=False,
            detect_intermediate=True, butd=True, butd_cls=False, butd_gt=False,
            augment_det=False, skip_missing_superpoints=True)
        assert len(dataset) == 9508
        partitions = {'holdout': list(range(9508))}
    else:
        dataset = FitDataset(dataset_dict={'scanrefer': 1}, test_dataset='scanrefer', split='train',
            data_path=manifest['data_root'], use_color=True, use_height=False, use_multiview=False,
            detect_intermediate=True, butd=True, butd_cls=False, butd_gt=False,
            augment_det=False, skip_missing_superpoints=True)
        assert len(dataset) == 36665
        physical = {part: {dataset.annos[index]['scan_id'].split('_')[0] for index in ids}
                    for part, ids in partitions.items()}
        assert not physical['fit'].intersection(physical['holdout'])
        dataset.augment = False
        reference = json.loads((Path(spec['reference_fixtures']) / 'receipt.json').read_bytes())
        reset_rng()
        for row in reference['rows']:
            sample = dataset[row['training_row_id']]
            checks = dict(point_clouds=sample['point_clouds'], det_boxes=sample['all_detected_boxes'],
                det_bbox_label_mask=sample['all_detected_bbox_label_mask'],
                det_class_ids=sample['all_detected_class_ids'], superpoint=sample['superpoint'].numpy())
            assert sample['utterances'] == row['text']
            for name, array in checks.items():
                assert hashlib.sha256(array.tobytes()).hexdigest() == row['tensor_sha256'][name], name
    evaluator_path=runtime/'PV-Ground/src/grounding_evaluator.py'
    evaluator_spec=importlib.util.spec_from_file_location('pvground_official_evaluator',str(evaluator_path))
    evaluator_module=importlib.util.module_from_spec(evaluator_spec)
    evaluator_spec.loader.exec_module(evaluator_module)
    GroundingEvaluator=evaluator_module.GroundingEvaluator
    assert sha(evaluator_path)==spec['native_evaluator_sha256']
    imported['native_evaluator']=str(evaluator_path)
    write_json(output/'imports.json',dict(files=imported,sha256={key:sha(value) for key,value in imported.items()}))
    from native_root_bbs import native_root_bbs
    selected=set(trainable)

    fit_saved=json.loads(Path(manifest['split_protocol']).read_bytes())['row_ids']['fit']
    partitions['fit_saved']=fit_saved

    def loader(part,shuffle):
        return DataLoader(Subset(dataset,partitions[part]),batch_size=8,shuffle=shuffle,num_workers=2,
            generator=torch.Generator().manual_seed(spec['seed']),pin_memory=True,drop_last=False)

    def prepare(batch,mode):
        voxel_rows=[processors[mode].forward(dict(points=pc.numpy().copy(),use_lead_xyz=True))
                    for pc in batch['point_clouds']]
        voxels=processors[mode].collate_batch(voxel_rows)
        size=len(batch['utterances'])
        assert np.array_equal(voxels['points'][:,1:].reshape(size,50000,6),batch['point_clouds'].numpy())
        batch={key:value.cuda(non_blocking=True) if torch.is_tensor(value) else value
               for key,value in batch.items()}
        inputs={key:torch.from_numpy(voxels[key]).float().cuda()
                for key in ('points','voxels','voxel_coords','voxel_num_points')}
        inputs.update(batch_size=size,text=batch['utterances'],superpoint=batch['superpoint'],train=False,
            det_boxes=batch['all_detected_boxes'],det_bbox_label_mask=batch['all_detected_bbox_label_mask'],
            det_class_ids=batch['all_detected_class_ids'])
        return inputs,batch

    def native_loss(predictions,batch):
        assert not set(predictions).intersection(batch)
        predictions.update(batch)
        native,predictions=criterion(predictions,6,set_criterion,
            query_points_obj_topk=training.query_points_obj_topk)
        assert torch.isfinite(native)
        return native,predictions

    def box_iou(boxes,truth):
        low=torch.maximum(boxes[...,:3]-boxes[...,3:]/2,truth[:3]-truth[3:]/2)
        high=torch.minimum(boxes[...,:3]+boxes[...,3:]/2,truth[:3]+truth[3:]/2)
        intersection=(high-low).clamp(min=0).prod(-1)
        iou=intersection/(boxes[...,3:].prod(-1)+truth[3:].prod()-intersection)
        assert torch.isfinite(iou).all()
        return iou

    runner=PairedGeometryRun(model,config,spec,output,args,initial,core_names,prepare,loader,dataset,
        partitions,native_loss,reset_rng,box_iou,GroundingEvaluator,sha,write_json,cfg,official_payload,
        g_payload,selected_payload,manifest['data_root'])
    runner.run(set_criterion,manifest,begin)


if __name__=='__main__':
    main()
