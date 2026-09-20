"""Real training-batch equality before task-read optimization, with native loss."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8388608), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    assert sha(spec['checkpoint']) == spec['checkpoint_sha256']
    for name, digest in spec['source_files'].items():
        assert sha(Path(spec['source']) / name) == digest, name
    os.chdir(spec['source'])
    sys.path.insert(0, spec['source'])
    sys.path.insert(1, str(Path(spec['source']) / 'pointnet2'))
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset
    from models import EG
    from models.eg3dvg_task_read import install_task_read
    from main_utils import BaseTrainTester
    from src.joint_det_dataset import Joint3DDataset
    random.seed(2027)
    np.random.seed(2027)
    torch.manual_seed(2027)
    torch.cuda.manual_seed_all(2027)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = EG(num_class=256, num_obj_class=485, input_feature_dim=3, num_queries=256,
               num_decoder_layers=6, self_position_embedding='loc_learned',
               contrastive_align_loss=True, butd=True, pointnet_ckpt=None,
               data_path=spec['data_root'], self_attend=True)
    state = {k[7:]: v for k, v in torch.load(spec['checkpoint'], map_location='cpu')['model'].items()}
    model.load_state_dict(state, strict=True)
    model.cuda().train()
    native_args = SimpleNamespace(num_decoder_layers=6, query_points_obj_topk=4,
                                  use_contrastive_align=True, use_soft_token_loss=True)
    criterion, set_criterion = BaseTrainTester.get_criterion(native_args)
    set_criterion.cuda()
    dataset = Joint3DDataset(dataset_dict={'scanrefer': 1, 'scannet': 10}, test_dataset='scanrefer',
                            split='train', data_path=spec['data_root'], use_color=True,
                            detect_intermediate=True, butd=True, augment_det=True)
    batch = next(iter(DataLoader(Subset(dataset, spec['preflight_indices'][:8]),
                                batch_size=8, num_workers=0, shuffle=False)))
    batch = BaseTrainTester._to_gpu(batch)
    inputs = dict(point_clouds=batch['point_clouds'].float(), text=batch['utterances'],
                  det_boxes=batch['all_detected_boxes'], det_bbox_label_mask=batch['all_detected_bbox_label_mask'],
                  det_class_ids=batch['all_detected_class_ids'], superpoint=batch['superpoint'])
    negative = dict(inputs, text=batch['negative_utterances'])
    keys = ['last_center', 'last_pred_size', 'last_sem_cls_scores', 'last_proj_queries', 'last_pred_masks']

    def forward():
        with torch.no_grad():
            end = model(inputs)
            other = model(negative)
            selected = {key: ([v.cpu().clone() for v in end[key]] if isinstance(end[key], list)
                              else [end[key].cpu().clone()]) for key in keys}
            end.update(batch)
            loss, end = BaseTrainTester._compute_loss(end, criterion, set_criterion, native_args)
            loss, end = BaseTrainTester._compute_negative_loss(loss, end, other)
            return selected, float(loss)

    cpu_before, cuda_before = torch.get_rng_state(), torch.cuda.get_rng_state()
    reference, reference_loss = forward()
    cpu_after, cuda_after = torch.get_rng_state(), torch.cuda.get_rng_state()
    buffers_after = {k: v.cpu().clone() for k, v in model.named_buffers()}
    model.load_state_dict(state, strict=True)
    install_task_read(model)
    model.cuda()
    torch.set_rng_state(cpu_before)
    torch.cuda.set_rng_state(cuda_before)
    actual, actual_loss = forward()
    errors = {key: max(float((a - b).abs().max()) for a, b in zip(reference[key], actual[key])) for key in keys}
    assert all(value == 0 for value in errors.values()), errors
    assert reference_loss == actual_loss, (reference_loss, actual_loss)
    assert torch.equal(cpu_after, torch.get_rng_state())
    assert torch.equal(cuda_after, torch.cuda.get_rng_state())
    assert all(torch.equal(v.cpu(), buffers_after[k]) for k, v in model.named_buffers())
    report = {'status': 'pass', 'rows': 8, 'model_forwards': 4, 'optimizer_steps': 0,
              'max_absolute_errors': errors, 'native_loss': reference_loss, 'task_loss': actual_loss,
              'rng_equal': True, 'buffers_equal': True, 'spec_sha256': sha(args.spec),
              'checkpoint_sha256': spec['checkpoint_sha256'], 'accuracy_result': False}
    (args.spec.parent / 'initial_equality.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
