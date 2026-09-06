import copy, datetime, gc, hashlib, json, os, sys
from pathlib import Path
root = Path(sys.argv[1])
source = root / 'model_source'
os.chdir(str(source))
sys.path.insert(0, str(source))
assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
import torch
from main_utils import BaseTrainTester, load_checkpoint, parse_option, prepare_source_moe_gate_checkpoint_config
from train_dist_mod import TrainTester
from models.candidate_local_visual_training import local_visual_state_keys
torch.set_num_threads(1)
assert not torch.cuda.is_available()
expected = json.loads((root / 'expected.json').read_text())
hasher = hashlib.sha256()
with Path(expected['checkpoint']).open('rb') as stream:
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
        hasher.update(block)
assert hasher.hexdigest() == expected['checkpoint_sha256']
payload = torch.load(expected['checkpoint'], map_location='cpu')
assert payload['evaluation_only'] and 'optimizer' not in payload and 'scheduler' not in payload
assert all(name.startswith('module.') for name in payload['model'])
contract = json.loads((root / 'nr_contract.json').read_text())
results = {}
for dataset, rows in [('nr3d', 7899), ('sr3d', 17726)]:
    argv = list(contract['eval_argv'])
    for key, value in [('--dataset', dataset), ('--test_dataset', dataset), ('--expected_eval_sample_count', str(rows))]:
        argv[argv.index(key) + 1] = value
    sys.argv = ['native-local-cpu-preparation'] + argv + ['--use_candidate_local_visual']
    args = prepare_source_moe_gate_checkpoint_config(parse_option())
    assert args.model == 'MCLN' and args.num_decoder_layers == 6 and args.use_color
    assert args.butd_cls and not args.butd and not args.butd_gt
    assert args.eval_use_selector_choice_scores and args.use_source_choice_selector
    assert not args.eval_use_rec_reranker_scores and not args.eval_use_rec_geometry_reranker_scores
    args.eval = False
    args.checkpoint_path = expected['checkpoint']
    args.model_only_initialization = True
    args.checkpoint_start_epoch = 1
    args.lr = args.lr_backbone = args.source_choice_selector_lr = 1e-6
    args.candidate_local_visual_lr = 1e-4
    model = TrainTester.get_model(args).cpu().eval()
    wrapped = torch.nn.Module()
    wrapped.module = model
    local_keys = local_visual_state_keys(wrapped.state_dict())
    initial = {name: value.clone() for name, value in wrapped.state_dict().items() if name in local_keys}
    optimizer = BaseTrainTester.get_optimizer(args, wrapped)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2], gamma=.1)
    before_optimizer = copy.deepcopy(optimizer.state_dict())
    before_scheduler = copy.deepcopy(scheduler.state_dict())
    load_checkpoint(args, wrapped, optimizer, scheduler)
    assert args.start_epoch == 1
    assert optimizer.state_dict() == before_optimizer and scheduler.state_dict() == before_scheduler
    state = wrapped.state_dict()
    assert len(payload['model']) == 1144 and len(state) == 1154 and len(local_keys) == 10
    assert set(state) == set(payload['model']) | local_keys
    assert all(torch.equal(state[name], value) for name, value in payload['model'].items())
    assert all(torch.equal(state[name], value) for name, value in initial.items())
    reader = model.decoder[-1].local_visual
    assert torch.count_nonzero(reader.output_projection.weight) == 0
    assert torch.count_nonzero(reader.output_projection.bias) == 0
    groups = optimizer.param_groups
    assert [group['name'] for group in groups] == ['decoder', 'backbone', 'mask_head', 'selector', 'candidate_local_visual']
    assert len(groups[-1]['params']) == 10 and groups[-1]['lr'] == 1e-4
    all_ids = [id(parameter) for group in groups for parameter in group['params']]
    assert len(all_ids) == len(set(all_ids))
    assert set(all_ids) == {id(parameter) for parameter in wrapped.parameters() if parameter.requires_grad}
    results[dataset] = {'model_tensors': len(state), 'pretrained_tensors_equal': 1144,
        'new_tensors_unchanged_from_initialization': len(local_keys),
        'model_parameter_count': sum(parameter.numel() for parameter in model.parameters()),
        'local_parameter_count': sum(parameter.numel() for parameter in reader.parameters()),
        'native_weights_only_loader_pass': True, 'fresh_optimizer_and_scheduler': True,
        'optimizer_groups': [{'name': group['name'], 'tensors': len(group['params']), 'lr': group['lr']} for group in groups],
        'native_filter_non_gt_boxes': args.butd_cls, 'native_selector_output': args.eval_use_selector_choice_scores,
        'train_start_epoch': args.start_epoch, 'native_model_updates': 0}
    del model, wrapped, state, initial, optimizer, scheduler, groups, reader
    gc.collect()
assert not torch.cuda.is_initialized()
result = {'schema': 'mcln-candidate-local-native-preparation-v1', 'status': 'pass',
    'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    'source_manifest_sha256': hashlib.sha256((source / 'native_source_manifest.json').read_bytes()).hexdigest(),
    'nr_checkpoint_sha256': expected['checkpoint_sha256'], 'protocols': results,
    'gpu_forwards': 0, 'real_dataset_rows': 0, 'native_model_optimizer_updates': 0,
    'checkpoint_writes': 0, 'formal_rows': 0, 'sr_protected_checkpoint_restored': False,
    'limits': 'Original-environment CPU native model/loading/optimizer integration only; neither Nr/Sr data preflight nor training nor benchmark performance. Scan promotion remains required.'}
with (root / 'receipt.json').open('x') as stream:
    json.dump(result, stream, indent=2, sort_keys=True)
print(json.dumps(result), flush=True)
