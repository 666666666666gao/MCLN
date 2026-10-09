"""Wire the prepared native model into PV's ordinary trainer and full resume."""
import ast
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
source = root / 'source'
assert not (root / 'ENTRY_PREPARATION.json').exists()
snapshots = root / 'entry_r0_snapshots'
assert not snapshots.exists()
snapshots.mkdir()
for name in ('main_utils.py', 'train_dist_mod.py'):
    (snapshots / name).write_bytes((source / name).read_bytes())

def replace_once(text, old, new):
    assert text.count(old) == 1, old[:80]
    return text.replace(old, new, 1)

train_path = source / 'train_dist_mod.py'
train = train_path.read_text(encoding='utf-8')
train = replace_once(train, '        self.data_processor = DataProcessor(model_cfg.DATA_PROCESSOR, pc_range, training, num_point_features)',
    "        self.data_processors = {mode: DataProcessor(model_cfg.DATA_PROCESSOR, pc_range, mode == 'train', num_point_features)\n"
    "                                for mode in ('train', 'eval')}")
train = replace_once(train, '        return model\n',
    '        from native_model_initialization import configure_native_model\n'
    '        return configure_native_model(model, args.native_init_spec, initialize_weights=not bool(args.checkpoint_path))\n')
train = replace_once(train, '    def _get_inputs(self, batch_data):',
    "    def _get_inputs(self, batch_data, training):\n"
    "        processor = self.data_processors['train' if training else 'eval']")
train = train.replace('self.data_processor.forward(batch_dict)', 'processor.forward(batch_dict)')
train = train.replace('self.data_processor.collate_batch(voxel_batch)', 'processor.collate_batch(voxel_batch)')
train = replace_once(train, '    opt = parse_option()    ', '''    opt = parse_option()
    assert opt.model == 'PVGround' and opt.dataset == ['scanrefer'] and opt.test_dataset == 'scanrefer'
    assert opt.rng_seed == 2027 and opt.use_color and not opt.use_height and not opt.use_multiview
    assert opt.butd and not opt.butd_cls and not opt.butd_gt and not opt.joint_det
    assert opt.use_soft_token_loss and opt.use_contrastive_align
    assert not opt.frozen and not opt.small_lr and opt.batch_size == 8
    random.seed(opt.rng_seed)
    np.random.seed(opt.rng_seed)
    torch.manual_seed(opt.rng_seed)
    torch.cuda.manual_seed_all(opt.rng_seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False''')
train = replace_once(train, '    torch.backends.cudnn.benchmark = True', '    torch.backends.cudnn.benchmark = False')
train_path.write_text(train, encoding='utf-8')

main_path = source / 'main_utils.py'
main = main_path.read_text(encoding='utf-8')
main = replace_once(main, '    # Model\n',
    "    parser.add_argument('--native_init_spec', required=True, help='Explicit native architecture and initialization manifest')\n"
    "    parser.add_argument('--lr_modules', type=float, default=1e-5)\n"
    '    # Model\n')
main = replace_once(main, '    args, _ = parser.parse_known_args()', '    args = parser.parse_args()')
main = replace_once(main, '        g.manual_seed(0)', '        g.manual_seed(args.rng_seed)')
main = replace_once(main, '            train_sampler = DistributedSampler(train_dataset)',
    '            train_sampler = DistributedSampler(train_dataset, seed=args.rng_seed)')
main = replace_once(main, '    def _get_inputs(batch_data):', '    def _get_inputs(batch_data, training):')
main = replace_once(main, '            inputs = self._get_inputs(batch_data)', '            inputs = self._get_inputs(batch_data, training=True)')
main = replace_once(main, '        inputs = self._get_inputs(batch_data)', '        inputs = self._get_inputs(batch_data, training=False)')

start = main.index('    @staticmethod\n    def get_optimizer(')
end = main.index('    # BRIEF main training/testing', start)
optimizer = '''    @staticmethod
    def get_optimizer(args, model):
        assert not args.frozen and not args.small_lr
        modules = ('candidate_support_corrector.', 'candidate_span_mixer.')
        named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
        core = [parameter for name, parameter in named
                if not name.startswith(modules) and 'backbone_net' not in name and 'text_encoder' not in name]
        backbone = [parameter for name, parameter in named if 'backbone_net' in name]
        text = [parameter for name, parameter in named if 'text_encoder' in name]
        additions = [parameter for name, parameter in named if name.startswith(modules)]
        groups = [dict(params=core, lr=args.lr), dict(params=backbone, lr=args.lr_backbone),
                  dict(params=text, lr=args.text_encoder_lr), dict(params=additions, lr=args.lr_modules)]
        assert core and backbone and additions and not text
        grouped = [parameter for group in groups for parameter in group['params']]
        assert len(grouped) == len({id(parameter) for parameter in grouped}) == len(named)
        return optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)


'''
main = main[:start] + optimizer + main[end:]
start = main.index('def load_checkpoint(')
end = main.index('# BRIEF save model.', start)
loader = '''def load_checkpoint(args, model, optimizer, scheduler):
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    assert checkpoint['architecture'] == model.module.native_training_architecture
    model.load_state_dict(checkpoint['model'], strict=True)
    if not args.eval:
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        args.start_epoch = int(checkpoint['epoch']) + 1
        random.setstate(checkpoint['rng']['python'])
        np.random.set_state(checkpoint['rng']['numpy'])
        torch.set_rng_state(checkpoint['rng']['torch'])
        torch.cuda.set_rng_state_all(checkpoint['rng']['cuda'])
    del checkpoint
    torch.cuda.empty_cache()


'''
main = main[:start] + loader + main[end:]
main = replace_once(main, "            'config': args,", "            'config': args,\n"
    "            'architecture': model.module.native_training_architecture,\n"
    "            'rng': {'python': random.getstate(), 'numpy': np.random.get_state(),\n"
    "                    'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()},")
main = replace_once(main, "        save_checkpoint(args, 'last', model, optimizer, scheduler, True)",
    '        save_checkpoint(args, epoch, model, optimizer, scheduler, True)')
main = replace_once(main, "        saved_path = os.path.join(args.log_dir, 'ckpt_epoch_last.pth')",
    "        saved_path = os.path.join(args.log_dir, f'ckpt_epoch_{epoch}.pth')")
main_path.write_text(main, encoding='utf-8')

g_root = Path('C:/Users/gb/.codex_pvground_cs_20261002/refine-logs/pvground_scanrefer_finetune_20260918_semantic_assignment_v1')
dependencies = {}
for name in ('pvground_task_observation_query.py', 'pvground_observation_query.py', 'pvground_source_query.py'):
    path = g_root / name
    raw = path.read_bytes()
    (source / name).write_bytes(raw)
    dependencies[str(path)] = hashlib.sha256(raw).hexdigest()
score = root.parent / 'pvground_extremal_span_evidence_20261009/runner_v1/source_review/snapshots/033_native_root_bbs.py'
assert hashlib.sha256(score.read_bytes()).hexdigest() == 'b0357bd54e8753672ee31d3056e50c1d53a854d7f0c71cd01a7bc8cecf422c8b'
(source / 'native_root_bbs.py').write_bytes(score.read_bytes())
dependencies[str(score)] = hashlib.sha256(score.read_bytes()).hexdigest()
files = {}
for path in source.rglob('*.py'):
    ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    files[path.relative_to(source).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
receipt = dict(status='NORMAL_PV_TRAIN_ENTRY_SOURCE_PREPARED_NOT_REVIEWED_OR_DEPLOYED',
    source_files=files, dependency_sources=dependencies, native_model_class=True,
    native_training_loop_and_joint_criterion=True, core_and_additions_in_optimizer=True,
    checkpoint_full_model_optimizer_scheduler_rng=True, inactive_zero_geometry_and_R_removed_from_architecture=True,
    train_eval_voxel_processors_separate=True, no_evaluator_box_replacement=True,
    original_text_encoder_freeze_preserved=True, actual_gpu_preflight_passed=False,
    native_joint_training_started=False, runtime_dependency_overlay_pending=True,
    actual_initial_equivalence_pending=True, single_a100_current_pair_unchanged=True,
    new_weights_created=0, accuracy_result=False)
(root / 'ENTRY_PREPARATION.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({key:value for key,value in receipt.items() if key not in ('source_files', 'dependency_sources')}))
