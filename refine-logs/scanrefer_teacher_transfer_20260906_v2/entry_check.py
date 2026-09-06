import json, sys
sys.path.insert(0, '/root/autodl-tmp/mcln_g0_view_pair_20260905/inputs_v3/fixed_source')
from pathlib import Path
from main_utils import parse_option
from scripts.run_frozen_v99_pareto_contextual_official import build_authoritative_command
command = build_authoritative_command(Path('/tmp/unused_scanrefer_teacher_output'))
sys.argv = ['check'] + command[command.index('train_dist_mod.py') + 1:]
args = parse_option()
assert args.dataset == ['scanrefer'] and args.test_dataset == 'scanrefer'
assert args.butd and not args.butd_cls and not args.butd_gt
assert args.eval_use_selector_choice_scores
print(json.dumps({'dataset': args.dataset, 'butd': args.butd, 'butd_cls': args.butd_cls,
    'selector_choice': args.eval_use_selector_choice_scores, 'checkpoint_path': args.checkpoint_path,
    'batch_size': args.batch_size, 'model': args.model}))
