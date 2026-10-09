"""Keep normal epochs measurable and save only best plus live resume state."""
import ast
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
source = root / 'source'
assert not (root / 'METRIC_PREPARATION.json').exists()
snapshots = root / 'entry_r1_snapshots'
snapshots.mkdir()
for name in ('main_utils.py', 'train_dist_mod.py'):
    (snapshots / name).write_bytes((source / name).read_bytes())

def replace_once(text, old, new):
    assert text.count(old) == 1, old[:80]
    return text.replace(old, new, 1)

train_path = source / 'train_dist_mod.py'
train = train_path.read_text(encoding='utf-8')
train = replace_once(train, '        return None\n    \n    # BRIEF Scannet detection evalution', '''        keys = [('last_', threshold, 1, 'bbs') for threshold in (.25, .5)]
        rows = [int(evaluator.gts[key]) for key in keys]
        assert rows == [9508, 9508]
        return dict(rows=9508, hits025=int(evaluator.dets[keys[0]]),
                    hits050=int(evaluator.dets[keys[1]]), primary_score='last/bbs',
                    same_complete_model=True)

    # BRIEF Scannet detection evalution''')
train = replace_once(train, '        voxel_batch = processor.collate_batch(voxel_batch)',
    '        voxel_batch = processor.collate_batch(voxel_batch)\n'
    "        assert np.array_equal(voxel_batch['points'][:, 1:].reshape(len(batch_data['point_clouds']), 50000, 6),\n"
    "                              batch_data['point_clouds'].detach().cpu().numpy())")
train_path.write_text(train, encoding='utf-8')

main_path = source / 'main_utils.py'
main = main_path.read_text(encoding='utf-8')
main = replace_once(main, '        optimizer.load_state_dict(checkpoint[\'optimizer\'])',
    "        optimizer.load_state_dict(checkpoint['optimizer'])\n"
    "        args._retained_native_metrics = checkpoint['retained_metrics']")
start = main.index('def save_checkpoint(')
end = main.index('\n\nclass BaseTrainTester:', start)
saver = '''def save_checkpoint(args, epoch, model, optimizer, scheduler, save_cur=False):
    state = dict(config=args, model=model.state_dict(), optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict(), epoch=epoch,
        architecture=model.module.native_training_architecture,
        retained_metrics=args._retained_native_metrics,
        rng=dict(python=random.getstate(), numpy=np.random.get_state(),
                 torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all()))
    name = 'latest.pth' if save_cur else 'best.pth'
    destination = os.path.join(args.log_dir, name)
    temporary = destination + '.tmp'
    assert not os.path.exists(temporary)
    torch.save(state, temporary)
    os.replace(temporary, destination)
    print('Saved complete native checkpoint ' + destination)


def native_metric_order(metrics):
    return (metrics['hits025'] >= 5658 and metrics['hits050'] >= 4850,
            metrics['hits050'] >= 4850, metrics['hits025'], metrics['hits050'])


def record_native_metrics(args, epoch, metrics):
    assert metrics['rows'] == 9508 and metrics['primary_score'] == 'last/bbs'
    path = os.path.join(args.log_dir, 'native_metrics.jsonl')
    with open(path, 'a') as stream:
        stream.write(json.dumps(dict(epoch=epoch, **metrics)) + '\\n')
'''
main = main[:start] + saver + main[end:]
start = main.index('        # ##############################\n        # NOTE Training and Validation #')
end = main.index('    @staticmethod\n    def _to_gpu', start)
loop = '''        assert args.val_freq == 1 and args.start_epoch <= args.max_epoch
        if not args.checkpoint_path:
            initial = self.evaluate_one_epoch(0, test_loader, model, criterion, set_criterion, args)
            args._retained_native_metrics = initial
            record_native_metrics(args, 0, initial)
            save_checkpoint(args, 0, model, optimizer, scheduler)
        for epoch in range(args.start_epoch, args.max_epoch + 1):
            train_loader.sampler.set_epoch(epoch)
            train_loader.generator.manual_seed(args.rng_seed + epoch)
            tic = time.time()
            self.train_one_epoch(epoch, train_loader, model, criterion, set_criterion,
                                 optimizer, scheduler, args)
            self.logger.info('epoch {}, total time {:.2f}, lr_core {:.6f}, lr_backbone {:.6f}, lr_modules {:.6f}'.format(
                epoch, time.time() - tic, optimizer.param_groups[0]['lr'],
                optimizer.param_groups[1]['lr'], optimizer.param_groups[3]['lr']))
            metrics = self.evaluate_one_epoch(epoch, test_loader, model, criterion, set_criterion, args)
            record_native_metrics(args, epoch, metrics)
            if native_metric_order(metrics) > native_metric_order(args._retained_native_metrics):
                args._retained_native_metrics = metrics
                save_checkpoint(args, epoch, model, optimizer, scheduler)
            save_checkpoint(args, epoch, model, optimizer, scheduler, save_cur=True)
        return os.path.join(args.log_dir, 'best.pth')

'''
main = main[:start] + loop + main[end:]
main = replace_once(main, '        return data_dict\n',
    "        assert ((data_dict['gt_masks'] == 0) | (data_dict['gt_masks'] == 1)).all()\n"
    "        data_dict['gt_masks'] = data_dict['gt_masks'].bool()\n"
    '        return data_dict\n')
main_path.write_text(main, encoding='utf-8')

model_path = source / 'models/pv_ground.py'
model = model_path.read_text(encoding='utf-8')
model = replace_once(model, ".view(inputs['batch_size'], 50000, 3).detach().cpu().numpy()",
    ".view(inputs['batch_size'], 50000, 3).detach().cpu().numpy().astype('float64')")
model_path.write_text(model, encoding='utf-8')

files = {}
for path in source.rglob('*.py'):
    ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    files[path.relative_to(source).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
receipt = dict(status='NATIVE_JOINT_SOURCE_COMPLETE_STATIC_PARSE_ONLY_REVIEW_AND_GPU_PENDING',
    source_files=files, metric_scope='full9508_native_last_bbs_same_complete_model',
    target_minimum_hits=[5658,4850], checkpoints=['best.pth','latest.pth'],
    strict_full_model_and_adam_scheduler_restore=True, epoch_boundary_rng_restore_only=True,
    full_gpu_preflight=False, deployment=False, native_joint_training_started=False,
    trained_module_effectiveness_confirmed=False, current_frozen_pair_changed=False,
    runtime_overlay_binding_and_storage_gate_pending=True)
(root / 'METRIC_PREPARATION.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({key:value for key,value in receipt.items() if key!='source_files'}))
