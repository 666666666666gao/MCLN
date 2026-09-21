"""One-pass protected-MCLN adaptation, native versus final token replacement.

No augmentation and no joint detection prompts in this bounded Nr3D-only pair.
Both arms retain the protected architecture and every native loss coefficient.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-manifest', type=Path, required=True)
    parser.add_argument('--module', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint-dir', type=Path, required=True)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--preflight-receipt', type=Path)
    options = parser.parse_args()
    root = options.output
    root.mkdir(exist_ok=False)
    checkpoint_dir=options.checkpoint_dir
    checkpoint_dir.mkdir(parents=True,exist_ok=True)
    manifest = json.loads(options.input_manifest.read_text())
    source = Path(manifest['model_source'])
    spec = importlib.util.spec_from_file_location('nr_assignment', str(options.module))
    assignment = importlib.util.module_from_spec(spec); spec.loader.exec_module(assignment)
    os.chdir(str(source)); sys.path.insert(0, str(source))
    import numpy as np
    import torch
    from main_utils import parse_option
    from train_dist_mod import TrainTester
    from src.joint_det_dataset import Joint3DDataset
    from src.grounding_evaluator import GroundingEvaluator
    contract_spec = importlib.util.spec_from_file_location(
        'nr_candidate_contract', str(Path(__file__).with_name('nr3d_candidate_contract.py')))
    contract = importlib.util.module_from_spec(contract_spec)
    contract_spec.loader.exec_module(contract)
    diagnose_root_candidates = contract.diagnose_root_candidates

    def seed(value):
        random.seed(value); np.random.seed(value)
        torch.manual_seed(value); torch.cuda.manual_seed_all(value)

    def write(name, value):
        (root/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')

    seed(2027)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    payload = torch.load(manifest['checkpoint'], map_location='cpu')
    assert payload['evaluation_only'] and 'optimizer' not in payload
    sys.argv = [sys.argv[0]]; args = parse_option(); vars(args).update(vars(payload['config']))
    assert args.use_source_choice_selector and not args.use_source_moe
    assert not args.frozen and not args.small_lr and not args.source_choice_selector_train_only
    assert args.source_choice_selector_loss_weight == .5
    assert args.mask_loss_scale == args.consistency_loss_scale == 1
    assert args.num_decoder_layers == 6 and args.butd_cls
    initial = {k[7:]: v for k,v in payload['model'].items()}; del payload
    models, optimizers, criteria, parameters = {}, {}, {}, {}
    for arm in ['native', 'replacement']:
        model = TrainTester.get_model(args).cuda()
        model.load_state_dict(initial, strict=True)
        models[arm] = model
        parameters[arm] = {n:p for n,p in model.named_parameters() if p.requires_grad}
        optimizers[arm] = torch.optim.AdamW(parameters[arm].values(), lr=1e-5,
                                          weight_decay=.0005)
        criteria[arm] = TrainTester.get_criterion(args)
    assert list(parameters['native']) == list(parameters['replacement'])
    intervention = assignment.LastLayerSemanticAssignment(criteria['replacement'][1])
    config = {'seed':2027, 'epochs':1, 'train_batch_size':4, 'eval_batch_size':16,
              'learning_rate_all_native_trainable_parameters':1e-5, 'weight_decay':.0005,
              'clip_norm':.1, 'augmentation':False, 'joint_detection_samples':False,
              'source_choice_loss_weight':args.source_choice_selector_loss_weight,
              'model_source':str(source), 'checkpoint':manifest['checkpoint'],
              'checkpoint_directory':str(checkpoint_dir),
              'trainable_names':list(parameters['native']),
              'trainable_parameters':sum(p.numel() for p in parameters['native'].values()),
              'preflight_only':options.preflight_only}
    write('config.json', config)
    if not options.preflight_only:
        preflight = json.loads(options.preflight_receipt.read_text())
        assert preflight['status']=='pass' and preflight['optimizer_steps_per_arm']==2
        assert preflight['trainable_names']==list(parameters['native'])

    class Dataset(Joint3DDataset):
        def _scene_graph_parse(self, annos):
            if options.preflight_only:
                annos[:] = [annos[i] for i in manifest['row_ids']['fit'][:8]]
            super()._scene_graph_parse(annos)

        def __getitem__(self, index):
            result = super().__getitem__(index)
            result['pair_row_id'] = index
            return result

    print('NR_PAIR_DATASET_LOADING', flush=True)
    train = Dataset(dataset_dict={'nr3d':1}, test_dataset='nr3d', split='train',
        data_path=args.data_root, use_color=args.use_color, detect_intermediate=args.detect_intermediate,
        butd_cls=args.butd_cls, skip_missing_superpoints=args.skip_missing_superpoints)
    train.augment = False
    assert len(train)==(8 if options.preflight_only else 32919)
    validation = None
    if not options.preflight_only:
        validation = Dataset(dataset_dict={'nr3d':1}, test_dataset='nr3d', split='val',
            data_path=args.data_root, use_color=args.use_color, detect_intermediate=args.detect_intermediate,
            butd_cls=args.butd_cls, skip_missing_superpoints=args.skip_missing_superpoints)
        validation.augment = False
        assert len(validation)==7899
        assert not {a['scan_id'] for a in train.annos}.intersection(a['scan_id'] for a in validation.annos)
    evaluator = GroundingEvaluator(only_root=True,prefixes=['last_'],topks=[1],
        filter_non_gt_boxes=True,eval_use_selector_choice_scores=True)

    def evaluate(stage, arms):
        seed(2027)
        rows=[]
        for arm in arms: models[arm].eval()
        loader=torch.utils.data.DataLoader(validation,batch_size=16,shuffle=False,num_workers=0,
                                           generator=torch.Generator().manual_seed(2027))
        with torch.no_grad():
            for raw in loader:
                batch=TrainTester._to_gpu(raw); inputs=TrainTester._get_inputs(batch); inputs['train']=False
                arm_rows={}
                for arm in arms:
                    outputs=models[arm](inputs); outputs.update(batch)
                    arm_rows[arm]=diagnose_root_candidates(outputs,evaluator)
                    del outputs
                for i,row_id in enumerate(batch['pair_row_id'].tolist()):
                    record={'row_id':row_id,'scan_id':validation.annos[row_id]['scan_id'],
                            'point_sha256':hashlib.sha256(inputs['point_clouds'][i].cpu().numpy().tobytes()).hexdigest()}
                    for arm in arms:
                        obs=arm_rows[arm][i]; selected=obs['rec_selection']
                        record[arm]={'query':None if selected is None else selected['query'],
                            'iou':0. if selected is None else selected['box_iou'],
                            'candidate_profile':obs['score_profiles']['protected_selector']}
                    rows.append(record)
                if len(rows)%512==0: print('NR_PAIR_EVAL',stage,len(rows),flush=True)
        assert [row['row_id'] for row in rows]==list(range(7899))
        write(stage+'_rows.json',rows)
        summary={arm:{'hits025':sum(row[arm]['iou']>.25 for row in rows),
                      'hits050':sum(row[arm]['iou']>.5 for row in rows)} for arm in arms}
        write(stage+'_summary.json',summary)
        print('NR_PAIR_EVAL_COMPLETE',stage,json.dumps(summary),flush=True)
        return rows,summary

    if not options.preflight_only:
        baseline_rows, baseline_summary=evaluate('baseline',['native'])
    for model in models.values(): model.train()
    seed(2027)
    loader=torch.utils.data.DataLoader(train,batch_size=4,shuffle=True,num_workers=0,
                                      generator=torch.Generator().manual_seed(2027))
    started=time.time(); torch.cuda.reset_peak_memory_stats(); seen=[]; first_outputs={}
    stats=[]
    with (root/'train.jsonl').open('w') as log:
        for step,raw in enumerate(loader,1):
            batch=TrainTester._to_gpu(raw); inputs=TrainTester._get_inputs(batch); inputs['train']=True
            assert all(x=='nr3d' for x in batch['sample_dataset'])
            record={'step':step,'row_ids':batch['pair_row_id'].tolist(),
                    'point_sha256':hashlib.sha256(inputs['point_clouds'].cpu().numpy().tobytes()).hexdigest()}
            for arm in models:
                seed(2027+step)
                optimizer=optimizers[arm]; optimizer.zero_grad(set_to_none=True)
                outputs=models[arm](inputs)
                if step==1:
                    first_outputs[arm]={k:outputs[k].detach().clone() for k in
                                        ['last_center','last_pred_size','last_sem_cls_scores','selected_source_scores']}
                outputs.update(batch)
                if arm=='replacement': intervention.bind(outputs,batch['sample_dataset'])
                loss,outputs=TrainTester._compute_loss(outputs,*criteria[arm],args)
                assert bool(torch.isfinite(loss).all())
                loss.backward()
                grads=[p.grad for p in parameters[arm].values() if p.grad is not None]
                assert grads and all(bool(torch.isfinite(g).all()) for g in grads)
                norm=torch.nn.utils.clip_grad_norm_(list(parameters[arm].values()),.1)
                assert bool(torch.isfinite(norm)) and float(norm) > 0
                optimizer.step()
                record[arm]={'loss':float(loss),'gradient_norm':float(norm),
                             'parameters_with_gradient':len(grads)}
                if arm=='replacement':
                    assert len(intervention.records)==1
                    record[arm]['reassigned_per_sample']=intervention.records[0]['selected'].sum(-1).tolist()
                    record[arm]['delta_ce']=float(intervention.records[0]['delta'])
                del outputs,loss,grads
            if step==1:
                assert all(torch.equal(first_outputs['native'][k],first_outputs['replacement'][k]) for k in first_outputs['native'])
                first_outputs.clear()
            seen.extend(record['row_ids']); record['elapsed_seconds']=time.time()-started
            log.write(json.dumps(record,allow_nan=False)+'\n');log.flush()
            stats.append(record)
            if step%64==0 or options.preflight_only: print('NR_PAIR_TRAIN '+json.dumps(record),flush=True)
            if step%512==0 and not options.preflight_only:
                for arm in models:
                    torch.save({'model':models[arm].state_dict(),'optimizer':optimizers[arm].state_dict(),
                                'step':step,'config':args,'run_config':config},checkpoint_dir/(arm+'_recovery.pth'))
    assert sorted(seen)==list(range(len(train)))
    expected_steps=2 if options.preflight_only else 8230
    assert step==expected_steps
    changed={arm:[n for n,p in parameters[arm].items() if not torch.equal(p.detach().cpu(),initial[n])]
             for arm in models}
    assert all(changed.values())
    if options.preflight_only:
        for arm in models:
            state={'model':models[arm].state_dict(),'optimizer':optimizers[arm].state_dict()}
            path=checkpoint_dir/'optimizer_preflight.pth';torch.save(state,path)
            restored=torch.load(path,map_location='cpu')
            models[arm].load_state_dict(initial,strict=True)
            models[arm].load_state_dict(restored['model'],strict=True)
            assert all(torch.equal(v.detach().cpu(),restored['model'][n]) for n,v in models[arm].state_dict().items())
            restored_optimizer=torch.optim.AdamW(parameters[arm].values(),lr=1e-5,weight_decay=.0005)
            restored_optimizer.load_state_dict(restored['optimizer'])
            restored_state=restored_optimizer.state_dict()['state']
            assert set(restored_state)==set(restored['optimizer']['state'])
            for pid,values in restored_state.items():
                assert float(values['step'])==float(restored['optimizer']['state'][pid]['step'])
                for key in ['exp_avg','exp_avg_sq']:
                    assert torch.equal(values[key].detach().cpu(),restored['optimizer']['state'][pid][key])
            del restored_optimizer,restored
        report={'status':'pass','optimizer_steps_per_arm':2,'formal_rows':0,
                'trainable_names':list(parameters['native']), 'changed_parameters':changed,
                'first_training_predictions_identical':True,'checkpoint_model_reload_equal':True,
                'optimizer_state_reload_equal':True,
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'batches':stats}
        write('optimizer_preflight.json',report)
        print('NR_PAIR_PREFLIGHT_COMPLETE '+json.dumps({k:v for k,v in report.items() if k not in ['trainable_names','changed_parameters','batches']}),flush=True)
        return
    for arm in models:
        recovery=checkpoint_dir/(arm+'_recovery.pth')
        torch.save({'model':models[arm].state_dict(),'optimizer':optimizers[arm].state_dict(),
                    'step':step,'config':args,'run_config':config},recovery)
        recovery.replace(checkpoint_dir/(arm+'_terminal.pth'))
        saved=torch.load(checkpoint_dir/(arm+'_terminal.pth'),map_location='cpu')
        models[arm].load_state_dict(saved['model'],strict=True)
        del saved
    terminal_rows,terminal_summary=evaluate('terminal',['native','replacement'])
    assert all(a['point_sha256']==b['point_sha256'] for a,b in zip(baseline_rows,terminal_rows))
    paired={}
    for threshold in [.25,.5]:
        native_hits=[r['native']['iou']>threshold for r in terminal_rows]
        method_hits=[r['replacement']['iou']>threshold for r in terminal_rows]
        paired[str(threshold)]={'repairs':sum(b and not a for a,b in zip(native_hits,method_hits)),
                                'breaks':sum(a and not b for a,b in zip(native_hits,method_hits))}
    write('receipt.json',{'status':'complete','optimizer_steps_per_arm':step,'training_rows_per_arm':len(train),
                         'formal_rows':7899,'baseline':baseline_summary,'terminal':terminal_summary,
                         'replacement_vs_native':paired,
                         'terminal_weights_reloaded_before_evaluation':True,
                         'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                         'elapsed_seconds_after_initial_evaluation':time.time()-started})


if __name__=='__main__':
    main()
