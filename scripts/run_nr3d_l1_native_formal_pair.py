"""Paired native Nr3D evaluation, eligible only after L1's terminal screen.

Both arms use the original TrainTester evaluation branch on the same batches.
This script does not train, replace a protected checkpoint, or promote a result.
"""

import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time


def file_sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def selected_row(observed):
    rec = observed['rec_selection']
    oracle = observed['box_oracle_after_filter']
    return {'rec_query': None if rec is None else rec['query'],
            'rec_box_iou': None if rec is None else rec['box_iou'],
            'rec_query_mask_iou': None if rec is None else rec['mask_iou'],
            'mask_query': observed['mask_selection']['query'],
            'mask_iou': observed['mask_selection']['mask_iou'],
            'legal_box_oracle_iou': None if oracle is None else oracle['box_iou'],
            'legal_box_oracle_query_mask_iou': None if oracle is None else oracle['mask_iou'],
            'candidate_profiles': observed['score_profiles']['protected_selector'],
            'candidate_count': observed['score_profiles']['protected_selector']['after_filter']['candidate_count']}


def require_native_metric_parity(rows, arm, native_metrics):
    selected = [row[arm] for row in rows]
    summary = {'sample_count':len(rows)}
    for suffix, threshold in [('025',.25),('050',.5)]:
        summary['rec_hits'+suffix] = sum(row['rec_box_iou'] is not None and row['rec_box_iou'] > threshold for row in selected)
        summary['mask_hits'+suffix] = sum(row['mask_iou'] > threshold for row in selected)
        assert summary['rec_hits'+suffix] == sum(group['hits'+suffix] for group in native_metrics['position_subgroups'].values())
        assert summary['mask_hits'+suffix] == native_metrics['mask']['hits'+suffix]
    summary['mask_iou_sum'] = sum(row['mask_iou'] for row in selected)
    assert math.isclose(summary['mask_iou_sum'],native_metrics['mask']['iou_sum'],abs_tol=1e-8,rel_tol=0)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    options = parser.parse_args()
    directory = options.manifest.parent
    manifest = json.loads(options.manifest.read_text())
    source = Path(manifest['model_source'])
    training = Path(manifest['l1_train_directory'])

    def verify_inputs():
        source_manifest = source/'g0_source_manifest.json'
        assert file_sha(source_manifest) == manifest['source_manifest_sha256']
        for name,digest in json.loads(source_manifest.read_text())['files'].items():
            assert file_sha(source/name) == digest,name
        for name,digest in manifest['files'].items():
            assert file_sha(directory/name) == digest,name
        for name,metadata in manifest['data_files'].items():
            assert file_sha(Path(name)) == metadata['sha256'],name
        assert file_sha(Path(manifest['checkpoint'])) == manifest['checkpoint_sha256']
        assert file_sha(training/'receipt.json') == manifest['l1_training_receipt_sha256']
        assert file_sha(training.parent/'input_manifest.json') == manifest['l1_training_manifest_sha256']

    verify_inputs()
    os.chdir(str(source));sys.path.insert(0,str(source))
    import scripts
    scripts.__path__ = [str(directory/'scripts')]+list(scripts.__path__)
    import torch
    from main_utils import parse_option, prepare_source_moe_gate_checkpoint_config
    from train_dist_mod import TrainTester
    from scripts.nr3d_candidate_contract import diagnose_root_candidates
    from scripts.nr3d_text_position_key import LastTextAttentionIntervention
    from scripts.load_nr3d_l1_terminal import load_terminal_position_key
    from scripts.summarize_nr3d_text_position_l1 import verify_terminal_run, compare

    gate = verify_terminal_run(training)
    assert gate['integrity_pass'] and gate['fixed_screen_pass']
    training_receipt = json.loads((training/'receipt.json').read_text())
    training_manifest = json.loads((training.parent/'input_manifest.json').read_text())
    assert training_manifest['checkpoint_sha256'] == manifest['checkpoint_sha256']
    assert training_manifest['source_manifest_sha256'] == manifest['source_manifest_sha256']
    position_artifact = training_receipt['artifacts']['position']
    loaded_position = load_terminal_position_key(position_artifact['path'],position_artifact['sha256'],
                                                  manifest['checkpoint_sha256'])
    expected_addon_state = loaded_position.weight.detach().clone()
    output = directory/'results'
    output.mkdir(exist_ok=False)
    write_json(output/'recomputed_training_screen.json',gate)
    sys.argv = [str(source/'train_dist_mod.py')]+manifest['eval_argv']
    args = prepare_source_moe_gate_checkpoint_config(parse_option())
    historical = json.loads((directory/'historical_config.json').read_text())
    config_changes = {key:{'historical':value,'current':vars(args)[key]}
                      for key,value in historical.items() if vars(args)[key] != value}
    assert set(config_changes) <= {'checkpoint_path','log_dir','exp'},config_changes
    assert args.eval and not args.eval_train and args.expected_eval_sample_count == 7899
    assert args.batch_size == 16 and args.num_workers == 4
    assert args.butd_cls and not args.butd and not args.butd_gt
    assert args.eval_use_selector_choice_scores and not args.use_source_moe
    assert not any(getattr(args,key) for key in [
        'eval_use_rec_reranker_scores','eval_use_rec_geometry_reranker_scores','eval_use_rec_joint_box_mask'])
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl',init_method='env://',
                                         timeout=datetime.timedelta(seconds=5400))
    assert torch.distributed.get_world_size() == 1
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    parent = torch.load(manifest['checkpoint'],map_location='cpu')
    assert parent['epoch'] == 57 and parent['evaluation_only']
    assert 'optimizer' not in parent and 'scheduler' not in parent
    assert all(name.startswith('module.') for name in parent['model'])
    expected_state = {name[7:]:value for name,value in parent['model'].items()}
    del parent

    def verify_model_state(model):
        actual = model.module.state_dict()
        assert set(actual) == set(expected_state)
        for name,value in actual.items():
            assert torch.equal(value.detach().cpu(),expected_state[name]),name
        assert torch.equal(loaded_position.weight.detach().cpu(),expected_addon_state)

    class NativePairTester(TrainTester):
        def get_loaders(self,current_args):
            train_loader,test_loader = super().get_loaders(current_args)
            assert train_loader is None and len(test_loader.dataset) == 7899
            assert test_loader.dataset.split == 'val' and not test_loader.dataset.augment
            assert list(test_loader.sampler) == list(range(7899))
            self.annos = test_loader.dataset.annos
            self.rows = []
            self.position_statistics = {}
            self.row_stream = (output/'rows.jsonl').open('x')
            return train_loader,test_loader

        def _build_grounding_evaluator(self,current_args,prefixes):
            self.native_evaluator = super()._build_grounding_evaluator(current_args,prefixes)
            self.position_evaluator = super()._build_grounding_evaluator(current_args,prefixes)
            assert self.native_evaluator.only_root and self.native_evaluator.filter_non_gt_boxes
            assert self.position_evaluator.only_root and self.position_evaluator.filter_non_gt_boxes
            return self.native_evaluator

        def _main_eval_branch(self,batch_idx,batch_data,test_loader,model,statistics,criterion,set_criterion,current_args):
            # Evaluate position first and release its outputs before native forward.
            # The enclosing native epoch evaluates the returned protected outputs.
            attachment = LastTextAttentionIntervention(model.module,loaded_position)
            self.position_statistics,position = super()._main_eval_branch(
                batch_idx,batch_data,test_loader,model,self.position_statistics,criterion,set_criterion,current_args)
            attachment.remove()
            assert attachment.position_calls == attachment.decoder_calls == attachment.attention_calls == 1
            self.position_evaluator.evaluate(position,'last_')
            position_rows = diagnose_root_candidates(position,self.position_evaluator)
            earlier = {name:position[name].detach().clone()
                       for name in ['seed_inds','query_points_sample_inds','4head_center','4head_pred_size']}
            del position
            statistics,native = super()._main_eval_branch(
                batch_idx,batch_data,test_loader,model,statistics,criterion,set_criterion,current_args)
            for name,value in earlier.items():
                assert torch.equal(native[name],value),name
            native_rows = diagnose_root_candidates(native,self.native_evaluator)
            assert len(native_rows) == len(position_rows)
            for bid,(a,b) in enumerate(zip(native_rows,position_rows)):
                index = len(self.rows);anno = self.annos[index]
                assert native['scan_ids'][bid] == anno['scan_id']
                assert int(native['target_id'][bid]) == anno['target_id']
                record = {'id':index,'scan_id':anno['scan_id'],'target_id':anno['target_id'],
                          'input_point_sha256':hashlib.sha256(batch_data['point_clouds'][bid].detach().cpu().numpy().tobytes()).hexdigest(),
                          'protected':selected_row(a),'position':selected_row(b)}
                self.rows.append(record)
                self.row_stream.write(json.dumps(record,sort_keys=True,allow_nan=False)+'\n')
            self.row_stream.flush()
            return statistics,native

        @torch.no_grad()
        def evaluate_one_epoch(self,epoch,test_loader,model,criterion,set_criterion,current_args):
            verify_model_state(model)
            loaded_position.to(next(model.module.parameters()).device)
            self.started = time.time()
            self.native_metrics = super().evaluate_one_epoch(
                epoch,test_loader,model,criterion,set_criterion,current_args)
            self.position_evaluator.synchronize_between_processes()
            self.position_metrics = self.position_evaluator.export_retrain_metrics(expected_sample_count=7899)
            self.elapsed = time.time()-self.started
            self.row_stream.close()
            verify_model_state(model)
            return self.native_metrics

    tester = NativePairTester(args)
    write_json(output/'preflight.json',{'status':'eligible','config_changes':config_changes,
              'training_screen_pass':True,'training_receipt_sha256':gate['receipt_sha256'],
              'manifest_sha256':file_sha(options.manifest),'position_artifact':position_artifact,
              'optimizer_steps':0,'checkpoint_writes':0,'formal_promotion':False})
    tester.main(args)
    assert len(tester.rows) == 7899
    summaries = {'protected':require_native_metric_parity(tester.rows,'protected',tester.native_metrics),
                 'position':require_native_metric_parity(tester.rows,'position',tester.position_metrics)}
    paired = compare(tester.rows,tester.rows,'protected')
    # compare also returns the module-screen rule; it is not a formal benchmark gate.
    paired.pop('fixed_screen_pass')
    verify_inputs()
    assert file_sha(Path(position_artifact['path'])) == position_artifact['sha256']
    write_json(output/'receipt.json',{'schema':'mcln-l1-native-formal-pair-v1','status':'complete',
               'sample_count':7899,'summary':summaries,
               'native_metrics':{'protected':tester.native_metrics,'position':tester.position_metrics},
               'paired_changes':paired,'native_row_metric_parity':True,'same_batch_inputs':True,
               'earlier_query_sampling_and_decoder_outputs_identical':True,
               'source_data_parent_and_addon_unchanged':True,'optimizer_steps':0,'checkpoint_writes':0,
               'formal_promotion':False,'elapsed_seconds':tester.elapsed,
               'rows_sha256':file_sha(output/'rows.jsonl'),'manifest_sha256':file_sha(options.manifest)})
    print('L1 NATIVE FORMAL PAIR COMPLETE',json.dumps(summaries),flush=True)
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
