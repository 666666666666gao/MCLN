"""Conditional native Nr3D evaluation of protected, native and sparse states.

The complete frozen module screen must pass before any formal GPU evaluation.
All three states use the original evaluator and the same 7899 input batches.
"""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time

def require_same_selection(rows):
    reference = rows['protected']
    for arm in ['native', 'sparse']:
        for key in ['rec_query', 'rec_box_iou', 'mask_query', 'legal_box_oracle_query',
                    'legal_box_oracle_iou', 'candidate_profiles', 'candidate_count']:
            assert rows[arm][key] == reference[key], (arm, key)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    options = parser.parse_args()
    directory = options.manifest.parent
    manifest = json.loads(options.manifest.read_text())
    source = Path(manifest['model_source'])
    training = Path(manifest['sparse_train_directory'])
    training_manifest = json.loads((training / 'input_manifest.json').read_text())
    os.chdir(str(source))
    sys.path.insert(0, str(source))
    import scripts
    scripts.__path__ = [str(directory / 'scripts')] + list(scripts.__path__)
    from scripts.run_nr3d_l1_native_formal_pair import (
        file_sha, require_native_metric_parity, selected_row, write_json)

    def verify_inputs():
        source_manifest = source / 'g0_source_manifest.json'
        assert file_sha(source_manifest) == manifest['source_manifest_sha256']
        for name, digest in json.loads(source_manifest.read_text())['files'].items():
            assert file_sha(source / name) == digest, name
        for name, digest in manifest['files'].items():
            assert file_sha(directory / name) == digest, name
        for name, metadata in manifest['data_files'].items():
            assert file_sha(Path(name)) == metadata['sha256'], name
        assert file_sha(Path(manifest['checkpoint'])) == manifest['checkpoint_sha256']
        assert file_sha(training / 'input_manifest.json') == manifest['training_manifest_sha256']
        assert file_sha(training / 'receipt.json') == manifest['training_receipt_sha256']
        assert file_sha(training / 'artifact_verification.json') == manifest['artifact_verification_sha256']
        for name in ['nr3d_sparse_point_memory.py', 'nr3d_point_voxel_mapping.py',
                     'summarize_nr3d_sparse_point_pair.py']:
            relative = 'scripts/' + name
            assert file_sha(directory / relative) == training_manifest['files'][relative], relative

    verify_inputs()
    assert training_manifest['checkpoint_sha256'] == manifest['checkpoint_sha256']
    assert training_manifest['source_manifest_sha256'] == manifest['source_manifest_sha256']
    from scripts.summarize_nr3d_sparse_point_pair import verify_terminal_run, compare, metrics

    # Recompute from the complete rows; a saved "pass" label is insufficient.
    gate = verify_terminal_run(training)
    assert gate['integrity_pass'] and gate['fixed_screen_pass']
    artifact_check = json.loads((training / 'artifact_verification.json').read_text())
    assert artifact_check['status'] == 'pass'
    assert artifact_check['manifest_sha256'] == gate['manifest_sha256']
    receipt = json.loads((training / 'receipt.json').read_text())
    import torch
    import spconv
    import cumm
    assert sys.prefix == training_manifest['python_prefix']
    assert torch.__version__ == '1.10.2+cu111'
    assert spconv.__version__ == '2.3.6' and cumm.__version__ == '0.4.11'
    from main_utils import parse_option, prepare_source_moe_gate_checkpoint_config
    from train_dist_mod import TrainTester
    from scripts.nr3d_candidate_contract import diagnose_root_candidates
    from scripts.nr3d_sparse_point_memory import SparsePointSuperpointResidual, SparseSuperpointIntervention
    from scripts.nr3d_sparse_formal_state import load_mask_state, MaskProjectionSwitch

    parent = torch.load(manifest['checkpoint'], map_location='cpu')
    assert parent['epoch'] == 57 and parent['evaluation_only']
    assert 'optimizer' not in parent and 'scheduler' not in parent
    assert all(name.startswith('module.') for name in parent['model'])
    expected_state = {name[7:]: value for name, value in parent['model'].items()}
    del parent
    shared_names = training_manifest['shared_parameter_names']
    assert len(shared_names) == 16
    shared_reference = {name: expected_state[name] for name in shared_names}
    addon = SparsePointSuperpointResidual().eval().requires_grad_(False)
    assert len(addon.state_dict()) == 17
    endpoints = {}
    for arm in ['native', 'sparse']:
        metadata = receipt['artifacts'][arm]
        assert Path(metadata['path']) == training / (arm + '_mask_state.pt')
        assert artifact_check['artifacts'][arm]['artifact_sha256'] == metadata['sha256']
        shared, new = load_mask_state(metadata, manifest['checkpoint_sha256'], arm,
                                     shared_reference, addon.state_dict())
        endpoints[arm] = shared
        if arm == 'sparse':
            addon.load_state_dict(new, strict=True)
    expected_addon = {name: value.detach().clone() for name, value in addon.state_dict().items()}
    output = directory / 'results'
    output.mkdir(exist_ok=False)
    write_json(output / 'recomputed_training_screen.json', gate)
    sys.argv = [str(source / 'train_dist_mod.py')] + manifest['eval_argv']
    args = prepare_source_moe_gate_checkpoint_config(parse_option())
    historical = json.loads((directory / 'historical_config.json').read_text())
    config_changes = {key: {'historical': value, 'current': vars(args)[key]}
                      for key, value in historical.items() if vars(args)[key] != value}
    assert set(config_changes) <= {'checkpoint_path', 'log_dir', 'exp'}, config_changes
    assert args.eval and not args.eval_train and args.expected_eval_sample_count == 7899
    assert args.batch_size == 16 and args.num_workers == 4
    assert args.butd_cls and not args.butd and not args.butd_gt
    assert args.eval_use_selector_choice_scores and not args.use_source_moe
    assert not any(getattr(args, key) for key in [
        'eval_use_rec_reranker_scores', 'eval_use_rec_geometry_reranker_scores',
        'eval_use_rec_selective_residual_scores', 'eval_use_rec_hierarchical_reranker_scores',
        'eval_use_rec_joint_box_mask'])
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://',
                                        timeout=datetime.timedelta(seconds=5400))
    assert torch.distributed.get_world_size() == 1
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    def verify_model_state(model):
        actual = model.module.state_dict()
        assert set(actual) == set(expected_state)
        for name, value in actual.items():
            assert torch.equal(value.detach().cpu(), expected_state[name]), name
        for name, value in addon.state_dict().items():
            assert torch.equal(value.detach().cpu(), expected_addon[name]), name

    class NativeComparisonTester(TrainTester):
        def get_loaders(self, current_args):
            train_loader, test_loader = super().get_loaders(current_args)
            assert train_loader is None and len(test_loader.dataset) == 7899
            assert test_loader.dataset.split == 'val' and not test_loader.dataset.augment
            assert list(test_loader.sampler) == list(range(7899))
            self.annos = test_loader.dataset.annos
            self.rows = []
            self.statistics = {'native': {}, 'sparse': {}}
            self.row_stream = (output / 'rows.jsonl').open('x')
            return train_loader, test_loader

        def _build_grounding_evaluator(self, current_args, prefixes):
            self.evaluators = {arm: super(NativeComparisonTester, self)._build_grounding_evaluator(
                current_args, prefixes) for arm in ['protected', 'native', 'sparse']}
            assert all(item.only_root and item.filter_non_gt_boxes for item in self.evaluators.values())
            return self.evaluators['protected']

        def _main_eval_branch(self, batch_idx, batch_data, test_loader, model, statistics,
                              criterion, set_criterion, current_args):
            self.projections.require_protected()
            observations, snapshots = {}, {}
            for arm in ['native', 'sparse']:
                self.projections.apply(arm)
                if arm == 'sparse':
                    attachment = SparseSuperpointIntervention(model.module, addon)
                self.statistics[arm], outputs = super()._main_eval_branch(
                    batch_idx, batch_data, test_loader, model, self.statistics[arm],
                    criterion, set_criterion, current_args)
                if arm == 'sparse':
                    assert attachment.inputs is None
                    assert attachment.scene_index == len(batch_data['point_clouds'])
                    attachment.remove()
                self.evaluators[arm].evaluate(outputs, 'last_')
                observations[arm] = diagnose_root_candidates(outputs, self.evaluators[arm])
                snapshots[arm] = {key: outputs[key].detach().clone()
                                  for key in ['last_center', 'last_pred_size', 'selected_source_scores']}
                del outputs
            self.projections.apply('protected')
            self.projections.require_protected()
            statistics, protected = super()._main_eval_branch(
                batch_idx, batch_data, test_loader, model, statistics, criterion, set_criterion, current_args)
            for arm in ['native', 'sparse']:
                for key, value in snapshots[arm].items():
                    assert torch.equal(protected[key], value), (arm, key)
            observations['protected'] = diagnose_root_candidates(protected, self.evaluators['protected'])
            assert len({len(items) for items in observations.values()}) == 1
            for bid in range(len(observations['protected'])):
                index = len(self.rows)
                anno = self.annos[index]
                assert protected['scan_ids'][bid] == anno['scan_id']
                assert int(protected['target_id'][bid]) == anno['target_id']
                record = {'id': index, 'scan_id': anno['scan_id'], 'target_id': anno['target_id'],
                    'input_point_sha256': hashlib.sha256(
                        batch_data['point_clouds'][bid].detach().cpu().numpy().tobytes()).hexdigest()}
                for arm, observed in observations.items():
                    record[arm] = selected_row(observed[bid])
                    oracle = observed[bid]['box_oracle_after_filter']
                    record[arm]['legal_box_oracle_query'] = None if oracle is None else oracle['query']
                require_same_selection(record)
                self.rows.append(record)
                self.row_stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + '\n')
            self.row_stream.flush()
            return statistics, protected

        @torch.no_grad()
        def evaluate_one_epoch(self, epoch, test_loader, model, criterion, set_criterion, current_args):
            verify_model_state(model)
            addon.to(next(model.module.parameters()).device)
            self.projections = MaskProjectionSwitch(model.module, endpoints)
            started = time.time()
            protected_metrics = super().evaluate_one_epoch(
                epoch, test_loader, model, criterion, set_criterion, current_args)
            self.native_metrics = {'protected': protected_metrics}
            for arm in ['native', 'sparse']:
                self.evaluators[arm].synchronize_between_processes()
                self.native_metrics[arm] = self.evaluators[arm].export_retrain_metrics(expected_sample_count=7899)
            self.elapsed = time.time() - started
            self.row_stream.close()
            self.projections.require_protected()
            verify_model_state(model)
            return protected_metrics

    tester = NativeComparisonTester(args)
    write_json(output / 'preflight.json', {'status': 'eligible', 'config_changes': config_changes,
        'training_screen_pass': True, 'training_receipt_sha256': gate['receipt_sha256'],
        'manifest_sha256': file_sha(options.manifest), 'artifacts': receipt['artifacts'],
        'arms': ['protected', 'native', 'sparse'], 'optimizer_steps': 0, 'formal_promotion': False})
    tester.main(args)
    assert len(tester.rows) == 7899
    summaries = {arm: require_native_metric_parity(tester.rows, arm, tester.native_metrics[arm])
                 for arm in ['protected', 'native', 'sparse']}
    comparisons = {}
    for arm in ['protected', 'native']:
        reference = [dict(row, native=row[arm]) for row in tester.rows]
        good = [index for index, row in enumerate(reference)
                if row['native']['legal_box_oracle_iou'] is not None
                and row['native']['legal_box_oracle_iou'] > .5]
        comparisons['sparse_minus_' + arm] = {
            'selected_mask': compare(reference, tester.rows),
            'selected_rec_query_mask': compare(reference, tester.rows, 'rec_query_mask_iou'),
            'fixed_good_box_query_mask': compare([reference[i] for i in good],
                [tester.rows[i] for i in good], 'legal_box_oracle_query_mask_iou')}
    verify_inputs()
    for metadata in receipt['artifacts'].values():
        assert file_sha(Path(metadata['path'])) == metadata['sha256']
    write_json(output / 'receipt.json', {'schema': 'mcln-sparse-native-formal-v1', 'status': 'complete',
        'sample_count': 7899, 'summary': summaries, 'native_metrics': tester.native_metrics,
        'mask_metrics': {arm: metrics(tester.rows, arm) for arm in ['protected', 'native', 'sparse']},
        'comparisons': comparisons, 'native_row_metric_parity': True, 'same_batch_inputs': True,
        'all_rec_outputs_and_query_selections_equal': True,
        'source_data_parent_and_endpoints_unchanged': True, 'optimizer_steps': 0,
        'checkpoint_writes': 0, 'formal_promotion': False, 'elapsed_seconds': tester.elapsed,
        'rows_sha256': file_sha(output / 'rows.jsonl'), 'manifest_sha256': file_sha(options.manifest)})
    print('SPARSE NATIVE FORMAL COMPLETE', json.dumps(summaries), flush=True)
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
