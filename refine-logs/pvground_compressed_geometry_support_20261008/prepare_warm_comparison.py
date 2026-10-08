"""Reuse the closed native pair, changing only warm start and geometry encoding."""
import ast
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
old = root.parent / 'pvground_mask_support_correction_20261008_v2'
spec = json.loads((old / 'pair_spec.json').read_bytes())


def replace_once(text, before, after):
    assert text.count(before) == 1, before
    return text.replace(before, after)


for name, digest in spec['new_runner_files'].items():
    raw = (old / name).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == digest, name
    text = raw.decode().replace('\r\n', '\n')
    if name == 'mask_support_corrector.py':
        text = replace_once(text,
            '        if not self.use_box_geometry:\n',
            '        if self.use_box_geometry:\n'
            '            position = position.sign() * position.abs().log1p()\n'
            '        if not self.use_box_geometry:\n')
    if name == 'paired_support_loop.py':
        text = replace_once(text,
            "        self.heads = {'content': CandidateMaskSupportCorrector(False).cuda()}\n",
            "        warm = torch.load(spec['warm_support_terminal'], map_location='cpu')\n"
            "        assert self.sha(spec['warm_support_terminal']) == spec['warm_support_terminal_sha256']\n"
            "        assert warm['arm'] == 'content' and warm['step'] == 3723\n"
            "        assert warm['reference_mode'] == 'fused_mask' and warm['mask_loss_coefficients'] == [5, 1, 10, 2]\n"
            "        for key in ('checkpoint_sha256', 'base_terminal_sha256', 'selected_terminal_sha256', 'source_port_sha256'):\n"
            "            assert warm[key] == spec[key]\n"
            "        prefix = 'candidate_support_corrector.'\n"
            "        assert len(warm['state_delta']) == 10 and all(name.startswith(prefix) for name in warm['state_delta'])\n"
            "        self.warm_state = {name[len(prefix):]: value for name, value in warm['state_delta'].items()}\n"
            "        self.heads = {'content': CandidateMaskSupportCorrector(False).cuda()}\n"
            "        self.heads['content'].load_state_dict(self.warm_state, strict=True)\n"
            "        assert torch.count_nonzero(self.heads['content'].output.weight) > 0\n"
            "        with torch.no_grad():\n"
            "            self.heads['content'].member[0].weight[:, -9:].zero_()\n"
            "        for name, value in self.heads['content'].state_dict().items():\n"
            "            expected = self.warm_state[name].clone()\n"
            "            if name == 'member.0.weight':\n"
            "                expected[:, -9:].zero_()\n"
            "            assert torch.equal(value.cpu(), expected)\n")
        text = replace_once(text,
            "        return parent, pair, call\n",
            "        if self.steps == 0:\n"
            "            assert all(torch.equal(a, b) for a, b in zip(\n"
            "                pair['content']['sp_last_pred_masks'], pair['box_conditioned']['sp_last_pred_masks']))\n"
            "            for key in ('last_center', 'last_pred_size', 'last_sem_cls_scores'):\n"
            "                assert torch.equal(pair['content'][key], pair['box_conditioned'][key])\n"
            "        return parent, pair, call\n")
        text = replace_once(text,
            "            if preflight and self.steps == 0:\n"
            "                assert all(torch.equal(a, b) for a, b in zip(\n"
            "                    predictions['sp_last_pred_masks'], parent['sp_last_pred_masks']))\n"
            "                assert torch.equal(predictions['last_center'], parent['last_center'])\n"
            "                assert torch.equal(predictions['last_pred_size'], parent['last_pred_size'])\n",
            "            if preflight and self.steps == 0:\n"
            "                original = CandidateMaskSupportCorrector(False).cuda()\n"
            "                original.load_state_dict(self.warm_state, strict=True)\n"
            "                original.eval()\n"
            "                with torch.no_grad():\n"
            "                    raw = inputs['points'][:, 1:].reshape(inputs['batch_size'], 50000, 6)\n"
            "                    expected_masks, _ = original(parent['support_query_features'],\n"
            "                        parent['support_super_features'], raw, parent['native_coarse_center'],\n"
            "                        parent['native_coarse_size'], parent)\n"
            "                assert all(torch.equal(a, b) for a, b in zip(\n"
            "                    predictions['sp_last_pred_masks'], expected_masks))\n"
            "                del original, expected_masks\n")
        text = replace_once(text,
            "                record['zero_update_mask_and_box_exact'] = self.steps == 0\n",
            "                record['zero_update_warm_mask_exact'] = self.steps == 0\n"
            "                record['warm_arm_mask_and_box_exact'] = self.steps == 0\n"
            "                record['geometry_column_gradient_norm'] = float(head.member[0].weight.grad[:, -9:].norm())\n"
            "                if self.steps == 0:\n"
            "                    assert (record['geometry_column_gradient_norm'] > 0) == (arm == 'box_conditioned')\n")
        text = replace_once(text,
            "            numpy_rng=np.random.get_state(), python_rng=random.getstate())\n",
            "            numpy_rng=np.random.get_state(), python_rng=random.getstate(),\n"
            "            geometry_encoding='signed_log' if arm == 'box_conditioned' else 'zero',\n"
            "            warm_support_terminal_sha256=self.spec['warm_support_terminal_sha256'],\n"
            "            support_prior_updates=3723, total_support_updates=3723 + self.steps,\n"
            "            optimizer_reinitialized=True, unused_geometry_columns_zeroed_at_initialization=True)\n")
        text = replace_once(text,
            "            assert payload['spec_sha256'] == self.sha(self.args.spec)\n",
            "            assert payload['spec_sha256'] == self.sha(self.args.spec)\n"
            "            assert payload['warm_support_terminal_sha256'] == self.spec['warm_support_terminal_sha256']\n"
            "            assert payload['total_support_updates'] == 7446 and payload['optimizer_reinitialized']\n"
            "            assert payload['geometry_encoding'] == ('signed_log' if arm == 'box_conditioned' else 'zero')\n")
        text = replace_once(text,
            "            current_parent = [receipt['parent_hits25'], receipt['parent_hits50']]\n"
            "            assert all([value['rec_hits25'], value['rec_hits50']] == current_parent for value in metrics.values())\n"
            "            receipt.update(historical_protected_hits=[5598, 4848],\n"
            "                historical_protected_difference=[actual-old for actual,old in zip(current_parent,[5598,4848])],\n"
            "                historical_best_updated=False, zero_output_head_gain_claim=False)\n",
            "            actual_warm = [metrics['content']['rec_hits25'], metrics['content']['rec_hits50']]\n"
            "            assert [metrics['box_conditioned']['rec_hits25'], metrics['box_conditioned']['rec_hits50']] == actual_warm\n"
            "            receipt.update(historical_protected_hits=[5598, 4856],\n"
            "                historical_protected_difference=[actual-old for actual,old in zip(actual_warm,[5598,4856])],\n"
            "                historical_best_updated=False, both_warm_heads_same_forward_exact=True)\n")
    if name == 'mask_support_model_factory.py':
        text = replace_once(text,
            "        assert payload['arm'] in ARMS and payload['reference_mode'] == 'fused_mask'\n",
            "        assert payload['arm'] in ARMS and payload['reference_mode'] == 'fused_mask'\n"
            "        assert payload['geometry_encoding'] == ('signed_log' if payload['arm'] == 'box_conditioned' else 'zero')\n")
    if name == 'run_mask_support_pair.py':
        text = replace_once(text, "assert spec['starting_hits']==[5598,4848]", "assert spec['starting_hits']==[5598,4856]")
        text = replace_once(text, "initial_support_output_zero=True,", "initial_support_output_zero=False, warm_support_restore_required=True,")
    assert not (root / name).exists(), name
    (root / name).write_text(text, encoding='utf-8', newline='\n')

controller = (old / 'controller.py').read_text().replace('\r\n', '\n')
controller = replace_once(controller,
    "parents = {Path(spec[key]): spec[key + '_sha256'] for key in ('base_terminal', 'selected_terminal')}",
    "parents = {Path(spec[key]): spec[key + '_sha256'] for key in ('base_terminal', 'selected_terminal', 'warm_support_terminal')}")
controller = replace_once(controller, "protected_best_hits=[5598, 4848]", "protected_best_hits=[5598, 4856]")
(root / 'controller.py').write_text(controller, encoding='utf-8', newline='\n')
(root / 'observe_preflight_authorized.py').write_bytes((old / 'observe_preflight_authorized.py').read_bytes())
for key in ('initial_import_failure_preserved', 'retry_reason'):
    del spec[key]
spec.update(root='/root/autodl-tmp/pvground_compressed_geometry_support_20261008',
    starting_hits=[5598, 4856], warm_support_terminal='/root/autodl-tmp/pvground_mask_support_correction_20261008_v2/content/terminal.pth',
    warm_support_terminal_sha256='09e60496cd0a87c355083126557639447ab270d262ae85a2cb95b77b1b9e46da',
    geometry_encoding='signed_log', geometry_formula='sign(position)*log1p(abs(position))',
    warm_support_prior_updates=3723, support_total_updates_at_terminal=7446,
    optimizer_reinitialized=True, unused_geometry_columns_zeroed_at_initialization=True,
    source_scale_diagnostic='/root/autodl-tmp/pvground_support_geometry_scale_20261008',
    previous_completed_experiment='/root/autodl-tmp/pvground_mask_support_correction_20261008_v2')
spec['new_runner_files'] = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                          for name in spec['new_runner_files']}
(root / 'pair_spec.json').write_text(json.dumps(spec, indent=2) + '\n', encoding='utf-8')
for path in root.glob('*.py'):
    ast.parse(path.read_bytes())
(root / 'PREPARED_SOURCE.json').write_text(json.dumps(dict(status='SOURCE_PREPARED_NOT_EXECUTED',
    source_files=len(spec['new_runner_files']), formal_accuracy=False, neural_execution=False,
    warm_parent_hits=[5598,4856], seed=2027, changes=['same warm content state',
    'both unused geometry weight columns zeroed', 'signed-log geometry9 only'],
    original_sources_unchanged=all(hashlib.sha256((old/name).read_bytes()).hexdigest()==digest
        for name,digest in json.loads((old/'pair_spec.json').read_bytes())['new_runner_files'].items())), indent=2)+'\n')
print(json.dumps(dict(status='WARM_COMPARISON_PREPARED_NOT_EXECUTED',source_files=8,
    same_seed=2027,warm_hits=[5598,4856],geometry_encoding='signed_log')))
