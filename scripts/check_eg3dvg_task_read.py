"""CPU behavior checks with the real EG decoder/head and author parameters.

This is not a scene-level forward, optimizer run, or accuracy evaluation.
"""
import argparse
import importlib
import io
import json
from pathlib import Path
import sys
import types


def namespace(name, path):
    package = types.ModuleType(name)
    package.__path__ = [str(path)]
    sys.modules[name] = package
    return importlib.import_module(name + '.encoder_decoder_layers')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--native-source', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    import torch
    from torch import nn
    torch.set_num_threads(1)
    torch.manual_seed(2027)
    sys.path.insert(0, str(args.source))
    native_module = namespace('native_fixture', args.native_source / 'models')
    task_module = namespace('task_fixture', args.source / 'models')
    heads = importlib.import_module('task_fixture.modules')
    checkpoint = torch.load(args.checkpoint, map_location='cpu')['model']
    options = dict(d_model=288, n_heads=8, dim_feedforward=256,
                   butd=True, super_refine=True)
    native = native_module.BiDecoderLayer(**options)
    task = task_module.BiDecoderLayer(**options)
    state = {k[len('module.decoder.5.'):]: v for k, v in checkpoint.items()
             if k.startswith('module.decoder.5.')}
    native.load_state_dict(state, strict=True)
    task.load_state_dict(state, strict=True)
    task.task_queries = nn.Parameter(torch.zeros(2, 288, 288))
    head = heads.ClsAgnosticPredictHead(256, 1, 256, 288, objectness=False)
    head.load_state_dict({k[len('module.prediction_heads.5.'):]: v for k, v in checkpoint.items()
                         if k.startswith('module.prediction_heads.5.')}, strict=True)
    del checkpoint
    q = torch.randn(2, 256, 288)
    inputs = dict(query=q, vis_feats=torch.randn(2, 32, 288),
                  lang_feats=torch.randn(2, 9, 288), query_pos=torch.rand(2, 256, 6),
                  padding_mask=None, text_key_padding_mask=torch.zeros(2, 9, dtype=torch.bool),
                  detected_feats=torch.randn(2, 10, 288), detected_mask=torch.zeros(2, 10, dtype=torch.bool),
                  xyz_embed=torch.randn(2, 288, 32), points_xyz=torch.rand(2, 32, 3),
                  super_features=[torch.randn(288, 27), torch.randn(288, 35)],
                  attn_mask_list=[torch.zeros(1, 256, 27, dtype=torch.bool), torch.zeros(1, 256, 35, dtype=torch.bool)])
    checks = {}
    for training in (False, True):
        native.train(training)
        task.train(training)
        torch.manual_seed(2027)
        with torch.no_grad():
            reference, ref_super = native(**inputs)
        expected_rng = torch.get_rng_state()
        torch.manual_seed(2027)
        with torch.no_grad():
            actual, act_super = task(**inputs)
        assert torch.equal(reference, actual)
        assert torch.equal(reference, task.semantic_query)
        assert all(torch.equal(a, b) for a, b in zip(ref_super, act_super))
        assert torch.equal(expected_rng, torch.get_rng_state())
        checks['train' if training else 'eval'] = 'exact outputs, super features and RNG'
    task.eval()
    head.eval()

    def outputs():
        geometry, super_features = task(**inputs)
        end = {}
        center, size, mask = head(geometry.transpose(1, 2), inputs['query_pos'][:, :, :3], end,
                                  semantic_features=task.semantic_query.transpose(1, 2))
        return end['sem_cls_scores'], center, size, mask, super_features

    baseline = outputs()
    with torch.no_grad():
        task.task_queries[0].copy_(torch.randn(288, 288) * .001)
    semantic_change = outputs()
    assert not torch.equal(baseline[0], semantic_change[0])
    assert all(torch.equal(a, b) for a, b in zip(baseline[1:4], semantic_change[1:4]))
    assert all(torch.equal(a, b) for a, b in zip(baseline[4], semantic_change[4]))
    with torch.no_grad():
        task.task_queries.zero_()
        task.task_queries[1].copy_(torch.randn(288, 288) * .001)
    geometry_change = outputs()
    assert torch.equal(baseline[0], geometry_change[0])
    assert all(not torch.equal(a, b) for a, b in zip(baseline[1:4], geometry_change[1:4]))
    checks['interventions'] = 'semantic changes only semantic; geometry changes box/mask and refined memory'
    task.zero_grad()
    outputs()[0].square().mean().backward()
    grad = task.task_queries.grad
    assert torch.isfinite(grad).all() and grad[0].abs().sum() > 0 and torch.count_nonzero(grad[1]) == 0
    task.zero_grad()
    outputs()[1].square().mean().backward()
    grad = task.task_queries.grad
    assert torch.isfinite(grad).all() and grad[1].abs().sum() > 0 and torch.count_nonzero(grad[0]) == 0
    checks['gradients'] = 'both requests receive their own head gradients; no opposite-request gradient'
    buffer = io.BytesIO()
    torch.save(task.state_dict(), buffer)
    buffer.seek(0)
    recovered = task_module.BiDecoderLayer(**options)
    recovered.task_queries = nn.Parameter(torch.zeros(2, 288, 288))
    restored = recovered.load_state_dict(torch.load(buffer), strict=True)
    assert not restored.missing_keys and not restored.unexpected_keys
    recovered.eval()
    with torch.no_grad():
        expected = task(**inputs)
        actual = recovered(**inputs)
    assert torch.equal(expected[0], actual[0])
    assert torch.equal(task.semantic_query, recovered.semantic_query)
    assert all(torch.equal(a, b) for a, b in zip(expected[1], actual[1]))
    checks['serialization'] = 'strict task-layer state round trip'
    report = {'status': 'pass', 'checks': checks, 'new_parameters': task.task_queries.numel(),
              'optimizer_steps': 0, 'scene_forwards': 0, 'accuracy_result': False}
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
