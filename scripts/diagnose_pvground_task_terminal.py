"""Fixed D terminal interventions on four existing training fixtures, not evaluation."""
import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

from evaluate import expanded_parent_state, terminal_state, sha, write_json, now


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', type=Path, required=True)
    args = parser.parse_args()
    root = args.spec.parent
    spec = json.loads(args.spec.read_bytes())
    begin = time.time()
    for name, digest in spec['files'].items():
        assert sha(root/name) == digest, name
    training = Path(spec['training_root'])
    assert (training/'controller.exit').read_text().strip() == '0'
    train_spec = json.loads((training/'spec.json').read_bytes())
    receipt = json.loads((training/'receipt.json').read_bytes())
    assert receipt['status'] == 'complete' and receipt['training_steps'] == 3723
    assert sha(training/'spec.json') == spec['training_spec_sha256'] == receipt['spec_sha256']
    assert sha(training/'terminal.pth') == spec['terminal_sha256'] == receipt['terminal_sha256']
    runtime = Path(train_spec['runtime'])
    environment = json.loads((runtime/'env_spec.json').read_bytes())
    assert hashlib.sha256(json.dumps(environment,sort_keys=True,separators=(',',':')).encode()).hexdigest() == train_spec['env_spec_sha256']
    source = Path(train_spec['model_source'])
    assert sha(train_spec['source_port']) == train_spec['source_port_sha256']
    for name, digest in json.loads(Path(train_spec['source_port']).read_bytes())['files'].items():
        assert sha(source/name) == digest, name
    for filename, key in [('pvground_source_query.py','source_query_module_sha256'),
                          ('pvground_observation_query.py','observation_module_sha256'),
                          ('pvground_task_observation_query.py','task_module_sha256')]:
        assert sha(root/filename) == train_spec[key] == receipt[key]
    fixtures = Path(spec['fixtures'])
    assert sha(fixtures/'receipt.json') == spec['fixture_receipt_sha256']
    fixture_receipt = json.loads((fixtures/'receipt.json').read_bytes())
    assert fixture_receipt['status'] == 'pass' and fixture_receipt['separate_training_labels']
    assert [r['training_row_id'] for r in fixture_receipt['rows']] == [0,173,237,455]
    import numpy as np
    import torch
    from torch.utils.data._utils.collate import default_collate
    from types import SimpleNamespace

    def reset_rng():
        random.seed(2027); np.random.seed(2027)
        torch.manual_seed(2027); torch.cuda.manual_seed_all(2027)

    reset_rng()
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.chdir(str(source)); sys.path.insert(0,str(source))
    from models.pv_ground import PVGround
    from pcdet.config import cfg, cfg_from_yaml_file
    from prepare_data import DataProcessor
    from train_dist_mod import TrainTester
    assert Path(sys.modules['models.pv_ground'].__file__).resolve() == source/'models/pv_ground.py'
    checkpoint = environment['weight_dirs']['scanrefer']
    assert sha(checkpoint['path']) == checkpoint['sha256'] == receipt['checkpoint_sha256']
    saved = torch.load(checkpoint['path'],map_location='cpu')
    config = saved['config']
    assert config.butd and not config.butd_cls and not config.butd_gt
    assert all(name.startswith('module.') for name in saved['model'])
    parent = {name[7:]:value for name,value in saved['model'].items()}
    cfg_from_yaml_file(str(runtime/'PV-Ground/wandb_config.yaml'),cfg)
    model = PVGround(cfg,num_class=256,num_queries=256,num_decoder_layers=6,
        self_position_embedding=config.self_position_embedding,contrastive_align_loss=True,butd=True,
        pointnet_ckpt=None,data_path=fixture_receipt['data_root'],self_attend=config.self_attend)
    initial = expanded_parent_state(model,parent)
    saved_delta = torch.load(str(training/'terminal.pth'),map_location='cpu')
    manifest = json.loads(Path(train_spec['input_manifest']).read_bytes())
    partitions = json.loads(Path(manifest['split_protocol']).read_bytes())['row_ids']
    assert saved_delta['step'] == 3723 and Counter(saved_delta['row_ids']) == Counter(partitions['fit'])
    assert saved_delta['parent_checkpoint_sha256'] == checkpoint['sha256']
    assert saved_delta['spec_sha256'] == sha(training/'spec.json')
    for key in ['source_query_read','source_query_module_sha256','source_port_sha256',
                'observation_state','observation_module_sha256','task_read','task_module_sha256']:
        assert saved_delta[key] == train_spec[key] == receipt[key], key
    terminal = terminal_state(model,initial,saved_delta['state_delta'])
    model.load_state_dict(terminal,strict=True)
    assert len(terminal) == 1271
    assert all(torch.equal(value,terminal[name]) for name,value in model.state_dict().items())
    model.cuda().eval()
    reader = model.decoder[-1].source_query_read
    assert reader.enabled and model.decoder[-1].task_read
    task_weights = reader.task_queries.detach().clone()
    assert all(float(w.abs().max()) > 0 for w in task_weights)
    owner = SimpleNamespace(data_processor=DataProcessor(cfg.DATA_PROCESSOR,np.asarray(cfg.DATA_CONFIG.POINT_CLOUD_RANGE),False,6))
    captured = {}

    def capture(module, inputs, outputs):
        captured['reader_semantic'] = outputs[0].detach().cpu().clone()
        captured['reader_geometry'] = outputs[1].detach().cpu().clone()

    handle = reader.register_forward_hook(capture)
    conditions = [('normal',()),('normal_repeat',()),('zero_semantic',(0,)),
                  ('zero_geometry',(1,)),('zero_both',(0,1))]
    assert [x[0] for x in conditions] == spec['conditions']
    tensor_keys = ['query_points_xyz','last_center','last_pred_size','last_sem_cls_scores',
                   'last_proj_queries','proj_tokens']
    list_keys = ['last_pred_masks','sp_last_pred_masks','adaptive_weights']
    records = []
    arrays = {}
    for start in (0,2):
        rows = fixture_receipt['rows'][start:start+2]
        values = []
        for row in rows:
            path = fixtures/row['file']
            assert sha(path) == row['file_sha256']
            value = torch.load(str(path),map_location='cpu')
            assert set(value) == {'inputs','labels'}
            values.append(value)
        batch = default_collate([v['labels'] for v in values])
        for old,new in [('point_clouds','point_clouds'),('det_boxes','all_detected_boxes'),
                        ('det_bbox_label_mask','all_detected_bbox_label_mask'),
                        ('det_class_ids','all_detected_class_ids'),('superpoint','superpoint')]:
            batch[new] = torch.cat([v['inputs'][old] for v in values],0)
        batch['utterances'] = [v['inputs']['text'][0] for v in values]
        batch = {k:v.cuda() if torch.is_tensor(v) else v for k,v in batch.items()}
        inputs = TrainTester._get_inputs(owner,batch)
        inputs['train'] = False
        assert torch.equal(inputs['points'][:,1:].reshape(2,50000,6),batch['point_clouds'])
        snapshots = {}
        selections = {}
        for condition, zero_indices in conditions:
            with torch.no_grad():
                reader.task_queries.copy_(task_weights)
                for index in zero_indices:
                    reader.task_queries[index].zero_()
                reset_rng()
                output = model({k:v.clone() if torch.is_tensor(v) else copy.deepcopy(v) for k,v in inputs.items()})
            snapshot = {key:output[key].detach().cpu().clone() for key in tensor_keys}
            for key in list_keys:
                for index,value in enumerate(output[key]):
                    snapshot[key+'_'+str(index)] = value.detach().cpu().clone()
            snapshot.update(captured); captured.clear()
            assert all(torch.isfinite(v).all() for v in snapshot.values())
            probabilities = output['last_sem_cls_scores'].softmax(-1)
            # Same text-span maps as the pinned bbs evaluation. GT boxes/masks are not model inputs.
            score = (probabilities*(batch['positive_map'][:,0]>0).float().unsqueeze(1)).sum(-1)
            for key in ['modify_positive_map','pron_positive_map','rel_positive_map']:
                score += (probabilities*batch[key][:,0].unsqueeze(1)).sum(-1)
            score -= (probabilities*batch['other_entity_map'][:,0].unsqueeze(1)).sum(-1)
            snapshot['bbs_score'] = score.detach().cpu().clone()
            selection = score.argsort(dim=-1,descending=True)[:,0].cpu()
            selections[condition] = selection.tolist()
            boxes = torch.cat([output['last_center'],output['last_pred_size'].clamp(min=1e-6)],-1).cpu()
            arrays[str(start)+'_'+condition+'_boxes'] = boxes.numpy()
            arrays[str(start)+'_'+condition+'_scores'] = snapshot['bbs_score'].numpy()
            arrays[str(start)+'_'+condition+'_selected'] = selection.numpy()
            snapshots[condition] = snapshot
            print('DIAGNOSTIC_FORWARD '+json.dumps(dict(batch=start,condition=condition,selected=selection.tolist())),flush=True)
            del output
        comparisons = {}
        for condition,_ in conditions[1:]:
            differences = {}
            for key,normal in snapshots['normal'].items():
                value = snapshots[condition][key]
                assert normal.shape == value.shape
                difference = (normal-value).abs().double()
                differences[key] = dict(shape=list(normal.shape),exact=torch.equal(normal,value),
                    max_abs=float(difference.max()),mean_abs=float(difference.mean()),
                    rms=float(difference.square().mean().sqrt()))
            comparisons[condition] = differences
        records.append(dict(training_row_ids=[r['training_row_id'] for r in rows],
                            selections=selections,changes_vs_normal=comparisons))
        del snapshots, batch, inputs, values
    with torch.no_grad():
        reader.task_queries.copy_(task_weights)
    handle.remove()
    model.cpu()
    assert all(torch.equal(value,terminal[name]) for name,value in model.state_dict().items()), 'model state mutated'
    np.savez_compressed(str(root/'boxes_scores.npz'),**arrays)
    record = dict(status='complete',time_cst=now(),elapsed_seconds=time.time()-begin,
        training_rows=4,batch_size=2,model_forwards=10,optimizer_steps=0,formal_rows=0,new_checkpoints=0,
        scope='fixed trained D interventions on four existing training fixtures; no performance or promotion claim',
        seed=2027,terminal_sha256=spec['terminal_sha256'],training_spec_sha256=spec['training_spec_sha256'],
        fixture_receipt_sha256=spec['fixture_receipt_sha256'],source_port_sha256=train_spec['source_port_sha256'],
        env_spec_sha256=train_spec['env_spec_sha256'],strict_terminal_restore=True,restored_states=len(terminal),
        state_unchanged_after_restore=True,task_weight_norms=[float(w.norm()) for w in task_weights],
        task_weight_max_abs=[float(w.abs().max()) for w in task_weights],batches=records,
        boxes_scores_sha256=sha(root/'boxes_scores.npz'),script_sha256=sha(__file__))
    write_json(root/'diagnostic.json',record)
    print('PVG_TASK_TERMINAL_DIAGNOSTIC_COMPLETE '+json.dumps({k:v for k,v in record.items() if k!='batches'}),flush=True)


if __name__ == '__main__':
    main()
