"""Fixed native ScanRefer official comparison: published parent and fit terminal."""
import argparse
from collections import Counter
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def row_metrics(rows, mode):
    values = [row[mode] for row in rows]
    mask_sum = sum(row['mask_iou'] for row in values)
    return {'rec_hits25': sum(row['iou'] > .25 for row in values),
            'rec_hits50': sum(row['iou'] > .5 for row in values),
            'mask_hits25': sum(row['mask_iou'] > .25 for row in values),
            'mask_hits50': sum(row['mask_iou'] > .5 for row in values),
            'mask_iou_sum': mask_sum, 'mask_miou': mask_sum / len(rows) * 100.}


def promotion_check(metrics):
    checks = {'rec25_historical_v99': metrics['rec_hits25'] >= 5572,
              'rec50_historical_v99': metrics['rec_hits50'] >= 4797,
              'mask25_paper': metrics['mask_hits25'] * 100. / 9508 >= 58.70,
              'mask50_paper': metrics['mask_hits50'] * 100. / 9508 >= 50.70,
              'mask_miou_paper': metrics['mask_miou'] >= 44.72}
    return {'checks': checks, 'advance_to_nr3d_sr3d_rec': all(checks.values()),
            'requires_independent_formal_audit': True, 'nr3d_sr3d_mask_gate': False}


def expanded_parent_state(model, parent):
    """Strict native restore precedes installation of the new reader."""
    import torch
    from pvground_source_query import install_source_query_read
    assert len(parent) == 1234
    assert set(model.state_dict()) - set(parent) == {'text_encoder.embeddings.position_ids'}
    assert not set(parent) - set(model.state_dict())
    assert torch.equal(model.text_encoder.embeddings.position_ids,
                       torch.arange(model.text_encoder.config.max_position_embeddings).expand((1, -1)))
    model.text_encoder.embeddings.register_buffer('position_ids', model.text_encoder.embeddings.position_ids, persistent=False)
    model.load_state_dict(parent, strict=True)
    install_source_query_read(model)
    added = set(model.state_dict()) - set(parent)
    assert len(added) == 24 and all(name.startswith('decoder.5.source_query_read.') for name in added)
    assert sum(p.numel() for p in model.decoder[-1].source_query_read.parameters()) == 714528
    state = dict(parent)
    state.update({name:value.detach().cpu().clone() for name,value in model.state_dict().items() if name in added})
    assert set(state) == set(model.state_dict())
    assert all(torch.equal(value.detach().cpu(), state[name]) for name,value in model.state_dict().items())
    return state


def terminal_state(model, initial, delta):
    """Require every trainable parameter and saved buffer, including all new keys."""
    import torch
    expected = {name for name,p in model.named_parameters() if p.requires_grad}
    expected |= {name for name,_ in model.named_buffers() if name in initial}
    assert set(delta) == expected
    result = dict(initial)
    result.update(delta)
    assert set(result) == set(model.state_dict())
    for name,value in result.items():
        assert value.shape == initial[name].shape and value.dtype == initial[name].dtype, name
        assert torch.isfinite(value).all(), name
    assert all(torch.equal(result[name], initial[name]) for name,p in model.named_parameters() if not p.requires_grad)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', type=Path, required=True)
    args = parser.parse_args()
    root = args.spec.parent
    spec = json.loads(args.spec.read_bytes())
    for name, digest in spec['files'].items():
        assert sha(root/name) == digest, name
    training = Path(spec['training_root'])
    audit_root = Path(spec['training_audit_root'])
    assert (training/'controller.exit').read_text().strip() == '0'
    assert (audit_root/'controller.exit').read_text().strip() == '0'
    receipt = json.loads((training/'receipt.json').read_bytes())
    audited = json.loads((audit_root/'audit.json').read_bytes())
    train_spec = json.loads((training/'spec.json').read_bytes())
    assert receipt['status'] == 'complete' and receipt['training_steps'] == 3723
    assert receipt['fit_rows'] == 29778 and receipt['holdout_rows'] == 6887 and receipt['formal_rows'] == 0
    assert audited['integrity_pass'] and audited['primary_rec_nonregression']
    assert audited['receipt_sha256'] == sha(training/'receipt.json')
    assert sha(training/'spec.json') == receipt['spec_sha256'] == spec['training_spec_sha256']
    assert sha(training/'train.py') == receipt['script_sha256'] == train_spec['files']['train.py']
    assert sha(training/'terminal.pth') == receipt['terminal_sha256']
    assert spec['primary_mode'] == train_spec['primary_mode'] == 'bbs'
    assert spec['seed'] == train_spec['seed'] == 2027 and spec['batch_size'] == 8
    manifest = json.loads(Path(train_spec['input_manifest']).read_bytes())
    source = Path(manifest['model_source'])
    assert sha(source/'appearance_source_manifest.json') == manifest['source_manifest_sha256']
    for name, digest in json.loads((source/'appearance_source_manifest.json').read_bytes())['files'].items():
        assert sha(source/name) == digest, name
    assert sha(manifest['split_protocol']) == manifest['split_protocol_sha256']
    partitions = json.loads(Path(manifest['split_protocol']).read_bytes())['row_ids']
    runtime = Path(train_spec['runtime'])
    environment = json.loads((runtime/'env_spec.json').read_bytes())
    assert hashlib.sha256(json.dumps(environment,sort_keys=True,separators=(',',':')).encode()).hexdigest() == train_spec['env_spec_sha256']
    model_source = Path(train_spec['model_source'])
    port = json.loads(Path(train_spec['source_port']).read_bytes())
    assert sha(train_spec['source_port']) == train_spec['source_port_sha256'] == receipt['source_port_sha256']
    for name,digest in port['files'].items():
        assert sha(model_source/name) == digest, name
    assert train_spec['source_query_read'] is True and receipt['source_query_read'] is True
    assert sha(root/'pvground_source_query.py') == train_spec['source_query_module_sha256'] == receipt['source_query_module_sha256']
    checkpoint = environment['weight_dirs']['scanrefer']
    assert sha(checkpoint['path']) == checkpoint['sha256'] == receipt['checkpoint_sha256']
    input_contract = json.loads((root/'formal_input_contract.json').read_bytes())
    assert input_contract['status'] == 'pass' and input_contract['expected_formal_rows'] == 9508
    assert input_contract['data_root'] == manifest['data_root']
    assert sha(input_contract['annotation_path']) == input_contract['annotation_sha256']
    assert sha(input_contract['scene_split_path']) == input_contract['scene_split_sha256']
    scan_ids = set(Path(input_contract['scene_split_path']).read_text().splitlines())
    raw_annos = [row for row in json.loads(Path(input_contract['annotation_path']).read_bytes()) if row['scene_id'] in scan_ids]
    raw_ids = [(r['scene_id'], int(r['object_id']), str(r.get('ann_id','')), ' '.join(r['token'])) for r in raw_annos]
    assert len(raw_ids) == 9508
    assert hashlib.sha256(json.dumps(raw_ids,ensure_ascii=False,separators=(',',':')).encode()).hexdigest() == input_contract['raw_identity_order_sha256']

    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    def reset_rng():
        random.seed(2027);np.random.seed(2027);torch.manual_seed(2027);torch.cuda.manual_seed_all(2027)

    reset_rng()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.chdir(str(source));sys.path.insert(0,str(source))
    from src.joint_det_dataset import Joint3DDataset
    from scripts.scanrefer_data_contract import verify_scanrefer_superpoints
    assert Path(sys.modules['src.joint_det_dataset'].__file__).resolve() == source/'src/joint_det_dataset.py'
    assert 'models' not in sys.modules
    os.chdir(str(model_source));sys.path.insert(0,str(model_source))
    from models.pv_ground import PVGround
    from main_utils import BaseTrainTester
    from prepare_data import DataProcessor
    from pcdet.config import cfg, cfg_from_yaml_file
    evaluator_path = runtime/'PV-Ground/src/grounding_evaluator.py'
    evaluator_spec = importlib.util.spec_from_file_location('pvground_official_evaluator',str(evaluator_path))
    evaluator_module = importlib.util.module_from_spec(evaluator_spec);evaluator_spec.loader.exec_module(evaluator_module)
    GroundingEvaluator = evaluator_module.GroundingEvaluator
    imports = {name:str(Path(sys.modules[name].__file__).resolve()) for name in
               ['src.joint_det_dataset','models.pv_ground','models.losses','main_utils','prepare_data']}
    for name in ['models.pv_ground','models.losses','main_utils','prepare_data']:
        assert model_source in Path(imports[name]).parents
    imports['evaluator'] = str(evaluator_path)
    write_json(root/'imports.json', {'files':imports,'sha256':{name:sha(path) for name,path in imports.items()}})
    payload = torch.load(checkpoint['path'],map_location='cpu')
    config = payload['config']
    parent = {key[7:]:value for key,value in payload['model'].items()}
    assert all(key.startswith('module.') for key in payload['model']) and len(parent) == 1234
    assert config.butd and not config.butd_cls and not config.butd_gt
    assert config.use_soft_token_loss and config.use_contrastive_align and config.num_decoder_layers == 6
    cfg_from_yaml_file(str(runtime/'PV-Ground/wandb_config.yaml'),cfg)
    model = PVGround(cfg,num_class=256,num_queries=256,num_decoder_layers=6,
        self_position_embedding=config.self_position_embedding,contrastive_align_loss=True,butd=True,
        pointnet_ckpt=None,data_path=manifest['data_root'],self_attend=config.self_attend)
    initial = expanded_parent_state(model, parent)
    delta = torch.load(str(training/'terminal.pth'),map_location='cpu')
    assert delta['step'] == 3723 and Counter(delta['row_ids']) == Counter(partitions['fit'])
    assert delta['parent_checkpoint_sha256'] == checkpoint['sha256']
    assert delta['spec_sha256'] == sha(training/'spec.json')
    assert delta['source_query_read'] is True
    assert delta['source_query_module_sha256'] == train_spec['source_query_module_sha256']
    assert delta['source_port_sha256'] == train_spec['source_port_sha256']
    terminal = terminal_state(model, initial, delta['state_delta'])
    model.cuda().eval()
    criterion,set_criterion = BaseTrainTester.get_criterion(config)
    processor = DataProcessor(cfg.DATA_PROCESSOR,np.asarray(cfg.DATA_CONFIG.POINT_CLOUD_RANGE),False,6)

    class FormalDataset(Joint3DDataset):
        def _scene_graph_parse(self,annos):
            identities = [(r['scan_id'],r['target_id'],str(r.get('ann_id','')),r['utterance']) for r in annos]
            assert identities == raw_ids
            super()._scene_graph_parse(annos)

        def __getitem__(self,index):
            sample = super().__getitem__(index)
            sample['formal_row_id'] = index
            assert np.isin(sample['gt_masks'],[0,1]).all()
            sample['gt_masks'] = sample['gt_masks'].astype(np.bool_)
            return sample

    print('PVG_FORMAL_DATASET_LOADING '+now(),flush=True)
    os.chdir(str(source))
    data_check = verify_scanrefer_superpoints(manifest['data_root'],'val',manifest['superpoint_files']['val'])
    assert data_check == input_contract['val_superpoints']
    dataset = FormalDataset(dataset_dict={'scanrefer':1},test_dataset='scanrefer',split='val',
        data_path=manifest['data_root'],use_color=True,use_height=False,use_multiview=False,
        detect_intermediate=True,butd=True,butd_cls=False,butd_gt=False,augment_det=False,skip_missing_superpoints=True)
    assert len(dataset) == 9508 and not dataset.augment and not dataset.augment_det
    write_json(root/'protocol.json',{'formal_rows':9508,'batch_size':8,'workers':2,'seed':2027,
        'primary_mode':'bbs','data_root':manifest['data_root'],'superpoints':data_check,
        'raw_identity_order_sha256':input_contract['raw_identity_order_sha256'],
        'parsed_identities':[(r['scan_id'],r['target_id'],r['utterance']) for r in dataset.annos],
        'source_query_read_by_arm':{'published_parent':False,'fit_terminal':True},
        'source_query_module_sha256':train_spec['source_query_module_sha256'],
        'source_port_sha256':train_spec['source_port_sha256'],
        'arms':['published_parent','fit_terminal'],'v99_reference':'historical protected complete system; not rerun in this comparison',
        'v99_rec_floor_hits':[5572,4797],'scan_mask_floor_percent':[58.70,50.70,44.72]})

    def prepare(raw):
        voxel_rows = [processor.forward({'points':pc.numpy().copy(),'use_lead_xyz':True}) for pc in raw['point_clouds']]
        voxel = processor.collate_batch(voxel_rows)
        count = len(raw['utterances'])
        assert np.array_equal(voxel['points'][:,1:].reshape(count,50000,6),raw['point_clouds'].numpy())
        batch = {k:v.cuda(non_blocking=True) if torch.is_tensor(v) else v for k,v in raw.items()}
        inputs = {k:torch.from_numpy(voxel[k]).float().cuda() for k in ['points','voxels','voxel_coords','voxel_num_points']}
        inputs.update(batch_size=count,text=raw['utterances'],det_boxes=batch['all_detected_boxes'],
            det_bbox_label_mask=batch['all_detected_bbox_label_mask'],det_class_ids=batch['all_detected_class_ids'],
            superpoint=batch['superpoint'],train=False)
        return inputs,batch

    @torch.no_grad()
    def evaluate(arm,state):
        assert arm in ['published_parent','fit_terminal']
        model.decoder[-1].source_query_read.enabled = (arm == 'fit_terminal')
        model.load_state_dict(state,strict=True);model.eval();reset_rng()
        evaluator = GroundingEvaluator(only_root=True,thresholds=[.25,.5],topks=[1,5,10],prefixes=['last_'],filter_non_gt_boxes=False,model='PVGround')
        loader = DataLoader(dataset,batch_size=8,shuffle=False,num_workers=2,pin_memory=True,
                            drop_last=False,generator=torch.Generator().manual_seed(2027))
        directory = root/arm;directory.mkdir()
        all_boxes = np.lib.format.open_memmap(str(directory/'boxes.npy'),mode='w+',dtype=np.float32,shape=(9508,256,6))
        all_scores = np.lib.format.open_memmap(str(directory/'scores.npy'),mode='w+',dtype=np.float32,shape=(9508,2,256))
        rows=[];begin=time.time()
        with (directory/'rows.jsonl').open('x') as stream:
            for raw in loader:
                inputs,batch=prepare(raw)
                output=model(inputs)
                raw_boxes=torch.cat([output['last_center'],output['last_pred_size']],-1)
                assert not set(output).intersection(batch)
                output.update(batch)
                loss,output=criterion(output,6,set_criterion,query_points_obj_topk=config.query_points_obj_topk)
                assert torch.isfinite(loss)
                for key in output:
                    if 'pred_size' in key:output[key]=output[key].clamp(min=1e-6)
                evaluator.evaluate(output,'last_')
                semantic=output['last_sem_cls_scores'].softmax(-1)
                projected=(torch.matmul(output['last_proj_queries'],output['proj_tokens'].transpose(-1,-2))/0.07).softmax(-1)
                contrastive=torch.zeros_like(semantic);contrastive[:,:,:projected.shape[-1]]=projected
                boxes=torch.cat([output['last_center'],output['last_pred_size']],-1)
                target=torch.cat([batch['center_label'][:,0,:3],batch['size_gts'][:,0]],-1)
                for bid in range(len(raw['utterances'])):
                    index=len(rows)
                    assert int(batch['formal_row_id'][bid])==index
                    record={'row_id':index,'scan_id':raw['scan_ids'][bid],'target_id':int(batch['target_id'][bid]),
                        'root_box':target[bid].cpu().tolist(),
                        'point_sha256':hashlib.sha256(raw['point_clouds'][bid].numpy().tobytes()).hexdigest()}
                    all_boxes[index]=raw_boxes[bid].cpu().numpy()
                    lo=torch.maximum(boxes[bid,:,:3]-boxes[bid,:,3:]/2,target[bid,:3]-target[bid,3:]/2)
                    hi=torch.minimum(boxes[bid,:,:3]+boxes[bid,:,3:]/2,target[bid,:3]+target[bid,3:]/2)
                    intersection=(hi-lo).clamp(min=0).prod(-1)
                    iou=intersection/(boxes[bid,:,3:].prod(-1)+target[bid,3:].prod()-intersection)
                    assert torch.isfinite(iou).all()
                    for mode_index,(mode,probability) in enumerate([('bbs',semantic),('bbf',contrastive)]):
                        score=(probability[bid]*(batch['positive_map'][bid,0]>0)).sum(-1)
                        for key in ['modify_positive_map','pron_positive_map','rel_positive_map']:
                            score=score+(probability[bid]*batch[key][bid,0]).sum(-1)
                        score=score-(probability[bid]*batch['other_entity_map'][bid,0]).sum(-1)
                        all_scores[index,mode_index]=score.cpu().numpy()
                        query=int(score.argsort(descending=True)[0])
                        alpha=output['adaptive_weights'][bid]
                        mask=((alpha*output['last_pred_masks'][bid][0,query]+(1-alpha)*output['sp_last_pred_masks'][bid][query]).sigmoid()>.5)[output['superpoints'][bid]]
                        truth=batch['gt_masks'][bid,0].bool()
                        mask_iou=float((mask&truth).sum().float()/(mask|truth).sum())
                        record[mode]={'query':query,'box':boxes[bid,query].cpu().tolist(),'iou':float(iou[query]),'mask_iou':mask_iou}
                    rows.append(record);stream.write(json.dumps(record,allow_nan=False)+'\n')
                if len(rows)%512<8:
                    stream.flush();print('PVG_FORMAL_PROGRESS '+json.dumps({'arm':arm,'rows':len(rows),'total':9508,'seconds':time.time()-begin}),flush=True)
                del output,inputs,batch,raw
        assert len(rows)==9508
        all_boxes.flush();all_scores.flush()
        metrics={mode:row_metrics(rows,mode) for mode in ['bbs','bbf']}
        for mode in ['bbs','bbf']:
            for threshold,suffix in [(.25,'25'),(.5,'50')]:
                assert metrics[mode]['rec_hits'+suffix]==evaluator.dets[('last_',threshold,1,mode)]
                assert evaluator.gts[('last_',threshold,1,mode)]==9508
            key='mask_pos' if mode=='bbs' else 'mask_sem'
            assert abs(metrics[mode]['mask_iou_sum']-float(evaluator.dets[key]))<1e-3
        assert all(torch.equal(value.detach().cpu(),state[key]) for key,value in model.state_dict().items())
        result={'status':'pass','arm':arm,'rows':9508,'formal_rows':9508,'metrics':metrics,'time_cst':now(),
            'elapsed_seconds':time.time()-begin,'state_unchanged':True,
            'rows_sha256':sha(directory/'rows.jsonl'),'boxes_sha256':sha(directory/'boxes.npy'),'scores_sha256':sha(directory/'scores.npy')}
        write_json(directory/'receipt.json',result)
        print('PVG_FORMAL_ARM_COMPLETE '+json.dumps(result),flush=True)
        return rows,result

    before,parent_result=evaluate('published_parent',initial)
    after,terminal_result=evaluate('fit_terminal',terminal)
    transitions={}
    for mode in ['bbs','bbf']:
        transitions[mode]={}
        for threshold in [.25,.5]:
            fixes=breaks=0
            for old,new in zip(before,after):
                assert all(old[key]==new[key] for key in ['row_id','scan_id','target_id','point_sha256','root_box'])
                fixes+=old[mode]['iou']<=threshold<new[mode]['iou']
                breaks+=new[mode]['iou']<=threshold<old[mode]['iou']
            transitions[mode][str(threshold)]={'fixes':fixes,'breaks':breaks,'net':fixes-breaks}
    assert sha(checkpoint['path'])==checkpoint['sha256'] and sha(training/'terminal.pth')==receipt['terminal_sha256']
    result={'status':'complete','time_cst':now(),'formal_rows':9508,'rows_per_arm':9508,'optimizer_steps':0,'new_checkpoint_files':0,
        'primary_mode':'bbs','all_model_states_unchanged':True,
        'metrics':{'published_parent':parent_result['metrics'],'fit_terminal':terminal_result['metrics']},
        'transitions':transitions,'promotion':promotion_check(terminal_result['metrics']['bbs']),
        'parent_checkpoint_sha256':checkpoint['sha256'],'terminal_checkpoint_sha256':receipt['terminal_sha256'],
        'training_receipt_sha256':sha(training/'receipt.json'),'training_audit_sha256':sha(audit_root/'audit.json'),
        'spec_sha256':sha(args.spec),'script_sha256':sha(__file__),'protocol_sha256':sha(root/'protocol.json')}
    write_json(root/'receipt.json',result)
    print('PVG_FORMAL_COMPLETE '+json.dumps(result),flush=True)


if __name__=='__main__':
    main()
