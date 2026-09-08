"""One fixed ScanRefer fit pass from the published full PV-Ground checkpoint."""
import argparse
from collections import Counter
import copy
import datetime
import hashlib
import importlib.util
import json
import math
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
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')


def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_bytes())
    output = Path(spec['root'])
    runtime = Path(spec['runtime'])
    manifest = json.loads(Path(spec['input_manifest']).read_bytes())
    dataset_source = Path(manifest['model_source'])
    assert sha(dataset_source/'appearance_source_manifest.json') == manifest['source_manifest_sha256']
    for name, digest in json.loads((dataset_source/'appearance_source_manifest.json').read_bytes())['files'].items():
        assert sha(dataset_source/name) == digest, name
    assert sha(manifest['split_protocol']) == manifest['split_protocol_sha256']
    partitions = json.loads(Path(manifest['split_protocol']).read_bytes())['row_ids']
    assert len(partitions['fit'])==29778 and len(partitions['holdout'])==6887
    assert not set(partitions['fit']).intersection(partitions['holdout'])
    env = json.loads((runtime/'env_spec.json').read_bytes())
    env_sha = hashlib.sha256(json.dumps(env,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    assert env_sha == spec['env_spec_sha256']
    interface = json.loads(Path(spec['training_interface_receipt']).read_bytes())
    assert interface['status']=='pass' and interface['optimizer_steps']==2
    assert interface['env_spec_sha256']==env_sha
    upstream = json.loads((runtime/'source_bundle_receipt.json').read_bytes())
    model_source = Path(spec['model_source'])
    port = json.loads(Path(spec['source_port']).read_bytes())
    for name, entry in upstream['sources']['PV-Ground']['files'].items():
        expected=port['after_sha256'] if 'PV-Ground/'+name==port['file'] else entry['sha256']
        assert sha(model_source/name)==expected,name
    checkpoint=env['weight_dirs']['scanrefer']
    assert sha(checkpoint['path'])==checkpoint['sha256']==spec['checkpoint_sha256']
    assert spec['batch_size']==8 and spec['fit_passes']==1 and spec['seed']==2027
    assert spec['lr']==spec['lr_backbone']==1e-5

    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset

    def reset_rng(seed):
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)

    reset_rng(spec['seed'])
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    # Keep the verified project's Dataset namespace, and import the official model
    # and evaluator from explicit source paths. Do not depend on accidental sys.path ordering.
    os.chdir(str(dataset_source))
    sys.path.insert(0,str(dataset_source))
    from src.joint_det_dataset import Joint3DDataset
    from scripts.scanrefer_data_contract import verify_scanrefer_superpoints
    assert Path(sys.modules['src.joint_det_dataset'].__file__).resolve()==dataset_source/'src/joint_det_dataset.py'
    assert 'models' not in sys.modules
    os.chdir(str(model_source))
    sys.path.insert(0,str(model_source))
    from models.pv_ground import PVGround
    from main_utils import BaseTrainTester
    from prepare_data import DataProcessor
    from pcdet.config import cfg, cfg_from_yaml_file
    evaluator_path=runtime/'PV-Ground/src/grounding_evaluator.py'
    module_spec=importlib.util.spec_from_file_location('pvground_official_evaluator',str(evaluator_path))
    evaluator_module=importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(evaluator_module)
    GroundingEvaluator=evaluator_module.GroundingEvaluator
    imported = {name: str(Path(sys.modules[name].__file__).resolve()) for name in
        ['src.joint_det_dataset','models.pv_ground','models.losses','main_utils','prepare_data']}
    for name in ['models.pv_ground','models.losses','main_utils','prepare_data']:
        assert model_source in Path(imported[name]).parents
    imported['evaluator']=str(evaluator_path)
    write_json(output/'imports.json',{'files':imported,'sha256':{k:sha(v) for k,v in imported.items()}})

    payload=torch.load(checkpoint['path'],map_location='cpu')
    config=payload['config']
    initial={k[7:]:v for k,v in payload['model'].items()}
    assert all(k.startswith('module.') for k in payload['model']) and len(initial)==1234
    assert config.butd and not config.butd_cls and not config.butd_gt
    assert config.use_soft_token_loss and config.use_contrastive_align
    cfg_from_yaml_file(str(runtime/'PV-Ground/wandb_config.yaml'),cfg)
    model=PVGround(cfg,num_class=256,num_queries=256,num_decoder_layers=6,
        self_position_embedding=config.self_position_embedding,contrastive_align_loss=True,butd=True,
        pointnet_ckpt=None,data_path=manifest['data_root'],self_attend=config.self_attend)
    position_ids=torch.arange(model.text_encoder.config.max_position_embeddings).expand((1,-1))
    assert set(model.state_dict())-set(initial)=={'text_encoder.embeddings.position_ids'}
    assert not set(initial)-set(model.state_dict())
    assert torch.equal(model.text_encoder.embeddings.position_ids,position_ids)
    model.text_encoder.embeddings.register_buffer('position_ids',model.text_encoder.embeddings.position_ids,persistent=False)
    model.load_state_dict(initial,strict=True)
    model.cuda()
    training=copy.copy(config)
    training.frozen=False;training.small_lr=False
    training.lr=spec['lr'];training.lr_backbone=spec['lr_backbone']
    assert training.weight_decay==0.0005 and training.clip_norm==0.1
    criterion,set_criterion=BaseTrainTester.get_criterion(training)
    processors={mode:DataProcessor(cfg.DATA_PROCESSOR,np.asarray(cfg.DATA_CONFIG.POINT_CLOUD_RANGE),mode=='train',6)
                for mode in ['train','eval']}
    trainable={n:p for n,p in model.named_parameters() if p.requires_grad}
    frozen={n:p for n,p in model.named_parameters() if not p.requires_grad}
    assert len(trainable)==783 and sum(p.numel() for p in trainable.values())==27959611

    class FitDataset(Joint3DDataset):
        def _scene_graph_parse(self,annos):
            assert len(annos)==36665
            actual={'fit':[],'holdout':[]}
            for index,row in enumerate(annos):
                row['_local_training_id']=index
                code=(manifest['split_salt']+'\0'+row['scan_id'].split('_')[0]).encode()
                fold=int(hashlib.sha256(code).hexdigest()[:8],16)%5
                actual['holdout' if fold==0 else 'fit'].append(index)
            assert actual==partitions
            super()._scene_graph_parse(annos)

        def __getitem__(self,index):
            result=super().__getitem__(index)
            result['local_training_id']=self.annos[index]['_local_training_id']
            assert np.isin(result['gt_masks'],[0,1]).all()
            result['gt_masks']=result['gt_masks'].astype(np.bool_)
            return result

    print('PVG_FINETUNE_DATASET_LOADING '+now(),flush=True)
    os.chdir(str(dataset_source))
    verify_scanrefer_superpoints(manifest['data_root'],'train',manifest['superpoint_files']['train'])
    dataset=FitDataset(dataset_dict={'scanrefer':1},test_dataset='scanrefer',split='train',
        data_path=manifest['data_root'],use_color=True,use_height=False,use_multiview=False,
        detect_intermediate=True,butd=True,butd_cls=False,butd_gt=False,augment_det=False,skip_missing_superpoints=True)
    assert [r['_local_training_id'] for r in dataset.annos]==list(range(36665))
    physical={part:{dataset.annos[i]['scan_id'].split('_')[0] for i in ids} for part,ids in partitions.items()}
    assert not physical['fit'].intersection(physical['holdout'])
    dataset.augment=False
    reference=json.loads((Path(spec['reference_fixtures'])/'receipt.json').read_bytes())
    reset_rng(spec['seed'])
    for row in reference['rows']:
        sample=dataset[row['training_row_id']]
        checks={'point_clouds':sample['point_clouds'],'det_boxes':sample['all_detected_boxes'],
            'det_bbox_label_mask':sample['all_detected_bbox_label_mask'],'det_class_ids':sample['all_detected_class_ids'],
            'superpoint':sample['superpoint'].numpy()}
        assert sample['utterances']==row['text']
        for name,array in checks.items():assert hashlib.sha256(array.tobytes()).hexdigest()==row['tensor_sha256'][name],name
    print('PVG_FULL_DATASET_INPUTS_PASS '+json.dumps({'fit':len(partitions['fit']),'holdout':len(partitions['holdout']),
        'physical_fit':len(physical['fit']),'physical_holdout':len(physical['holdout']),'reference_rows':4}),flush=True)

    def loader(part,shuffle,seed):
        return DataLoader(Subset(dataset,partitions[part]),batch_size=8,shuffle=shuffle,num_workers=2,
            generator=torch.Generator().manual_seed(seed),pin_memory=True,drop_last=False)

    def prepare(batch,mode):
        voxel_rows=[processors[mode].forward({'points':pc.numpy().copy(),'use_lead_xyz':True}) for pc in batch['point_clouds']]
        voxels=processors[mode].collate_batch(voxel_rows)
        bs=len(batch['utterances'])
        assert np.array_equal(voxels['points'][:,1:].reshape(bs,50000,6),batch['point_clouds'].numpy())
        batch={k:v.cuda(non_blocking=True) if torch.is_tensor(v) else v for k,v in batch.items()}
        inputs={k:torch.from_numpy(voxels[k]).float().cuda() for k in ['points','voxels','voxel_coords','voxel_num_points']}
        inputs.update(batch_size=bs,text=batch['utterances'],det_boxes=batch['all_detected_boxes'],
            det_bbox_label_mask=batch['all_detected_bbox_label_mask'],det_class_ids=batch['all_detected_class_ids'],
            superpoint=batch['superpoint'])
        return inputs,batch

    def native_loss(predictions,batch):
        assert not set(predictions).intersection(batch)
        predictions.update(batch)
        loss,predictions=criterion(predictions,6,set_criterion,query_points_obj_topk=training.query_points_obj_topk)
        assert torch.isfinite(loss)
        return loss,predictions

    def step(batch,optimizer,update):
        begin=time.time()
        inputs,batch=prepare(batch,'train')
        predictions=model(inputs)
        loss,predictions=native_loss(predictions,batch)
        optimizer.zero_grad()
        loss.backward()
        for name,p in trainable.items():
            if p.grad is not None:assert torch.isfinite(p.grad).all(),name
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),training.clip_norm)
        assert torch.isfinite(norm)
        if update:optimizer.step()
        torch.cuda.synchronize()
        return {'loss':float(loss),'grad_norm':float(norm),'seconds':time.time()-begin,
            'rows':batch['local_training_id'].cpu().tolist(),
            'loss_bbox':float(predictions['loss_bbox']),'loss_giou':float(predictions['loss_giou']),
            'loss_ce':float(predictions['loss_ce']),'loss_sem_align':float(predictions['loss_sem_align'])}

    dataset.augment=True;dataset.augment_det=True
    reset_rng(spec['seed'])
    model.train()
    optimizer=BaseTrainTester.get_optimizer(training,model)
    capacity_loader=loader('fit',True,spec['seed'])
    torch.cuda.reset_peak_memory_stats()
    capacity=step(next(iter(capacity_loader)),optimizer,False)
    capacity.update(status='pass',optimizer_steps=0,batch_size=8,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),time_cst=now())
    write_json(output/'capacity.json',capacity)
    print('PVG_BATCH8_CAPACITY_PASS '+json.dumps(capacity),flush=True)
    del capacity_loader,optimizer
    model.load_state_dict(initial,strict=True);model.zero_grad(set_to_none=True)
    assert all(torch.equal(v.detach().cpu(),initial[k]) for k,v in model.state_dict().items())
    reset_rng(spec['seed'])

    @torch.no_grad()
    def evaluate(stage):
        dataset.augment=False;dataset.augment_det=False
        model.eval();reset_rng(spec['seed'])
        evaluator=GroundingEvaluator(only_root=True,thresholds=[.25,.5],topks=[1,5,10],prefixes=['last_'],filter_non_gt_boxes=False,model='PVGround')
        directory=output/stage;directory.mkdir()
        n=len(partitions['holdout'])
        all_boxes=np.lib.format.open_memmap(str(directory/'boxes.npy'),mode='w+',dtype=np.float32,shape=(n,256,6))
        all_scores=np.lib.format.open_memmap(str(directory/'scores.npy'),mode='w+',dtype=np.float32,shape=(n,2,256))
        rows=[];begin=time.time()
        with (directory/'rows.jsonl').open('w') as stream:
            for batch in loader('holdout',False,spec['seed']):
                inputs,batch=prepare(batch,'eval');inputs['train']=False
                predictions=model(inputs)
                raw_boxes=torch.cat([predictions['last_center'],predictions['last_pred_size']],-1)
                _,predictions=native_loss(predictions,batch)
                for key in predictions:
                    if 'pred_size' in key:predictions[key]=predictions[key].clamp(min=1e-6)
                evaluator.evaluate(predictions,'last_')
                prob=predictions['last_sem_cls_scores'].softmax(-1)
                cr=(torch.matmul(predictions['last_proj_queries'],predictions['proj_tokens'].transpose(-1,-2))/0.07).softmax(-1)
                cp=torch.zeros_like(prob);cp[:,:,:cr.shape[-1]]=cr
                boxes=torch.cat([predictions['last_center'],predictions['last_pred_size']],-1)
                gt=torch.cat([batch['center_label'][:,0,:3],batch['size_gts'][:,0]],-1)
                for bid in range(len(batch['utterances'])):
                    row_id=int(batch['local_training_id'][bid])
                    index=len(rows)
                    assert row_id==partitions['holdout'][index]
                    all_boxes[index]=raw_boxes[bid].cpu().numpy()
                    record={'row_id':row_id,'scan_id':batch['scan_ids'][bid],'target_id':int(batch['target_id'][bid]),
                        'root_box':gt[bid].cpu().tolist(),'point_sha256':hashlib.sha256(batch['point_clouds'][bid].cpu().numpy().tobytes()).hexdigest()}
                    lo=torch.maximum(boxes[bid,:,:3]-boxes[bid,:,3:]/2,gt[bid,:3]-gt[bid,3:]/2)
                    hi=torch.minimum(boxes[bid,:,:3]+boxes[bid,:,3:]/2,gt[bid,:3]+gt[bid,3:]/2)
                    intersection=(hi-lo).clamp(min=0).prod(-1)
                    iou=intersection/(boxes[bid,:,3:].prod(-1)+gt[bid,3:].prod()-intersection)
                    assert torch.isfinite(iou).all()
                    for mode_index,(mode,probabilities) in enumerate([('bbs',prob),('bbf',cp)]):
                        score=(probabilities[bid]*(batch['positive_map'][bid,0]>0)).sum(-1)
                        for name in ['modify_positive_map','pron_positive_map','rel_positive_map']:
                            score=score+(probabilities[bid]*batch[name][bid,0]).sum(-1)
                        score=score-(probabilities[bid]*batch['other_entity_map'][bid,0]).sum(-1)
                        all_scores[index,mode_index]=score.cpu().numpy()
                        ranked=score.argsort(descending=True);q=int(ranked[0])
                        alpha=predictions['adaptive_weights'][bid]
                        mask=((alpha*predictions['last_pred_masks'][bid][0,q]+(1-alpha)*predictions['sp_last_pred_masks'][bid][q]).sigmoid()>.5)[predictions['superpoints'][bid]]
                        truth=batch['gt_masks'][bid,0].bool()
                        mask_iou=float((mask&truth).sum().float()/(mask|truth).sum())
                        record[mode]={'query':q,'box':boxes[bid,q].cpu().tolist(),'iou':float(iou[q]),'mask_iou':mask_iou,
                            'oracle25':[int((iou[ranked[:k]]>.25).any()) for k in [16,32,64,256]],
                            'oracle50':[int((iou[ranked[:k]]>.5).any()) for k in [16,32,64,256]]}
                    rows.append(record);stream.write(json.dumps(record)+'\n')
                if len(rows)%512<8:
                    stream.flush();print('PVG_EVAL_PROGRESS '+json.dumps({'stage':stage,'rows':len(rows),'total':n,'seconds':time.time()-begin}),flush=True)
                del predictions,inputs,batch
        assert len(rows)==n
        all_boxes.flush();all_scores.flush()
        metrics={}
        for mode in ['bbs','bbf']:
            hit25=sum(r[mode]['iou']>.25 for r in rows);hit50=sum(r[mode]['iou']>.5 for r in rows)
            assert hit25==evaluator.dets[('last_',.25,1,mode)] and hit50==evaluator.dets[('last_',.5,1,mode)]
            key='mask_pos' if mode=='bbs' else 'mask_sem'
            mask_sum=sum(r[mode]['mask_iou'] for r in rows)
            assert abs(mask_sum-float(evaluator.dets[key]))<1e-3
            metrics[mode]={'rec_hits25':hit25,'rec_hits50':hit50,'mask_hits25':sum(r[mode]['mask_iou']>.25 for r in rows),
                'mask_hits50':sum(r[mode]['mask_iou']>.5 for r in rows),'mask_iou_sum':mask_sum,'mask_miou':mask_sum/n*100}
        receipt={'status':'pass','stage':stage,'rows':n,'metrics':metrics,'elapsed_seconds':time.time()-begin,
            'time_cst':now(),'formal_rows':0,'rows_sha256':sha(directory/'rows.jsonl'),
            'boxes_sha256':sha(directory/'boxes.npy'),'scores_sha256':sha(directory/'scores.npy')}
        write_json(directory/'receipt.json',receipt)
        print('PVG_EVAL_COMPLETE '+json.dumps(receipt),flush=True)
        return rows,receipt

    initial_rows,initial_receipt=evaluate('initial')
    assert all(torch.equal(v.detach().cpu(),initial[k]) for k,v in model.state_dict().items())
    reset_rng(spec['seed']);dataset.augment=True;dataset.augment_det=True;model.train()
    optimizer=BaseTrainTester.get_optimizer(training,model)
    selected_names=set(trainable)|{n for n,_ in model.named_buffers() if n in initial}

    def save_checkpoint(name,step_number,seen_rows):
        current=model.state_dict()
        delta={k:v.detach().cpu() for k,v in current.items() if k in selected_names}
        data={'parent_checkpoint_sha256':checkpoint['sha256'],'state_delta':delta,'optimizer':optimizer.state_dict(),
            'step':step_number,'row_ids':seen_rows,'spec_sha256':sha(args.spec),'torch_rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state_all(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate()}
        temporary=output/(name+'.tmp')
        torch.save(data,str(temporary));os.replace(str(temporary),str(output/name))

    seen=[];begin=time.time();total=math.ceil(len(partitions['fit'])/8)
    with (output/'train.jsonl').open('w') as stream:
        for index,batch in enumerate(loader('fit',True,spec['seed']),1):
            record=step(batch,optimizer,True);seen.extend(record['rows'])
            record.update(step=index,total_steps=total,cumulative_seconds=time.time()-begin)
            stream.write(json.dumps(record)+'\n')
            if index==1 or index%64==0:
                stream.flush();print('PVG_TRAIN_PROGRESS '+json.dumps(record),flush=True)
            if index%512==0:save_checkpoint('latest.pth',index,seen)
    assert index==3723 and Counter(seen)==Counter(partitions['fit'])
    assert not set(seen).intersection(partitions['holdout'])
    assert all(torch.equal(p.detach().cpu(),initial[n]) for n,p in frozen.items())
    save_checkpoint('terminal.pth',index,seen)
    print('PVG_FIT_COMPLETE '+json.dumps({'steps':index,'rows':len(seen),'seconds':time.time()-begin,'time_cst':now()}),flush=True)
    final_rows,final_receipt=evaluate('terminal')
    transitions={}
    for mode in ['bbs','bbf']:
        transitions[mode]={}
        for threshold in [.25,.5]:
            fixes=breaks=0
            for old,new in zip(initial_rows,final_rows):
                assert old['row_id']==new['row_id'] and old['point_sha256']==new['point_sha256'] and old['root_box']==new['root_box']
                before=old[mode]['iou']>threshold;after=new[mode]['iou']>threshold
                fixes+=not before and after;breaks+=before and not after
            transitions[mode][str(threshold)]={'fixes':fixes,'breaks':breaks,'net':fixes-breaks}
    assert sha(checkpoint['path'])==checkpoint['sha256']
    changed=[n for n,p in trainable.items() if not torch.equal(p.detach().cpu(),initial[n])]
    receipt={'status':'complete','time_cst':now(),'training_steps':index,'fit_rows':len(seen),'holdout_rows':len(final_rows),
        'formal_rows':0,'primary_mode':'bbs','initial':initial_receipt['metrics'],'terminal':final_receipt['metrics'],
        'transitions':transitions,'primary_rec_nonregression':all(transitions['bbs'][str(t)]['net']>=0 for t in [.25,.5]),
        'checkpoint_sha256':checkpoint['sha256'],'terminal_sha256':sha(output/'terminal.pth'),
        'changed_parameter_tensors':len(changed),'frozen_parameters_unchanged':True,'fit_seen_exactly_once':True,
        'train_log_sha256':sha(output/'train.jsonl'),'spec_sha256':sha(args.spec),'script_sha256':sha(__file__)}
    write_json(output/'receipt.json',receipt)
    print('PVG_FINETUNE_COMPLETE '+json.dumps(receipt),flush=True)


if __name__=='__main__':
    main()
