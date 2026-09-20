import json,torch
from pathlib import Path
r=Path(__file__).resolve().parent
ck=torch.load(str(r/'official_scanrefer.pth'),map_location='cpu')
config=ck['config'];cfg=vars(config) if hasattr(config,'__dict__') else config
keys=['dataset','test_dataset','num_queries','num_decoder_layers','self_position_embedding','self_attend','use_color','use_height','use_multiview','butd','butd_cls','butd_gt','detect_intermediate','contrastive_align_loss','batch_size','seed','model','two_stage','obj_name','hidden_dim','voxel_size','no_lang_cls','use_aug','pointnet_ckpt','reduce_intermediate','num_target','num_encoder_layers','query_points_obj_topk','sampling','rng_seed','use_contrastive_align','use_soft_token_loss','wo_obj_name','joint_det','eval_train','augment_det']
rec={'epoch':ck['epoch'],'config_type':type(config).__name__,'settings':{k:cfg[k] for k in keys if k in cfg},'available_keys':sorted(cfg)}
(r/'checkpoint_config_audit.json').write_text(json.dumps(rec,indent=2,default=str));print(json.dumps(rec,indent=2,default=str))
