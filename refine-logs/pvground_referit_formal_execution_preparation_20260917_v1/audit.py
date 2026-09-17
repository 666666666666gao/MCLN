"""Independently recount fixed ReferIt formal native outputs and REC targets."""
import argparse
import datetime
import json
from pathlib import Path

from pvground_native_output_recount import recount_native_rows, sha


def audit(root):
    assert (root/'evaluation.exit').read_text().strip()=='0'
    receipt=json.loads((root/'receipt.json').read_bytes())
    spec=json.loads((root/'spec.json').read_bytes())
    protocol=json.loads((root/'protocol.json').read_bytes())
    dataset=spec['dataset']
    nformal={'nr3d':7899,'sr3d':17726}[dataset]
    floor25,floor50={'nr3d':(4726,4059),'sr3d':(12139,10335)}[dataset]
    assert protocol['dataset']==receipt['dataset']==dataset
    assert receipt['status']=='complete'
    assert receipt['formal_rows']==receipt['rows_per_arm']==protocol['formal_rows']==nformal
    assert receipt['optimizer_steps']==receipt['new_checkpoint_files']==0
    assert receipt['all_model_states_unchanged'] and receipt['primary_mode']==protocol['primary_mode']=='bbs'
    assert sha(root/'spec.json')==receipt['spec_sha256']
    assert sha(root/'evaluate.py')==receipt['script_sha256']==spec['files']['evaluate.py']
    assert sha(root/'protocol.json')==receipt['protocol_sha256']
    for name,digest in spec['files'].items():
        assert sha(root/name)==digest,name
    training=Path(spec['training_root'])
    audited=Path(spec['training_audit_root'])
    assert sha(training/'receipt.json')==receipt['training_receipt_sha256']
    assert sha(training/'terminal.pth')==receipt['terminal_checkpoint_sha256']
    assert sha(audited/'audit.json')==receipt['training_audit_sha256']
    endpoint=json.loads((audited/'audit.json').read_bytes())
    assert endpoint['integrity_pass'] and endpoint['primary_rec_nonregression'] and endpoint['rec_competition_verified']
    assert protocol['batch_size']==8 and protocol['workers']==2 and protocol['seed']==2027
    assert protocol['source_query_read_by_arm']=={'published_parent':False,'fit_terminal':True}
    train_spec=json.loads((training/'spec.json').read_bytes())
    assert train_spec['dataset']==dataset
    assert sha(training/'spec.json')==spec['training_spec_sha256']
    assert sha(train_spec['checkpoint']['path'])==receipt['parent_checkpoint_sha256']==train_spec['checkpoint_sha256']
    contract=json.loads((root/'formal_input_contract.json').read_bytes())
    assert contract['dataset']==dataset and contract['expected_formal_rows']==nformal
    assert protocol['raw_identity_order_sha256']==contract['raw_identity_order_sha256']
    assert protocol['input_protocol']==contract['input_protocol']
    assert protocol['source_query_module_sha256']==train_spec['source_query_module_sha256']
    assert protocol['observation_module_sha256']==train_spec['observation_module_sha256']
    assert protocol['task_module_sha256']==train_spec['task_module_sha256']
    assert protocol['task_read_by_arm']=={'published_parent':False,'fit_terminal':True}
    assert protocol['observation_state_by_arm']=={'published_parent':False,'fit_terminal':True}
    assert protocol['source_port_sha256']==train_spec['source_port_sha256']
    restore=json.loads((root/'terminal_restore.json').read_bytes())
    assert restore['status']=='pass' and restore['strict_terminal_and_parent_restore']
    assert restore['added_tensors']==37 and restore['scope']=='actual fixed terminal delta'
    assert restore['dataset']==dataset and restore['state_tensors']==1272
    assert restore['terminal_sha256']==receipt['terminal_checkpoint_sha256']
    assert protocol['rec_floor_hits']==[floor25,floor50]
    assert protocol['mask_gate'] is False
    identities=protocol['parsed_identities']
    assert len(identities)==nformal
    stages={};records={};metrics={}
    for arm in ['published_parent','fit_terminal']:
        directory=root/arm
        arm_receipt=json.loads((directory/'receipt.json').read_bytes())
        assert arm_receipt['arm']==arm and arm_receipt['formal_rows']==nformal and arm_receipt['state_unchanged']
        rows,actual,checks=recount_native_rows(directory)
        assert [row['row_id'] for row in rows]==list(range(nformal))
        for row,identity in zip(rows,identities):
            assert row['scan_id']==identity[0] and row['target_id']==identity[1]
        assert actual==receipt['metrics'][arm]
        records[arm]=rows;metrics[arm]=actual;stages[arm]=checks
    before,after=records['published_parent'],records['fit_terminal']
    transitions={};bands={}
    for mode in ['bbs','bbf']:
        transitions[mode]={};bands[mode]=[[0]*3 for _ in range(3)]
        for old,new in zip(before,after):
            for key in ['row_id','scan_id','target_id','root_box','point_sha256']:
                assert old[key]==new[key],key
            i=int(old[mode]['iou']>.25)+int(old[mode]['iou']>.5)
            j=int(new[mode]['iou']>.25)+int(new[mode]['iou']>.5)
            bands[mode][i][j]+=1
        for threshold in [.25,.5]:
            fixes=sum(old[mode]['iou']<=threshold<new[mode]['iou'] for old,new in zip(before,after))
            breaks=sum(new[mode]['iou']<=threshold<old[mode]['iou'] for old,new in zip(before,after))
            transitions[mode][str(threshold)]={'fixes':fixes,'breaks':breaks,'net':fixes-breaks}
    assert transitions==receipt['transitions']
    candidate=metrics['fit_terminal']['bbs']
    checks={'rec25_target':candidate['rec_hits25']>=floor25,
        'rec50_target':candidate['rec_hits50']>=floor50}
    assert checks==receipt['promotion']['checks']
    assert all(checks.values())==receipt['promotion']['rec_target_pass']
    assert receipt['promotion']['requires_independent_formal_audit']
    assert receipt['promotion']['mask_gate'] is False
    return {'integrity_pass':True,'time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        'formal_rows':nformal,'metrics':metrics,'transitions':transitions,'transition_counts':bands,
        'transition_bands':['[0,0.25]','(0.25,0.50]','(0.50,1]'],'checks':checks,
        'rec_target_pass':all(checks.values()),'mask_gate':False,'dataset':dataset,'stages':stages,
        'mask_audit_scope':'exported per-row IoU recount; binary predictions not reloaded',
        'reference':'fixed dataset REC target; parent rerun separately reported',
        'receipt_sha256':sha(root/'receipt.json'),'auditor_sha256':sha(__file__)}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    result=audit(args.root)
    with args.out.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print('PVG_FORMAL_AUDIT_COMPLETE '+json.dumps(result),flush=True)


if __name__=='__main__':
    main()
