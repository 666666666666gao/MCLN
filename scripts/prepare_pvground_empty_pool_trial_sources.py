"""Prepare one fixed empty-pool control from the completed native trial sources."""
from pathlib import Path

repo = Path(__file__).resolve().parents[1]

def replace_once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)

train = (repo/'scripts/run_pvground_scanrefer_vsa_order.py').read_text()
train = replace_once(train, "    assert interface['env_spec_sha256']==env_sha", "    assert interface['env_spec_sha256']==env_sha\n    assert interface['empty_pool_mask'] is True\n    assert spec['empty_pool_mask'] is True\n    assert interface['module_sha256']==spec['empty_pool_module_sha256']==sha(output/'pvground_empty_pool_mask.py')")
train = replace_once(train, '    position_ids=torch.arange', '    from pvground_empty_pool_mask import install_empty_pool_mask\n    install_empty_pool_mask(model,True)\n    position_ids=torch.arange')
train = replace_once(train, "        temporary=output/(name+'.tmp')", "        data.update(empty_pool_mask=True,empty_pool_module_sha256=spec['empty_pool_module_sha256'])\n        temporary=output/(name+'.tmp')")
train = replace_once(train, "    write_json(output/'receipt.json',receipt)", "    receipt.update(empty_pool_mask=True,empty_pool_module_sha256=spec['empty_pool_module_sha256'])\n    write_json(output/'receipt.json',receipt)")
(repo/'scripts/run_pvground_scanrefer_empty_pool.py').write_text(train, encoding='utf-8', newline='\n')

launch = (repo/'scripts/launch_pvground_vsa_order_training.py').read_text()
launch = launch.replace('20260908_vsaorder_v1', '20260908_emptypool_v1').replace('20260908_detalign_v1', '20260908_vsaorder_v1')
launch = replace_once(launch, "assert free>3*1024**3", "assert free>int(2.25*1024**3)")
anchor = "spec.update(root=train,model_source=prepared['model_source'],source_port=prepared['source_port'])"
launch = replace_once(launch, anchor, anchor+"\ninterface='/root/autodl-tmp/mcln_pvground_empty_pool_interface_20260908_v1'\nwith s.open(interface+'/controller.exit','rb') as f:assert f.read().strip()==b'0'\nwith s.open(interface+'/results/receipt.json','rb') as f:checked=json.loads(f.read())\nmodule=(repo/'models/pvground_empty_pool_mask.py').read_bytes()\nmodule_sha=hashlib.sha256(module).hexdigest()\nassert checked['status']=='pass' and checked['optimizer_steps']==2 and checked['empty_pool_mask']\nassert checked['module_sha256']==module_sha\nspec.update(training_interface_receipt=interface+'/results/receipt.json',empty_pool_mask=True,empty_pool_module_sha256=module_sha,\n    native_control_root='/root/autodl-tmp/mcln_pvground_scanrefer_finetune_20260908_vsaorder_v1',\n    disk_budget=dict(new_artifacts_bytes=int(1.25*1024**3),reserve_bytes=1024**3,free_before=free),\n    comparison='own initial and completed same-budget native control; primary bbs; unchanged formal V99 and Mask floors')")
launch = launch.replace('scripts/run_pvground_scanrefer_vsa_order.py','scripts/run_pvground_scanrefer_empty_pool.py')
launch = launch.replace("'plan.md':(repo/'docs/PVG_VSA_BATCH_ORDER_2026-09-08.md').read_bytes()", "'plan.md':(repo/'docs/PVG_EMPTY_POOL_CONTROL_2026-09-08.md').read_bytes(),\n       'pvground_empty_pool_mask.py':module")
launch = launch.replace("(oldaudit/'queue.py')", "(oldaudit/'audit_queue.py')")
launch = launch.replace('mcln_pvg_scan_vsaorder_v1', 'mcln_pvg_scan_emptypool_v1').replace('mcln_pvg_vsaorder_audit_v1','mcln_pvg_emptypool_audit_v1')
(repo/'scripts/launch_pvground_empty_pool_training.py').write_text(launch, encoding='utf-8', newline='\n')
for name in ['run_pvground_scanrefer_empty_pool.py', 'launch_pvground_empty_pool_training.py']:
    compile((repo/'scripts'/name).read_bytes(), name, 'exec')
print('Prepared training and launch sources; no remote launch performed.')

formal = (repo/'scripts/evaluate_pvground_scanrefer_vsa_order.py').read_text()
formal = replace_once(formal, "    assert set(model.state_dict())-set(parent) == {'text_encoder.embeddings.position_ids'}", "    from pvground_empty_pool_mask import install_empty_pool_mask\n    assert train_spec['empty_pool_mask'] is True\n    assert sha(root/'pvground_empty_pool_mask.py')==train_spec['empty_pool_module_sha256']\n    install_empty_pool_mask(model,False)\n    assert set(model.state_dict())-set(parent) == {'text_encoder.embeddings.position_ids'}")
formal = replace_once(formal, "    assert delta['spec_sha256'] == sha(training/'spec.json')", "    assert delta['spec_sha256'] == sha(training/'spec.json')\n    assert delta['empty_pool_mask'] is True\n    assert delta['empty_pool_module_sha256']==train_spec['empty_pool_module_sha256']\n    assert receipt['empty_pool_mask'] is True\n    assert receipt['empty_pool_module_sha256']==train_spec['empty_pool_module_sha256']")
formal = replace_once(formal, "    def evaluate(arm,state):", "    def evaluate(arm,state):\n        assert arm in ['published_parent','fit_terminal']\n        vsa=model.backbone_net.vsa\n        for reader in [vsa.SA_rawpoints]+list(vsa.SA_layers):\n            reader.mask_empty=(arm=='fit_terminal')")
formal = replace_once(formal, "        'arms':['published_parent','fit_terminal'],'v99_reference':", "        'empty_pool_mask_by_arm':{'published_parent':False,'fit_terminal':True},\n        'empty_pool_module_sha256':train_spec['empty_pool_module_sha256'],\n        'arms':['published_parent','fit_terminal'],'v99_reference':")
(repo/'scripts/evaluate_pvground_scanrefer_empty_pool.py').write_text(formal, encoding='utf-8', newline='\n')

prepare = (repo/'scripts/prepare_pvground_vsa_order_formal.py').read_text()
prepare = prepare.replace('20260908_vsaorder_v1','20260908_emptypool_v1').replace('20260908_detalign_v1','20260908_vsaorder_v1')
prepare = prepare.replace('scripts/evaluate_pvground_scanrefer_vsa_order.py','scripts/evaluate_pvground_scanrefer_empty_pool.py')
prepare = replace_once(prepare, "contract=json.loads", "files['pvground_empty_pool_mask.py']=(repo/'models/pvground_empty_pool_mask.py').read_bytes()\ncontract=json.loads")
prepare = replace_once(prepare, "for name in ['queue.py','controller.py','cpu_probe.py']:", "for name in ['formal_queue.py','controller.py','cpu_probe.py']:")
prepare = prepare.replace('mcln_pvg_vsaorder_formal_v1','mcln_pvg_emptypool_formal_v1')
(repo/'scripts/prepare_pvground_empty_pool_formal.py').write_text(prepare, encoding='utf-8', newline='\n')
for name in ['evaluate_pvground_scanrefer_empty_pool.py','prepare_pvground_empty_pool_formal.py']:
    compile((repo/'scripts'/name).read_bytes(),name,'exec')
print('Prepared formal sources with explicit architecture restoration; no launch performed.')
