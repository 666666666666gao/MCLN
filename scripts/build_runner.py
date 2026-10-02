"""Extend the frozen P3 runner only at explicit support-arm and preflight points."""
from pathlib import Path
import py_compile

root = Path(r'C:\Users\gb\.codex\tmp\pvground_fused_support_20261002')
parent = Path(r'C:\Users\gb\.codex_pvground_cs_20261002\scripts\run_pvground_p3.py')
text = parent.read_text()

def replace(before, after):
    global text
    assert text.count(before) == 1, before
    text = text.replace(before, after)

replace('"""G plus final-layer candidate-aligned geometry: same start and budget."""',
        '"""Paired same-tail raw versus native predicted fused-Mask support."""')
replace("    assert sha(output/'pvground_candidate_box_refiner.py') == spec['p3_module_sha256']\n",
    "    assert sha(output/'pvground_candidate_box_refiner.py') == spec['p3_module_sha256']\n"
    "    assert sha(output/'pvground_tail_support_box_refiner.py') == spec['tail_module_sha256']\n"
    "    assert spec['support_arm'] in ('tail_raw','tail_fused')\n"
    "    assert spec['fused_support'] == (spec['support_arm'] == 'tail_fused')\n"
    "    assert port['call_position'] == 'after native Mask generation'\n")
replace('    from pvground_candidate_box_refiner import install_candidate_box_refinement\n'
        '    install_candidate_box_refinement(model)\n',
        '    from pvground_tail_support_box_refiner import install_tail_support_refinement\n'
        "    install_tail_support_refinement(model, spec['fused_support'])\n")
replace("                        p3=True, p2=False, time_cst=now())\n",
        "                        p3=True, p2=False, support_arm=spec['support_arm'],\n"
        "                        fused_support=spec['fused_support'], time_cst=now())\n")
replace("        assert terminal['p3'] and not terminal['p2'] and terminal['step'] == 3723\n",
        "        assert terminal['p3'] and not terminal['p2'] and terminal['step'] == 3723\n"
        "        assert terminal['support_arm'] == spec['support_arm']\n"
        "        assert terminal['tail_module_sha256'] == spec['tail_module_sha256']\n")
replace("            assignment_counts['direct_semantic_to_p3_gradients_zero'] = True\n",
        "            assignment_counts['direct_semantic_to_p3_gradients_zero'] = True\n"
        "            mask_outputs = tuple(predictions['last_pred_masks'] + predictions['sp_last_pred_masks'] + predictions['adaptive_weights'])\n"
        "            mask_gradients = torch.autograd.grad(\n"
        "                predictions['loss_bbox']+predictions['loss_giou'],\n"
        "                mask_outputs,retain_graph=True,allow_unused=True)\n"
        "            assignment_counts['geometry_to_mask_output_gradient'] = sum(\n"
        "                float(g.norm()) for g in mask_gradients if g is not None)\n"
        "            if not spec['fused_support']:\n"
        "                assert assignment_counts['geometry_to_mask_output_gradient'] == 0\n")
replace("        support_statistics['negative_coarse_size_elements'] = int((with_p3['p3_coarse_size'] < 0).sum())\n",
        "        support_statistics['negative_coarse_size_elements'] = int((with_p3['p3_coarse_size'] < 0).sum())\n"
        "        raw_points = inputs['points'][:,1:].view(inputs['batch_size'],50000,6)\n"
        "        coarse_center = with_p3['p3_coarse_center'].detach()\n"
        "        layout_size = with_p3['p3_coarse_size'].detach().clamp(min=1e-6)\n"
        "        locations = coarse_center[:,:,None] + .5*layout_size[:,:,None]*read.locations\n"
        "        members,_ = read.nearest_members(raw_points[...,:3],locations.reshape(inputs['batch_size'],256*7,3))\n"
        "        mask_support = read.member_support(members,with_p3,inputs['batch_size'],256)\n"
        "        members = members.reshape(inputs['batch_size'],256,7,16)\n"
        "        for bid in range(inputs['batch_size']):\n"
        "            for qi in (0,42,255):\n"
        "                sid = inputs['superpoint'][bid][members[bid,qi]]\n"
        "                tl = with_p3['last_pred_masks'][bid][0,qi,sid]\n"
        "                ql = with_p3['sp_last_pred_masks'][bid][qi,sid]\n"
        "                alpha = with_p3['adaptive_weights'][bid]\n"
        "                expected = torch.stack([tl.sigmoid(),ql.sigmoid(),\n"
        "                    (tl.sigmoid()-ql.sigmoid()).abs(),(alpha*tl+(1-alpha)*ql).sigmoid()],dim=-1)\n"
        "                assert torch.equal(mask_support[bid,qi],expected)\n"
        "        del mask_support,members,locations,raw_points\n")
replace("                assert gradients['aggregate.0.weight'] > 0\n",
        "                assert gradients['aggregate.0.weight'] > 0\n"
        "                if spec['fused_support']:\n"
        "                    assert record['geometry_to_mask_output_gradient'] > 0\n")
replace("            serialization_bytes=stream.getbuffer().nbytes,optimizer_restore=True,formal_rows=0)\n",
        "            serialization_bytes=stream.getbuffer().nbytes,optimizer_restore=True,formal_rows=0,\n"
        "            support_arm=spec['support_arm'],fused_support=spec['fused_support'],\n"
        "            member_mask_mapping_exact=True,tail_after_native_masks=True)\n")
replace("        data.update(p3=True,p3_module_sha256=spec['p3_module_sha256'])\n",
        "        data.update(p3=True,p3_module_sha256=spec['p3_module_sha256'],\n"
        "                    support_arm=spec['support_arm'],fused_support=spec['fused_support'],\n"
        "                    tail_module_sha256=spec['tail_module_sha256'])\n")
replace("    write_json(output/'receipt.json',receipt)\n",
        "    receipt.update(support_arm=spec['support_arm'],fused_support=spec['fused_support'],\n"
        "                   tail_module_sha256=spec['tail_module_sha256'],primary_threshold=.5,\n"
        "                   paired_control_required=True,tail_after_native_masks=True)\n"
        "    write_json(output/'receipt.json',receipt)\n")
replace("    reset_rng(spec['seed'])\n    torch.backends.cudnn.benchmark=False\n",
    "    from pvground_tail_preflight import observed_forward, native_mask_loss_routes\n"
    "    assert sha(output/'pvground_tail_preflight.py') == spec['tail_preflight_module_sha256']\n"
    "    preflight_call_witnesses = []\n"
    "    reset_rng(spec['seed'])\n    torch.backends.cudnn.benchmark=False\n")
replace("        predictions=model(inputs)\n        matching=[]\n",
    "        if args.mode == 'preflight':\n"
    "            predictions,call_witness = observed_forward(model,inputs)\n"
    "            preflight_call_witnesses.append(call_witness)\n"
    "        else:\n"
    "            predictions=model(inputs)\n"
    "        matching=[]\n")
replace("            assignment_counts['direct_semantic_to_p3_gradients_zero'] = True\n",
    "            assignment_counts['direct_semantic_to_p3_gradients_zero'] = True\n"
    "            assignment_counts['native_mask_loss_to_refiner_gradients'] = native_mask_loss_routes(predictions,parameters)\n")
replace("            with_p3 = model(inputs)\n",
    "            with_p3,call_witness = observed_forward(model,inputs)\n"
    "            preflight_call_witnesses.append(call_witness)\n")
replace("            without_p3 = model(inputs)\n",
    "            without_p3,call_witness = observed_forward(model,inputs)\n"
    "            preflight_call_witnesses.append(call_witness)\n")
replace("            member_mask_mapping_exact=True,tail_after_native_masks=True)\n",
    "            member_mask_mapping_exact=True,tail_after_native_masks=True,\n"
    "            native_call_order_verified=True,native_call_witnesses=preflight_call_witnesses,\n"
    "            direct_native_mask_loss_to_refiner_gradients_zero=True)\n")
replace("        write_json(output/'preflight.json',receipt)\n",
    "        assert len(preflight_call_witnesses) == 4\n"
    "        write_json(output/'preflight.json',receipt)\n")
target = root/'run_pvground_tail_support.py'
target.write_text(text)
py_compile.compile(str(target),doraise=True)
print(str(target))
