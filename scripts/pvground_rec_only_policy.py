"""Apply the user's 2026-09-17 REC-only promotion policy to future formal runs."""

def replace_once(text, before, after):
    assert text.count(before) == 1, before
    return text.replace(before, after)


def rec_only_sources(evaluator, auditor, plan):
    evaluator = replace_once(evaluator,
        "              'rec50_historical_v99': metrics['rec_hits50'] >= 4797,\n"
        "              'mask25_paper': metrics['mask_hits25'] * 100. / 9508 >= 58.70,\n"
        "              'mask50_paper': metrics['mask_hits50'] * 100. / 9508 >= 50.70,\n"
        "              'mask_miou_paper': metrics['mask_miou'] >= 44.72}",
        "              'rec50_historical_v99': metrics['rec_hits50'] >= 4797}")
    evaluator = replace_once(evaluator,
        "'requires_independent_formal_audit': True, 'nr3d_sr3d_mask_gate': False}",
        "'requires_independent_formal_audit': True, 'scanrefer_mask_gate': False, 'nr3d_sr3d_mask_gate': False}")
    evaluator = replace_once(evaluator,
        "'scan_mask_floor_percent':[58.70,50.70,44.72]", "'scanrefer_mask_gate':False")
    auditor = replace_once(auditor,
        "    assert protocol['scan_mask_floor_percent']==[58.70,50.70,44.72]",
        "    assert protocol['scanrefer_mask_gate'] is False")
    auditor = replace_once(auditor,
        "        'rec50_historical_v99':candidate['rec_hits50']>=4797,\n"
        "        'mask25_paper':candidate['mask_hits25']*100./9508>=58.70,\n"
        "        'mask50_paper':candidate['mask_hits50']*100./9508>=50.70,\n"
        "        'mask_miou_paper':candidate['mask_miou']>=44.72}",
        "        'rec50_historical_v99':candidate['rec_hits50']>=4797}")
    auditor = replace_once(auditor,
        "    assert not receipt['promotion']['nr3d_sr3d_mask_gate']",
        "    assert receipt['promotion']['scanrefer_mask_gate'] is False\n"
        "    assert not receipt['promotion']['nr3d_sr3d_mask_gate']")
    auditor = replace_once(auditor,
        "'advance_to_nr3d_sr3d_rec':all(checks.values()),'stages':stages,",
        "'advance_to_nr3d_sr3d_rec':all(checks.values()),'scanrefer_mask_gate':False,'stages':stages,")
    plan = replace_once(plan,
        '5572/4797, Mask58.70/50.70/44.72 remain.',
        '5572/4797 remain. Per user instruction2026-09-17, Mask is recorded but is not a ScanRefer promotion gate.')
    return evaluator, auditor, plan
