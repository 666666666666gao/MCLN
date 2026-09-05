from scripts.summarize_nr3d_text_position_l1 import compare


def records(rec, mask, arm):
    return [{'id':i, 'scan_id':'scene_'+str(i % 3),
             arm:{'rec_query':0,'rec_box_iou':r,'mask_iou':m}}
            for i,(r,m) in enumerate(zip(rec,mask))]


def test_ten_net_rec25_hits_with_no_other_regression_pass():
    old=records([.2]*10+[.8]*10, [.8]*20, 'text')
    new=records([.3]*10+[.8]*10, [.8]*20, 'position')
    result=compare(old,new,'text')
    assert result['fixed_screen_pass']
    assert result['thresholds']['rec025'] == {'fixes':10,'breaks':0,'net':10}


def test_strict_rec_break_fails_even_with_enough_rec25_fixes():
    old=records([.2]*10+[.8]*10, [.8]*20, 'text')
    new=records([.3]*11+[.8]*9, [.8]*20, 'position')
    result=compare(old,new,'text')
    assert result['thresholds']['rec025']['net'] == 10
    assert result['thresholds']['rec050']['net'] == -1
    assert not result['fixed_screen_pass']


def test_mask_quality_drop_without_threshold_changes_fails():
    old=records([.2]*10+[.8]*10, [.8]*20, 'protected')
    new=records([.3]*10+[.8]*10, [.79]*20, 'position')
    result=compare(old,new,'protected')
    assert result['thresholds']['mask025']['net'] == result['thresholds']['mask050']['net'] == 0
    assert not result['fixed_screen_pass']


def test_nine_net_fixes_are_below_the_fixed_screen():
    old=records([None]*10+[.8]*10, [.8]*20, 'protected')
    new=records([.3]*9+[None]+[.8]*10, [.8]*20, 'position')
    result=compare(old,new,'protected')
    assert result['thresholds']['rec025']['net'] == 9
    assert not result['fixed_screen_pass']
