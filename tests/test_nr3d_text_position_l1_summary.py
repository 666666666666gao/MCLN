import hashlib
import json

import pytest

from scripts.summarize_nr3d_text_position_l1 import compare, verify_terminal_run


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


def write_terminal_fixture(directory, text_hits, protected_hits):
    """Synthetic complete row files; never a real experiment or GPU input."""
    directory.mkdir()

    def arm(i, hits):
        return {'rec_query': 0, 'rec_box_iou': .3 if i < hits else .2,
                'mask_iou': .8, 'legal_box_oracle_iou': .8}

    baseline, terminal = [], []
    for i in range(6172):
        identity = {'id': i, 'scan_id': 'scene_'+str(i % 98),
                    'input_point_sha256': hashlib.sha256(str(i).encode()).hexdigest()}
        baseline.append(dict(identity, protected=arm(i, protected_hits)))
        terminal.append(dict(identity, text=arm(i, text_hits), position=arm(i, 20)))
    contents = {'baseline_rows.json': baseline, 'terminal_rows.json': terminal}
    for name, rows in contents.items():
        (directory/name).write_text(json.dumps(rows))
    manifest = directory.parent/'input_manifest.json'
    manifest.write_text(json.dumps({'row_ids': {'holdout': list(range(6172))}}))
    receipt = {'status': 'complete', 'optimizer_steps_per_arm': 6687,
               'frozen_parameters_and_buffers_unchanged': True,
               'source_data_and_parent_checkpoint_unchanged': True,
               'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest()}
    for stage in ('baseline', 'terminal'):
        receipt[stage+'_rows_sha256'] = hashlib.sha256((directory/(stage+'_rows.json')).read_bytes()).hexdigest()
    (directory/'receipt.json').write_text(json.dumps(receipt))


@pytest.mark.parametrize('text_hits,protected_hits,expected', [(10, 10, True), (11, 10, False), (10, 11, False)])
def test_complete_receipt_requires_both_paired_controls(tmp_path, text_hits, protected_hits, expected):
    directory = tmp_path/'train'
    write_terminal_fixture(directory, text_hits, protected_hits)
    result = verify_terminal_run(directory)
    assert result['integrity_pass']
    assert result['fixed_screen_pass'] is expected
    assert result['metrics']['terminal_position']['rec_hits_025'] == 20
    assert result['formal_rows'] == 0 and not result['formal_promotion']


def test_complete_receipt_rejects_changed_terminal_rows(tmp_path):
    directory = tmp_path/'train'
    write_terminal_fixture(directory, 10, 10)
    path = directory/'terminal_rows.json'
    rows = json.loads(path.read_text())
    rows[0]['position']['rec_box_iou'] = .9
    path.write_text(json.dumps(rows))
    with pytest.raises(AssertionError):
        verify_terminal_run(directory)
