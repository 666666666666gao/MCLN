from scripts.summarize_nr3d_mask_neighborhood_m5 import compare


def rows(values, arm):
    return [{'id':i, 'scan_id':'scene' + str(i // 4), arm:{'mask_iou':value}}
            for i,value in enumerate(values)]


def test_mean_gain_does_not_override_strict_hit_loss():
    reference = rows([.51] + [.6] * 7, 'native')
    candidate = rows([.49] + [.62] * 7, 'nearest')
    result = compare(reference, candidate, 'native')
    assert result['delta_mask_mean_iou'] > .002
    assert result['thresholds']['050'] == {'fixes':0, 'breaks':1, 'net':-1}
    assert not result['fixed_screen_pass']


def test_positive_endpoint_must_pass_mean_and_both_hit_requirements():
    reference = rows([.49] + [.6] * 7, 'native')
    candidate = rows([.51] + [.62] * 7, 'nearest')
    result = compare(reference, candidate, 'native')
    assert result['thresholds']['050'] == {'fixes':1, 'breaks':0, 'net':1}
    assert result['thresholds']['025']['net'] == 0
    assert result['fixed_screen_pass']
    assert result['scene_count'] == 2
    assert all(value > 0 for value in result['paired_scene_bootstrap_95_ci']['delta_mask_mean_iou'])


def test_small_mean_gain_fails_even_when_hit_counts_hold():
    result = compare(rows([.6] * 8, 'native'), rows([.601] * 8, 'nearest'), 'native')
    assert all(value['net'] == 0 for value in result['thresholds'].values())
    assert not result['fixed_screen_pass']
