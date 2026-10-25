import pytest

from plate_ml.evaluation import edit_distance
from plate_ml.sequence_scoring import character_alignment, joint_plate_metrics, text_metrics


@pytest.mark.parametrize('truth,predicted,expected', [
    ('ABC', 'ADC', (2, 1, 1)), ('ABC', 'ABCD', (3, 1, 0)),
    ('ABC', 'AC', (2, 0, 1)), ('ABC', '', (0, 0, 3)), ('AA', 'A', (1, 0, 1)),
])
def test_character_fp_fn_distinguish_substitutions_insertions_deletions(truth, predicted, expected):
    result = text_metrics([dict(truth=truth, text=predicted)])
    assert tuple(result['character_prf'][k] for k in ['tp', 'fp', 'fn']) == expected
    assert result['alignment']['edit_distance'] == edit_distance(truth, predicted)


def test_character_alignment_exhaustive_short_binary_strings():
    import itertools
    strings = ['']+[''.join(x) for n in range(1, 4) for x in itertools.product('AB', repeat=n)]
    for a in strings:
        for b in strings:
            result = character_alignment(a, b)
            assert result['edit_distance'] == edit_distance(a, b)
            assert result['correct']+result['substitutions']+result['deletions'] == len(a)
            assert result['correct']+result['substitutions']+result['insertions'] == len(b)


def test_joint_metric_rejects_right_text_at_wrong_location_and_counts_misses():
    base = dict(truth='ABC', gt_box=[0, 0, 10, 10])
    rows = [dict(base, text='ABC', box=[0, 0, 10, 10]),
            dict(base, text='ABC', box=[20, 20, 10, 10]),
            dict(base, text='ADC', box=[0, 0, 10, 10]),
            dict(base, text='', box=[0, 0, 10, 10])]
    result = joint_plate_metrics(rows)
    assert (result['tp'], result['fp'], result['fn']) == (1, 2, 3)
    assert result['precision'] == pytest.approx(1/3)
    assert result['recall'] == pytest.approx(1/4)
