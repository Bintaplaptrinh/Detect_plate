import pytest

from plate_ml.comparison import (average_precision, paired_difference, select_test_subset, summarize,
                                 normalize_ocr, join_ocr_lines)


def row(path, truth='ABC', text='ABC', detections=None):
    return dict(path=path, truth=truth, text=text, raw=text, layout='one_row', gt_box=[0, 0, 10, 10],
                box=[0, 0, 10, 10] if detections else None, detections=detections or [], latency_ms=[10., 20., 30.])


def test_ap_penalizes_early_false_positive_and_duplicate():
    rows = [row('a', detections=[dict(box=[20, 20, 10, 10], score=.9),
                                  dict(box=[0, 0, 10, 10], score=.8),
                                  dict(box=[0, 0, 10, 10], score=.7)]),
            row('b', detections=[dict(box=[0, 0, 10, 10], score=.6)])]
    # Precision envelope is .5 at every recall point, with two matched GTs.
    assert average_precision(rows, .5) == pytest.approx(.5)
    assert average_precision([row('a')], .5) == 0


def test_balanced_selection_excludes_train_is_reproducible():
    manifest = [dict(path=f'{layout}/{i}', group=f'{layout}-{i}', layout=layout,
                     partition='test' if i < 5 else 'train') for layout in ['a', 'b', 'c'] for i in range(8)]
    chosen = select_test_subset(manifest, 10, 42)
    assert chosen == select_test_subset(manifest, 10, 42)
    assert all(r['partition'] == 'test' for r in chosen)
    assert [sum(r['layout'] == k for r in chosen) for k in ['a', 'b', 'c']] == [4, 3, 3]


def test_no_detection_is_not_excluded_from_sequence_metrics():
    result = summarize([row('a', text='')], repeats=3)
    assert result['decoded']['exact_match'] == 0
    assert result['decoded']['character_error_rate'] == 1
    assert result['detector_all']['fn'] == 1
    assert result['latency']['mean_ms'] == 20


def test_paired_bootstrap_sign_and_order():
    a, b = [row('a'), row('b')], [row('a', text=''), row('b', text='')]
    diff = paired_difference(a, b, draws=100)
    assert diff['exact_match_delta_ci95'] == [-1., -1.]
    assert diff['cer_delta'] == 1
    assert diff['current_only_correct'] == 2
    with pytest.raises(ValueError):
        paired_difference(a, b[::-1])


def test_ocr_quadrilaterals_use_scalar_y_and_preserve_reading_order():
    def snippet(text, y):
        return ([(0, y), (10, y), (10, y+8), (0, y+8)], text, .9)
    assert join_ocr_lines([snippet('29A', 0), snippet('123', 0), snippet('45', 20)]) == '29A 123\n45'
    assert normalize_ocr('29a-123.45\n') == '29A12345'
    assert normalize_ocr('京A123') == '京A123'
