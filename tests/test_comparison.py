import pytest

from plate_ml.comparison import average_precision, paired_difference, summarize


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
