import numpy as np
import pytest

from plate_ml.mtcnn_training import box_offsets, patch_label, sample_patch_boxes


def test_offsets_reconstruct_half_open_ground_truth_with_upstream_inclusive_math():
    crop, gt = (10, 20, 40, 12), (12, 18, 36, 18)
    offsets = box_offsets(crop, gt)
    # Upstream endpoints are inclusive: original width is x2-x1+1.
    start = np.float32([10, 20, 49, 31])
    reconstructed = start + offsets*np.float32([40, 12, 40, 12])
    np.testing.assert_allclose(reconstructed, [12, 18, 47, 35])


def test_patch_thresholds_ignore_ambiguous_iou_gap():
    gt = (0, 0, 100, 100)
    assert patch_label(gt, gt) == 1
    assert patch_label((0, 0, 50, 100), gt) == -1
    assert patch_label((0, 0, 35, 100), gt) is None
    assert patch_label((0, 0, 20, 100), gt) == 0


@pytest.mark.parametrize('gt', [(100, 100, 200, 40), (1, 1, 80, 60)])
def test_sampling_has_exact_quotas_and_no_mislabeled_boxes(gt):
    sampled = sample_patch_boxes((400, 600, 3), gt, np.random.default_rng(42))
    assert [sum(label == k for _, label in sampled) for k in [1, -1, 0]] == [12, 8, 32]
    assert all(patch_label(box, gt) == label for box, label in sampled)


def test_partial_crops_do_not_enter_classification_loss():
    torch = pytest.importorskip('torch')
    from plate_ml.mtcnn_training import mtcnn_loss
    offsets = torch.zeros(3, 4, requires_grad=True)
    logits = torch.zeros(3, 2, requires_grad=True)
    loss, _, _ = mtcnn_loss(offsets, logits, torch.tensor([1, -1, 0]), torch.ones(3, 4))
    loss.backward()
    assert torch.all(logits.grad[1] == 0)
    assert torch.all(offsets.grad[2] == 0)
    assert torch.any(offsets.grad[1] != 0)
