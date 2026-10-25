"""VNLP box-only patch generation for the upstream two-stage plate MTCNN.

No recognition labels or held-out images are used by this module.
"""
import cv2
import numpy as np

from .imaging import intersection_over_union


def patch_label(crop_box, gt_box):
    iou = intersection_over_union(crop_box, gt_box)
    if iou >= .65:
        return 1
    if iou >= .4:
        return -1
    if iou < .3:
        return 0
    return None


def box_offsets(crop_box, gt_box):
    """Targets compatible with upstream inclusive calibrate_box after slicing."""
    x, y, w, h = crop_box
    gx, gy, gw, gh = gt_box
    return np.float32([(gx-x)/w, (gy-y)/h, (gx+gw-x-w)/w, (gy+gh-y-h)/h])


def sample_patch_boxes(shape, gt_box, rng, quotas=(12, 8, 32)):
    """Fixed positive/partial/background quotas per training scene.

    Positives and partials jitter only the annotated box. Negatives use image
    context, including nearby regions. IoU gaps are ignored, as in upstream.
    """
    ih, iw = shape[:2]
    gx, gy, gw, gh = gt_box
    needed = {1: quotas[0], -1: quotas[1], 0: quotas[2]}
    selected = []
    for trial in range(5000):
        if not any(needed.values()):
            break
        if trial == 0:
            box = tuple(map(int, gt_box))
        elif needed[1] and trial % 3 == 0:
            w = max(8, round(gw*rng.uniform(.85, 1.15)))
            h = max(6, round(gh*rng.uniform(.85, 1.15)))
            x = round(gx+gw/2+rng.uniform(-.08, .08)*gw-w/2)
            y = round(gy+gh/2+rng.uniform(-.08, .08)*gh-h/2)
            box = (x, y, w, h)
        elif needed[-1] and trial % 3 != 1:
            w = max(8, round(gw*rng.uniform(.6, 1.6)))
            h = max(6, round(gh*rng.uniform(.6, 1.6)))
            x = round(gx+gw/2+rng.uniform(-.35, .35)*gw-w/2)
            y = round(gy+gh/2+rng.uniform(-.35, .35)*gh-h/2)
            box = (x, y, w, h)
        else:
            w = min(iw, max(12, round(gw*rng.uniform(.4, 1.8))))
            h = min(ih, max(6, round(gh*rng.uniform(.4, 1.8))))
            box = (int(rng.integers(0, iw-w+1)), int(rng.integers(0, ih-h+1)), w, h)
        x, y, w, h = box
        if x < 0 or y < 0 or x+w > iw or y+h > ih:
            continue
        label = patch_label(box, gt_box)
        if label is not None and needed[label]:
            selected.append((box, label))
            needed[label] -= 1
    if any(needed.values()):
        raise ValueError(f'Could not fill MTCNN patch quotas: {needed}, image={shape}, gt={gt_box}')
    return selected


def encode_patch(image, box, gt_box, label, size=(94, 24)):
    x, y, w, h = box
    crop = image[y:y+h, x:x+w]
    if not crop.size:
        raise ValueError('Empty training crop')
    # Cache RGB uint8; training applies the original (x-127.5)/128 transform.
    patch = cv2.resize(crop, size, interpolation=cv2.INTER_LINEAR)[:, :, ::-1].copy()
    return patch, box_offsets(box, gt_box) if label != 0 else np.zeros(4, np.float32)


def mtcnn_loss(offsets, logits, labels, targets):
    """Same masks and 0.02 CE + 0.6 MSE weights as the upstream trainers."""
    import torch.nn.functional as F
    offsets = offsets.reshape(-1, 4)
    logits = logits.reshape(-1, 2)
    classification = labels >= 0
    regression = labels != 0
    zero = logits.sum()*0
    ce = F.cross_entropy(logits[classification], labels[classification]) if classification.any() else zero
    mse = F.mse_loss(offsets[regression], targets[regression]) if regression.any() else zero
    return .02*ce + .6*mse, ce, mse
