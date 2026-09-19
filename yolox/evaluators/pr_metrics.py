#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""Single-class precision/recall and PR-curve utilities.

`COCOEvaluator` (coco_evaluator.py) only surfaces aggregate COCO AP (area
under pycocotools' 101-point *interpolated* PR curve) and AR (max recall at
a fixed maxDets budget, see `per_class_AP_table`/`per_class_AR_table` there
for how those get pulled out of `cocoEval.eval["precision"/"recall"]`).
Those are good training-progress summaries but don't answer "at the
confidence threshold I'll actually run inference with, what precision and
recall do I get", and pycocotools doesn't expose a raw (non-interpolated)
PR curve at all -- this module computes both directly from raw predictions
and ground truth, via the same greedy IoU-matching rule COCOeval uses
internally (highest-confidence predictions matched first; each ground-truth
box claimable by at most one prediction).
"""

from __future__ import annotations

import numpy as np
import torch

from yolox.utils.boxes import bboxes_iou

__all__ = [
    "match_predictions",
    "compute_pr_curve",
    "precision_recall_at_threshold",
    "f1_at_threshold",
    "best_f1_over_thresholds",
    "log_pr_curve_to_tensorboard",
]


def match_predictions(pred_boxes, pred_scores, gt_boxes, iou_thresh=0.5):
    """Greedy single-class IoU matching for one image's predictions against
    its ground truth.

    Args:
        pred_boxes: `np.ndarray` [N, 4], xyxy, this image's predicted boxes.
        pred_scores: `np.ndarray` [N], confidence scores, same order as
            `pred_boxes`.
        gt_boxes: `np.ndarray` [M, 4], xyxy, this image's ground-truth boxes
            (already filtered to the class being evaluated).
        iou_thresh: `float`, minimum IoU to count as a match.

    Returns:
        `np.ndarray` [N] bool, in the same order as `pred_boxes`/
        `pred_scores` (NOT confidence-sorted): True where that prediction is
        a true positive.
    """
    n = len(pred_boxes)
    is_tp = np.zeros(n, dtype=bool)
    if n == 0 or len(gt_boxes) == 0:
        return is_tp  # no predictions, or nothing to match against (all FP)

    order = np.argsort(-pred_scores)
    ious = bboxes_iou(
        torch.as_tensor(pred_boxes[order], dtype=torch.float32),
        torch.as_tensor(gt_boxes, dtype=torch.float32),
    ).numpy()

    claimed = np.zeros(len(gt_boxes), dtype=bool)
    for row, orig_idx in enumerate(order):
        gt_idx = int(np.argmax(ious[row]))
        if ious[row, gt_idx] >= iou_thresh and not claimed[gt_idx]:
            claimed[gt_idx] = True
            is_tp[orig_idx] = True

    return is_tp


def compute_pr_curve(predictions, coco_gt, iou_thresh=0.5, class_id=None):
    """Computes a raw (non-interpolated) precision/recall curve for a
    single class across an entire validation set.

    Pools every prediction from every image (matched against that image's
    own ground truth via `match_predictions`), sorts all predictions
    globally by confidence descending, and takes running precision/recall
    as progressively more (lower-confidence) predictions are kept -- i.e.
    each point on the curve is "if I only trust predictions with confidence
    >= this prediction's score, what are my precision and recall".

    Args:
        predictions: `dict`, `{image_id: {"bboxes": [[x1,y1,x2,y2], ...],
            "scores": [...], "categories": [...]}}` -- exactly the
            structure `COCOEvaluator.evaluate(..., return_outputs=True)`
            returns as its second value. Boxes are expected in original
            (pre-resize) image pixel space, matching that structure.
        coco_gt: `pycocotools.coco.COCO`, ground truth, e.g.
            `evaluator.dataloader.dataset.coco`.
        iou_thresh: `float`, IoU match threshold (0.5 = the standard
            "AP50"/PASCAL-VOC convention).
        class_id: `int`, COCO category_id to evaluate. Defaults to the
            single category present (raises if there isn't exactly one --
            this module assumes single-class; pass explicitly otherwise).

    Returns:
        A `dict`:
            "scores": `np.ndarray`, descending confidence -- the effective
                threshold at each curve point.
            "precision", "recall": `np.ndarray`, same length as "scores".
            "num_tp", "num_fp": `np.ndarray`, cumulative counts, same length.
            "num_gt": `int`, total ground-truth objects for this class
                (the fixed recall denominator).
    """
    cat_ids = coco_gt.getCatIds()
    if class_id is None:
        assert len(cat_ids) == 1, (
            "compute_pr_curve() defaults to the single category present; "
            "pass class_id explicitly for a multi-class dataset."
        )
        class_id = cat_ids[0]

    all_scores = []
    all_tp = []
    num_gt = 0

    for img_id in coco_gt.getImgIds():
        ann_ids = coco_gt.getAnnIds(imgIds=[img_id], catIds=[class_id], iscrowd=False)
        gt_anns = coco_gt.loadAnns(ann_ids)
        gt_boxes = np.array(
            [[a["bbox"][0], a["bbox"][1],
              a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
             for a in gt_anns],
            dtype=np.float32,
        ).reshape(-1, 4)
        num_gt += len(gt_boxes)

        pred = predictions.get(int(img_id))
        if pred is None or len(pred["bboxes"]) == 0:
            continue

        cats = np.array(pred["categories"])
        keep = cats == class_id
        if not np.any(keep):
            continue

        pred_boxes = np.array(pred["bboxes"], dtype=np.float32).reshape(-1, 4)[keep]
        pred_scores = np.array(pred["scores"], dtype=np.float32)[keep]

        is_tp = match_predictions(pred_boxes, pred_scores, gt_boxes, iou_thresh)
        all_scores.append(pred_scores)
        all_tp.append(is_tp)

    if not all_scores or num_gt == 0:
        empty = np.array([])
        return {
            "scores": empty, "precision": empty, "recall": empty,
            "num_tp": empty, "num_fp": empty, "num_gt": num_gt,
        }

    scores = np.concatenate(all_scores)
    tp = np.concatenate(all_tp)

    order = np.argsort(-scores)
    scores = scores[order]
    tp = tp[order]

    num_tp = np.cumsum(tp)
    num_fp = np.cumsum(~tp)

    precision = num_tp / np.maximum(num_tp + num_fp, 1)
    recall = num_tp / num_gt

    return {
        "scores": scores, "precision": precision, "recall": recall,
        "num_tp": num_tp, "num_fp": num_fp, "num_gt": num_gt,
    }


def precision_recall_at_threshold(pr_curve, conf_thresh):
    """Single-number precision/recall at a fixed confidence threshold, from
    a curve returned by `compute_pr_curve` -- e.g. the same `test_conf` an
    Exp actually filters detections with at inference time, so this answers
    "what precision/recall do I get with the model as deployed", which
    COCO AP/AR (aggregated across all thresholds/IoUs) don't directly.

    Returns:
        `(precision, recall)`. `precision` is `float('nan')` (not 0) if no
        prediction meets the threshold, since 0-out-of-0 is undefined
        rather than "bad"; `recall` is always a real number (0 if nothing
        was predicted, since the ground-truth count is unaffected).
    """
    scores = pr_curve["scores"]
    num_gt = pr_curve["num_gt"]
    if len(scores) == 0 or num_gt == 0:
        return float("nan"), 0.0

    keep = np.nonzero(scores >= conf_thresh)[0]
    if len(keep) == 0:
        return float("nan"), 0.0

    idx = keep[-1]  # last (lowest-score) prediction still >= threshold
    tp = pr_curve["num_tp"][idx]
    fp = pr_curve["num_fp"][idx]
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / num_gt
    return float(precision), float(recall)


def f1_at_threshold(pr_curve, conf_thresh):
    """Single-class F1 score at a fixed confidence threshold, from a curve
    returned by `compute_pr_curve`.

    Built on `precision_recall_at_threshold` -- see that function for why
    `conf_thresh` should normally be `exp.test_conf`, the same threshold
    used to filter detections at inference. Returns `0.0` (not NaN) when
    nothing scores >= `conf_thresh`: `precision_recall_at_threshold` always
    pairs a NaN precision with recall == 0.0 in that case (no prediction
    means no true positive, so `tp + fp` for the "how many predictions
    kept" denominator is 0 too), and 0 recall means the model detects
    nothing at this threshold regardless of precision being undefined --
    worse than any real detector, not "unknown". This makes F1 safe to use
    directly as a best-checkpoint comparison key without a separate NaN
    check at every call site.
    """
    precision, recall = precision_recall_at_threshold(pr_curve, conf_thresh)
    if recall == 0.0 or np.isnan(precision):
        return 0.0
    return 2 * precision * recall / (precision + recall)


def best_f1_over_thresholds(pr_curve):
    """Best F1 achievable at any confidence threshold, and the threshold
    that achieves it, from a curve returned by `compute_pr_curve`.

    Unlike `f1_at_threshold` (one fixed threshold, e.g. `exp.test_conf` --
    "what do I get as actually deployed"), this scans every threshold the
    curve has a point for: `compute_pr_curve` already returns cumulative
    precision/recall at each successive prediction's score (i.e. "keep the
    top-k highest-confidence predictions", for every k), so the best F1
    over all thresholds is just the max of the per-point F1 values, and its
    threshold is that point's score. This is an oracle metric -- the best a
    post-hoc-tuned threshold could do on this validation set -- useful for
    comparing model/checkpoint quality independent of whatever
    `exp.test_conf` happens to be set to, at the cost of being optimistic
    versus real deployment (where the threshold is fixed in advance, not
    re-picked per checkpoint).

    Unlike `precision_recall_at_threshold`, precision is never NaN at any
    in-range curve point: index i always corresponds to keeping exactly
    i + 1 predictions (the top i + 1 by score), so `num_tp + num_fp` at
    that point is always >= 1.

    Returns:
        `(best_f1, best_threshold)`. `best_threshold` is `float('nan')`
        when the curve is empty (no predictions or no ground truth), and
        `best_f1` is `0.0` in that case.
    """
    scores = pr_curve["scores"]
    if len(scores) == 0:
        return 0.0, float("nan")

    precision = pr_curve["precision"]
    recall = pr_curve["recall"]
    denom = precision + recall
    f1 = np.where(denom > 0, 2 * precision * recall / np.maximum(denom, 1e-12), 0.0)

    best_idx = int(np.argmax(f1))
    return float(f1[best_idx]), float(scores[best_idx])


def log_pr_curve_to_tensorboard(tblogger, pr_curve, global_step, tag="val/pr_curve", num_thresholds=127):
    """Logs a precision/recall curve to TensorBoard.

    Uses `add_pr_curve_raw` (precomputed precision/recall/TP/FP/FN arrays),
    not the higher-level `add_pr_curve` (which takes per-sample binary
    labels + scores and derives recall as TP/(TP+FN) *among scored
    samples*) -- that's wrong for detection, where a missed ground-truth
    object never produces a predicted score to threshold in the first
    place, so it can't appear in a "labels/predictions" sample list at all.
    `add_pr_curve_raw` lets us supply the correct, fixed recall denominator
    (total ground-truth count, from `compute_pr_curve`) directly instead.

    Args:
        tblogger: `torch.utils.tensorboard.SummaryWriter`.
        pr_curve: `dict`, as returned by `compute_pr_curve`.
        global_step: `int`, typically the epoch number.
        tag: `str`, TensorBoard tag.
        num_thresholds: `int`, number of points on the logged curve (capped
            at 127 by TensorBoard's PR-curve plugin). The raw curve from
            `compute_pr_curve` has one point per prediction, which is
            downsampled to this many evenly-spaced confidence thresholds.
    """
    scores = pr_curve["scores"]
    num_gt = pr_curve["num_gt"]
    thresholds = np.linspace(0.0, 1.0, num_thresholds)

    if len(scores) == 0:
        precision = np.ones(num_thresholds, dtype=np.float32)
        recall = np.zeros(num_thresholds, dtype=np.float32)
        tp = np.zeros(num_thresholds, dtype=np.float32)
        fp = np.zeros(num_thresholds, dtype=np.float32)
        fn = np.full(num_thresholds, num_gt, dtype=np.float32)
    else:
        # For each threshold, the index of the last (lowest-score)
        # prediction still >= that threshold (-1 if none qualify).
        # scores is sorted descending, so negate both sides to reuse
        # searchsorted's ascending-array convention:
        # count of scores >= t  ==  count of (-scores) <= -t.
        idx = np.searchsorted(-scores, -thresholds, side="right") - 1
        valid = idx >= 0
        clipped = np.clip(idx, 0, None)

        precision = np.where(valid, pr_curve["precision"][clipped], 1.0).astype(np.float32)
        recall = np.where(valid, pr_curve["recall"][clipped], 0.0).astype(np.float32)
        tp = np.where(valid, pr_curve["num_tp"][clipped], 0).astype(np.float32)
        fp = np.where(valid, pr_curve["num_fp"][clipped], 0).astype(np.float32)
        fn = (num_gt - tp).astype(np.float32)

    # True negatives are not a meaningful concept for detection (there is
    # no fixed universe of "negative" boxes to count), but the plugin
    # requires the field -- zero-fill it.
    tn = np.zeros(num_thresholds, dtype=np.float32)

    tblogger.add_pr_curve_raw(
        tag,
        true_positive_counts=tp,
        false_positive_counts=fp,
        true_negative_counts=tn,
        false_negative_counts=fn,
        precision=precision,
        recall=recall,
        global_step=global_step,
        num_thresholds=num_thresholds,
    )
