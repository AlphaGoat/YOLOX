#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import cv2
import numpy as np

__all__ = ["vis", "vis_tp_fp", "stretch_for_display"]


def stretch_for_display(img, lo_pct=1.0, hi_pct=99.5):
    """Percentile-clip + linear-stretch a float image to uint8 [0, 255] for
    human-viewable logging (e.g. TensorBoard/wandb image panels).

    SatSim-derived training images are decoded from 16-bit FITS and kept as
    normalized float32 in [0, 1] throughout the pipeline (see
    yolox/data/datasets/coco.py::_load_fits_image) to preserve full sensor
    precision for training. Real background+point-source frames only use a
    small slice of that [0, 1] range (a typical frame's max pixel is well
    under 1.0), so casting straight to uint8 for display looks near-black.
    This is a *display-only* transform -- never use it on data headed into
    the model or loss, only on a copy being rendered/logged for a human.

    Args:
        img: `np.ndarray`, float image, any value range, any number of
            channels (percentiles are computed across the whole array, so a
            3-channel image with identical replicated channels -- our FITS
            loader's convention -- stretches consistently across channels).
        lo_pct: `float`, lower percentile clipped to black.
        hi_pct: `float`, upper percentile clipped to white.

    Returns:
        `np.ndarray`, uint8, same shape as `img`.
    """
    img = np.asarray(img, dtype=np.float32)
    lo, hi = np.percentile(img, [lo_pct, hi_pct])
    if hi <= lo:
        # Degenerate (flat) image -- avoid a divide-by-zero.
        return np.zeros_like(img, dtype=np.uint8)
    stretched = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
    return (stretched * 255).astype(np.uint8)


def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):

    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img


def _draw_labeled_box(img, box, color, text):
    x0, y0, x1, y1 = (int(v) for v in box)
    font = cv2.FONT_HERSHEY_SIMPLEX
    txt_color = (0, 0, 0) if sum(color) > 380 else (255, 255, 255)

    txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
    cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

    txt_bk_color = tuple(int(c * 0.7) for c in color)
    cv2.rectangle(
        img,
        (x0, y0 + 1),
        (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
        txt_bk_color,
        -1
    )
    cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)


def vis_tp_fp(img, boxes, scores, is_tp, gt_boxes, conf=0.5):
    """Draws ground-truth boxes plus predicted boxes labeled by whether each
    prediction is a true or false positive, for visual sanity-checking
    against `yolox.evaluators.pr_metrics`' greedy IoU matching (the same
    rule that drives the PR curve, F1, and best-checkpoint selection) --
    seeing *which* predictions those metrics counted as TP/FP, on the
    actual image, catches matching-logic surprises a scalar metric alone
    would hide (e.g. a "correct-looking" box that's actually a duplicate
    FP because a higher-confidence box already claimed that ground truth).

    Args:
        img: `np.ndarray`, HWC, uint8. Modified in place and returned.
        boxes: `np.ndarray` [N, 4], xyxy, predicted boxes, same pixel space
            as `img`.
        scores: `np.ndarray` [N], predicted confidence scores, same order
            as `boxes`.
        is_tp: `np.ndarray` [N] bool, same order as `boxes` -- e.g. from
            `yolox.evaluators.pr_metrics.match_predictions`.
        gt_boxes: `np.ndarray` [M, 4], xyxy, ground-truth boxes, same pixel
            space as `img`.
        conf: `float`, only predictions with score >= conf are drawn (does
            not affect which ground-truth boxes are drawn -- those have no
            confidence score and are always drawn).

    Returns:
        `img`, modified in place.
    """
    GT_COLOR = (255, 255, 0)
    TP_COLOR = (0, 255, 0)
    FP_COLOR = (255, 0, 0)

    for gt_box in gt_boxes:
        _draw_labeled_box(img, gt_box, GT_COLOR, "GT")

    for i in range(len(boxes)):
        if scores[i] < conf:
            continue
        color = TP_COLOR if is_tp[i] else FP_COLOR
        text = "{}:{:.1f}%".format("TP" if is_tp[i] else "FP", scores[i] * 100)
        _draw_labeled_box(img, boxes[i], color, text)

    return img


_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)
