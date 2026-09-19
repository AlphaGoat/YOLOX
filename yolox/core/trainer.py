#!/usr/bin/env python3
# Copyright (c) Megvii, Inc. and its affiliates.

import datetime
import math
import os
import signal
import time
from loguru import logger

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from yolox.data import DataPrefetcher
from yolox.evaluators.pr_metrics import (
    best_f1_over_thresholds,
    compute_pr_curve,
    f1_at_threshold,
    log_pr_curve_to_tensorboard,
    match_predictions,
    precision_recall_at_threshold,
)
from yolox.exp import Exp
from yolox.utils import (
    MeterBuffer,
    MlflowLogger,
    ModelEMA,
    WandbLogger,
    adjust_status,
    all_reduce_norm,
    get_local_rank,
    get_model_info,
    get_rank,
    get_world_size,
    gpu_mem_usage,
    is_parallel,
    load_ckpt,
    mem_usage,
    occupy_mem,
    save_checkpoint,
    setup_logger,
    stretch_for_display,
    synchronize,
    vis_tp_fp,
)


class Trainer:
    def __init__(self, exp: Exp, args):
        # init function only defines some basic attr, other attrs like model, optimizer are built in
        # before_train methods.
        self.exp = exp
        self.args = args

        # training related attr
        self.max_epoch = exp.max_epoch
        self.amp_training = args.fp16
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
        self.is_distributed = get_world_size() > 1
        self.rank = get_rank()
        self.local_rank = get_local_rank()
        self.device = "cuda:{}".format(self.local_rank)
        self.use_model_ema = exp.ema
        self.save_history_ckpt = exp.save_history_ckpt

        # data/dataloader related attr
        self.data_type = torch.float16 if args.fp16 else torch.float32
        self.input_size = exp.input_size
        # Best-checkpoint selection metric. Single-class F1, maximized over
        # every confidence threshold each eval (not COCO AP50:95, and not
        # F1 at a single fixed threshold like exp.test_conf) -- with one
        # class and sparse, mostly one-object-per-image ground truth, an
        # IoU/confidence-averaged metric like AP is a weaker proxy for "is
        # this checkpoint good" than F1, and scanning all thresholds avoids
        # the choice depending on exp.test_conf possibly not being the best
        # operating point for a given epoch's model. See
        # yolox/evaluators/pr_metrics.py::best_f1_over_thresholds.
        self.best_metric = 0.0

        # metric record
        self.meter = MeterBuffer(window_size=exp.print_interval)
        self.file_name = os.path.join(exp.output_dir, args.experiment_name)

        if self.rank == 0:
            os.makedirs(self.file_name, exist_ok=True)

        setup_logger(
            self.file_name,
            distributed_rank=self.rank,
            filename="train_log.txt",
            mode="a",
        )

        # Graceful-shutdown support: SIGTERM/SIGINT set a flag instead of
        # acting directly (Python defers the actual handler body to a safe
        # point between bytecode instructions in the main thread, so normal
        # code -- including this logger call -- is fine here, unlike a raw
        # C signal handler). The flag is polled between iterations in
        # `train_in_iter`, never from inside the handler, so a signal can
        # never land mid-`torch.save()` in `save_ckpt` and corrupt
        # `latest_ckpt.pth`. Only catches SIGTERM/SIGINT -- SIGKILL
        # (`kill -9`) cannot be intercepted by any process, so reclaiming
        # the GPU that way still risks losing up to the last completed
        # iteration's checkpoint.
        self._stop_requested = False
        signal.signal(signal.SIGTERM, self._handle_stop_signal)
        signal.signal(signal.SIGINT, self._handle_stop_signal)

    def _handle_stop_signal(self, signum, frame):
        if self._stop_requested:
            # Second signal: the first request didn't stop training fast
            # enough for whoever's asking. Restore default handling and
            # re-raise so this one actually kills the process instead of
            # being silently absorbed again.
            logger.warning(
                "second interrupt (signal {}) received, forcing immediate exit "
                "-- the checkpoint from the graceful stop attempt may be stale "
                "or missing".format(signum)
            )
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)
            return
        logger.warning(
            "received signal {}, will finish the current iteration, save a "
            "checkpoint, and stop".format(signum)
        )
        self._stop_requested = True

    def train(self):
        self.before_train()
        try:
            self.train_in_epoch()
        except Exception as e:
            logger.error("Exception in training: ", e)
            raise
        finally:
            self.after_train()

    def train_in_epoch(self):
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.train_in_iter()
            if self._stop_requested:
                # Skip the normal after_epoch (which would also trigger a
                # full eval pass -- slow, and beside the point when the
                # goal is to free the GPU quickly). start_epoch in the
                # saved checkpoint becomes self.epoch + 1 either way (see
                # save_ckpt), so resuming redoes at most the remainder of
                # this epoch's worth of iterations, never more.
                logger.warning(
                    "stop requested, saving checkpoint at epoch {} and "
                    "ending training early".format(self.epoch + 1)
                )
                self.save_ckpt(ckpt_name="latest")
                break
            self.after_epoch()

    def train_in_iter(self):
        for self.iter in range(self.max_iter):
            self.before_iter()
            self.train_one_iter()
            self.after_iter()
            if self._stop_requested:
                break

    def train_one_iter(self):
        iter_start_time = time.time()

        inps, targets = self.prefetcher.next()
        inps = inps.to(self.data_type)
        targets = targets.to(self.data_type)
        targets.requires_grad = False
        inps, targets = self.exp.preprocess(inps, targets, self.input_size)
        data_end_time = time.time()

        with torch.cuda.amp.autocast(enabled=self.amp_training):
            outputs = self.model(inps, targets)

        loss = outputs["total_loss"]

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.use_model_ema:
            self.ema_model.update(self.model)

        lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        iter_end_time = time.time()
        self.meter.update(
            iter_time=iter_end_time - iter_start_time,
            data_time=data_end_time - iter_start_time,
            lr=lr,
            **outputs,
        )

    def before_train(self):
        logger.info("args: {}".format(self.args))
        logger.info("exp value:\n{}".format(self.exp))

        # model related init
        torch.cuda.set_device(self.local_rank)
        model = self.exp.get_model()
        logger.info(
            "Model Summary: {}".format(get_model_info(model, self.exp.test_size))
        )
        model.to(self.device)

        # solver related init
        self.optimizer = self.exp.get_optimizer(self.args.batch_size)

        # value of epoch will be set in `resume_train`
        model = self.resume_train(model)

        # data related init
        self.no_aug = self.start_epoch >= self.max_epoch - self.exp.no_aug_epochs
        self.train_loader = self.exp.get_data_loader(
            batch_size=self.args.batch_size,
            is_distributed=self.is_distributed,
            no_aug=self.no_aug,
            cache_img=self.args.cache,
        )
        logger.info("init prefetcher, this might take one minute or less...")
        self.prefetcher = DataPrefetcher(self.train_loader)
        # max_iter means iters per epoch
        self.max_iter = len(self.train_loader)

        self.lr_scheduler = self.exp.get_lr_scheduler(
            self.exp.basic_lr_per_img * self.args.batch_size, self.max_iter
        )
        if self.args.occupy:
            occupy_mem(self.local_rank)

        if self.is_distributed:
            model = DDP(model, device_ids=[self.local_rank], broadcast_buffers=False)

        if self.use_model_ema:
            self.ema_model = ModelEMA(model, 0.9998)
            self.ema_model.updates = self.max_iter * self.start_epoch

        self.model = model

        self.evaluator = self.exp.get_evaluator(
            batch_size=self.args.batch_size, is_distributed=self.is_distributed
        )
        # Tensorboard and Wandb loggers
        if self.rank == 0:
            if self.args.logger == "tensorboard":
                self.tblogger = SummaryWriter(os.path.join(self.file_name, "tensorboard"))
            elif self.args.logger == "wandb":
                self.wandb_logger = WandbLogger.initialize_wandb_logger(
                    self.args,
                    self.exp,
                    self.evaluator.dataloader.dataset
                )
            elif self.args.logger == "mlflow":
                self.mlflow_logger = MlflowLogger()
                self.mlflow_logger.setup(args=self.args, exp=self.exp)
            else:
                raise ValueError("logger must be either 'tensorboard', 'mlflow' or 'wandb'")

        logger.info("Training start...")
        logger.info("\n{}".format(model))

    def after_train(self):
        if self._stop_requested:
            logger.info(
                "Training stopped early by signal after epoch {}; best F1 so far "
                "is {:.4f}. Resume with --resume.".format(self.epoch + 1, self.best_metric)
            )
        else:
            logger.info(
                "Training of experiment is done and the best F1 is {:.4f}".format(self.best_metric)
            )
        if self.rank == 0:
            if self.args.logger == "wandb":
                self.wandb_logger.finish()
            elif self.args.logger == "mlflow":
                metadata = {
                    "epoch": self.epoch + 1,
                    "input_size": self.input_size,
                    'start_ckpt': self.args.ckpt,
                    'exp_file': self.args.exp_file,
                    "best_f1": float(self.best_metric)
                }
                self.mlflow_logger.on_train_end(self.args, file_name=self.file_name,
                                                metadata=metadata)

    def before_epoch(self):
        logger.info("---> start train epoch{}".format(self.epoch + 1))

        if self.epoch + 1 == self.max_epoch - self.exp.no_aug_epochs or self.no_aug:
            logger.info("--->No mosaic aug now!")
            self.train_loader.close_mosaic()
            logger.info("--->Add additional L1 loss now!")
            if self.is_distributed:
                self.model.module.head.use_l1 = True
            else:
                self.model.head.use_l1 = True
            self.exp.eval_interval = 1
            if not self.no_aug:
                self.save_ckpt(ckpt_name="last_mosaic_epoch")

    def after_epoch(self):
        self.save_ckpt(ckpt_name="latest")

        if (self.epoch + 1) % self.exp.eval_interval == 0:
            all_reduce_norm(self.model)
            self.evaluate_and_save_model()

    def before_iter(self):
        pass

    def after_iter(self):
        """
        `after_iter` contains two parts of logic:
            * log information
            * reset setting of resize
        """
        # log needed information
        if (self.iter + 1) % self.exp.print_interval == 0:
            # TODO check ETA logic
            left_iters = self.max_iter * self.max_epoch - (self.progress_in_iter + 1)
            eta_seconds = self.meter["iter_time"].global_avg * left_iters
            eta_str = "ETA: {}".format(datetime.timedelta(seconds=int(eta_seconds)))

            progress_str = "epoch: {}/{}, iter: {}/{}".format(
                self.epoch + 1, self.max_epoch, self.iter + 1, self.max_iter
            )
            loss_meter = self.meter.get_filtered_meter("loss")
            loss_str = ", ".join(
                ["{}: {:.1f}".format(k, v.latest) for k, v in loss_meter.items()]
            )

            time_meter = self.meter.get_filtered_meter("time")
            time_str = ", ".join(
                ["{}: {:.3f}s".format(k, v.avg) for k, v in time_meter.items()]
            )

            mem_str = "gpu mem: {:.0f}Mb, mem: {:.1f}Gb".format(gpu_mem_usage(), mem_usage())

            logger.info(
                "{}, {}, {}, {}, lr: {:.3e}".format(
                    progress_str,
                    mem_str,
                    time_str,
                    loss_str,
                    self.meter["lr"].latest,
                )
                + (", size: {:d}, {}".format(self.input_size[0], eta_str))
            )

            if self.rank == 0:
                if self.args.logger == "tensorboard":
                    self.tblogger.add_scalar(
                        "train/lr", self.meter["lr"].latest, self.progress_in_iter)
                    for k, v in loss_meter.items():
                        self.tblogger.add_scalar(
                            f"train/{k}", v.latest, self.progress_in_iter)
                    self.log_weight_stats()
                if self.args.logger == "wandb":
                    metrics = {"train/" + k: v.latest for k, v in loss_meter.items()}
                    metrics.update({
                        "train/lr": self.meter["lr"].latest
                    })
                    self.wandb_logger.log_metrics(metrics, step=self.progress_in_iter)
                if self.args.logger == 'mlflow':
                    logs = {"train/" + k: v.latest for k, v in loss_meter.items()}
                    logs.update({"train/lr": self.meter["lr"].latest})
                    self.mlflow_logger.on_log(self.args, self.exp, self.epoch+1, logs)

            self.meter.clear_meters()

        # random resizing
        if (self.progress_in_iter + 1) % 10 == 0:
            self.input_size = self.exp.random_resize(
                self.train_loader, self.epoch, self.rank, self.is_distributed
            )

    def log_weight_stats(self):
        """Logs aggregate parameter and gradient statistics (mean, std, L2
        grad norm), pooled across every trainable parameter, to TensorBoard
        -- called every `print_interval` iterations (same cadence as the
        loss/lr scalars just above this call), so a loss spike or NaN can
        be correlated against weight/gradient behavior on the same
        iteration axis. Useful for spotting training instability
        (gradient explosion/vanishing, runaway weight growth) at a glance;
        `log_weight_histograms` (called once per epoch from
        `evaluate_and_save_model`) gives the heavier per-layer detail to
        diagnose *which* layer if this aggregate view looks off.

        Called from `after_iter()`, i.e. after `train_one_iter()` (backward
        + optimizer step) has fully completed for this iteration -- under
        AMP (`--fp16`), `GradScaler.step()` unscales gradients in place
        before applying (or skipping, on an inf/nan step) the optimizer
        step, so `.grad` already holds real, comparable-across-iterations
        values here with no extra unscaling needed. Gradients aren't
        cleared until the *next* iteration's `zero_grad()`, so they're
        still live at this point.
        """
        model = self.model.module if is_parallel(self.model) else self.model

        try:
            weight_vals = []
            grad_vals = []
            grad_sq_sum = 0.0
            for p in model.parameters():
                if not p.requires_grad:
                    continue
                weight_vals.append(p.detach().reshape(-1).float())
                if p.grad is not None:
                    g = p.grad.detach().reshape(-1).float()
                    grad_vals.append(g)
                    grad_sq_sum += g.pow(2).sum().item()

            if weight_vals:
                w = torch.cat(weight_vals)
                w_mean, w_std = w.mean().item(), w.std().item()
                if math.isfinite(w_mean) and math.isfinite(w_std):
                    self.tblogger.add_scalar("train/weights/mean", w_mean, self.progress_in_iter)
                    self.tblogger.add_scalar("train/weights/std", w_std, self.progress_in_iter)
                else:
                    logger.warning(
                        "train/weights/mean or /std is non-finite at iter {}, skipping "
                        "-- model weights have diverged".format(self.progress_in_iter)
                    )

            if grad_vals:
                g = torch.cat(grad_vals)
                g_mean, g_std, g_norm = g.mean().item(), g.std().item(), grad_sq_sum ** 0.5
                if math.isfinite(g_mean) and math.isfinite(g_std) and math.isfinite(g_norm):
                    self.tblogger.add_scalar("train/grad/mean", g_mean, self.progress_in_iter)
                    self.tblogger.add_scalar("train/grad/std", g_std, self.progress_in_iter)
                    self.tblogger.add_scalar("train/grad/norm", g_norm, self.progress_in_iter)
                else:
                    # Unlike a non-finite weight, a non-finite pooled
                    # gradient here is expected and benign on this sparse,
                    # one-object-per-image dataset with small batches:
                    # SimOTA's dynamic label assignment can assign zero
                    # foreground anchors for a batch, producing a NaN
                    # gradient for some parameters without destabilizing
                    # training (AMP's GradScaler detects the same condition
                    # and skips that optimizer step entirely -- confirmed
                    # via a real smoke-test run, 2026-09-19). Plain tensor
                    # .mean()/.std() don't raise on NaN/Inf input -- they
                    # silently return NaN/Inf -- so this can't rely on the
                    # try/except below, which only catches actual
                    # exceptions and would never fire for this; the values
                    # are checked explicitly instead and skipped (not
                    # logged as NaN) so the TensorBoard chart doesn't get a
                    # literal NaN point breaking the line.
                    logger.warning(
                        "train/grad/* is non-finite at iter {}, skipping this point "
                        "(likely a zero-foreground-anchor batch, not a crash)".format(self.progress_in_iter)
                    )
        except Exception:
            # Metrics logging must never take down an actual training run --
            # and NaN/Inf parameters or gradients (exactly the instability
            # this is meant to help catch) are the case most likely to
            # trip an edge case here. logger.opt(exception=True), not
            # logger.warning(..., exc_info=True) -- see log_weight_histograms.
            logger.opt(exception=True).warning("log_weight_stats failed, skipping this iteration")

    @property
    def progress_in_iter(self):
        return self.epoch * self.max_iter + self.iter

    def resume_train(self, model):
        if self.args.resume:
            logger.info("resume training")
            if self.args.ckpt is None:
                ckpt_file = os.path.join(self.file_name, "latest" + "_ckpt.pth")
            else:
                ckpt_file = self.args.ckpt

            ckpt = torch.load(ckpt_file, map_location=self.device)
            # resume the model/optimizer state dict
            model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.best_metric = ckpt.pop("best_metric", 0.0)
            # resume the training states variables
            start_epoch = (
                self.args.start_epoch - 1
                if self.args.start_epoch is not None
                else ckpt["start_epoch"]
            )
            self.start_epoch = start_epoch
            logger.info(
                "loaded checkpoint '{}' (epoch {})".format(
                    self.args.resume, self.start_epoch
                )
            )  # noqa
        else:
            if self.args.ckpt is not None:
                logger.info("loading checkpoint for fine tuning")
                ckpt_file = self.args.ckpt
                ckpt = torch.load(ckpt_file, map_location=self.device)["model"]
                model = load_ckpt(model, ckpt)
            self.start_epoch = 0

        return model

    def evaluate_and_save_model(self):
        if self.use_model_ema:
            evalmodel = self.ema_model.ema
        else:
            evalmodel = self.model
            if is_parallel(evalmodel):
                evalmodel = evalmodel.module

        with adjust_status(evalmodel, training=False):
            (ap50_95, ap50, summary), predictions = self.exp.eval(
                evalmodel, self.evaluator, self.is_distributed, return_outputs=True
            )

        # Best-checkpoint selection uses single-class F1 maximized over
        # every confidence threshold (an oracle/best-achievable metric),
        # not COCO AP50:95 and not F1 at a single fixed threshold -- see
        # Trainer.__init__ for why. Computed once here and passed to
        # log_precision_recall below rather than recomputed there, since
        # compute_pr_curve walks every validation prediction.
        #
        # F1 at exp.test_conf is also still computed and logged (val/f1) as
        # a separate, more conservative diagnostic: "what F1 do I actually
        # get at the threshold I'll deploy with", vs. val/f1_best's "what's
        # the best F1 any threshold could get this checkpoint" (which is
        # what drives best_ckpt.pth). The two will diverge whenever
        # exp.test_conf isn't this epoch's actual optimum.
        coco_gt = self.evaluator.dataloader.dataset.coco
        pr_curve = compute_pr_curve(predictions, coco_gt, iou_thresh=0.5)
        f1_at_test_conf = f1_at_threshold(pr_curve, self.exp.test_conf)
        f1_best, f1_best_thresh = best_f1_over_thresholds(pr_curve)

        update_best_ckpt = f1_best > self.best_metric
        self.best_metric = max(self.best_metric, f1_best)

        if self.rank == 0:
            if self.args.logger == "tensorboard":
                self.tblogger.add_scalar("val/COCOAP50", ap50, self.epoch + 1)
                self.tblogger.add_scalar("val/COCOAP50_95", ap50_95, self.epoch + 1)
                self.tblogger.add_scalar("val/f1", f1_at_test_conf, self.epoch + 1)
                self.tblogger.add_scalar("val/f1_best", f1_best, self.epoch + 1)
                if not np.isnan(f1_best_thresh):
                    self.tblogger.add_scalar("val/f1_best_threshold", f1_best_thresh, self.epoch + 1)
                self.log_prediction_images(predictions)
                self.log_precision_recall(pr_curve)
                self.log_weight_histograms()
            if self.args.logger == "wandb":
                self.wandb_logger.log_metrics({
                    "val/COCOAP50": ap50,
                    "val/COCOAP50_95": ap50_95,
                    "val/f1": f1_at_test_conf,
                    "val/f1_best": f1_best,
                    "val/f1_best_threshold": f1_best_thresh,
                    "train/epoch": self.epoch + 1,
                })
                self.wandb_logger.log_images(predictions)
            if self.args.logger == "mlflow":
                logs = {
                    "val/COCOAP50": ap50,
                    "val/COCOAP50_95": ap50_95,
                    "val/f1": f1_at_test_conf,
                    "val/f1_best": f1_best,
                    "val/f1_best_threshold": f1_best_thresh,
                    "val/best_f1": round(self.best_metric, 3),
                    "train/epoch": self.epoch + 1,
                }
                self.mlflow_logger.on_log(self.args, self.exp, self.epoch+1, logs)
            logger.info("\n" + summary)
        synchronize()

        self.save_ckpt(
            "last_epoch", update_best_ckpt, ap=ap50_95, f1=f1_best, f1_threshold=f1_best_thresh
        )
        if self.save_history_ckpt:
            self.save_ckpt(
                f"epoch_{self.epoch + 1}", ap=ap50_95, f1=f1_best, f1_threshold=f1_best_thresh
            )

        if self.args.logger == "mlflow":
            metadata = {
                    "epoch": self.epoch + 1,
                    "input_size": self.input_size,
                    'start_ckpt': self.args.ckpt,
                    'exp_file': self.args.exp_file,
                    "best_f1": float(self.best_metric)
                }
            self.mlflow_logger.save_checkpoints(self.args, self.exp, self.file_name, self.epoch,
                                                metadata, update_best_ckpt)

    def log_prediction_images(self, predictions, max_images=8):
        """Logs a handful of validation images with ground-truth boxes plus
        predicted boxes (labeled TP or FP) drawn in, to TensorBoard, for
        visual sanity-checking during training.

        TensorBoard has no declarative box-overlay format (unlike wandb's
        `wandb.Image(img, boxes=...)`, used elsewhere in this file), so
        boxes are drawn directly into the pixel data via `vis_tp_fp()`.
        TP/FP status comes from the same greedy IoU matching
        (`match_predictions`, iou_thresh=0.5) that drives the PR curve, F1,
        and best-checkpoint selection elsewhere in this file -- so a box
        labeled TP/FP here is exactly what those metrics counted it as, not
        a separately-eyeballed judgment call. A ground-truth box with no
        nearby TP-colored box is a visible miss (false negative), without
        needing a separate label for that case.

        `predictions` values and ground-truth boxes are both in *original*
        (pre-resize) image pixel space (see COCOEvaluator.convert_to_coco_
        format for predictions: `bboxes /= scale` undoes the resize before
        storing; ground truth comes straight from the COCO json built by
        convert_to_coco.py against full sensor width/height), so both are
        re-scaled by the same `min(test_size / orig_size)` factor the
        evaluator used before being drawn onto the resized-but-unpadded
        image `pull_item` returns -- drawing unscaled original-space boxes
        onto a resized image would misalign them.

        Images are decoded from 16-bit FITS and kept as normalized float32
        for training precision (see yolox/data/datasets/coco.py); real
        frames only use a small slice of that value range, so
        `stretch_for_display()` (percentile clip + linear stretch) is
        applied before drawing/logging -- without it, most frames would
        look almost entirely black.
        """
        dataset = self.evaluator.dataloader.dataset
        coco_gt = dataset.coco
        test_size = self.exp.test_size
        # Single-class assumption already made throughout this file (see
        # compute_pr_curve's default class_id) -- the only category present.
        class_id = coco_gt.getCatIds()[0]

        ids = dataset.ids[:max_images]
        if not ids:
            return

        for img_id in ids:
            index = dataset.ids.index(img_id)
            img, _, (orig_h, orig_w), _ = dataset.pull_item(index)
            scale = min(test_size[0] / orig_h, test_size[1] / orig_w)

            ann_ids = coco_gt.getAnnIds(imgIds=[int(img_id)], catIds=[class_id], iscrowd=False)
            gt_anns = coco_gt.loadAnns(ann_ids)
            gt_boxes = np.array(
                [[a["bbox"][0], a["bbox"][1],
                  a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
                 for a in gt_anns],
                dtype=np.float32,
            ).reshape(-1, 4) * scale

            pred = predictions.get(
                int(img_id), {"bboxes": [], "scores": [], "categories": []}
            )
            keep = np.array(pred["categories"]) == class_id
            boxes = np.array(pred["bboxes"], dtype=np.float32).reshape(-1, 4)[keep] * scale
            scores = np.array(pred["scores"], dtype=np.float32)[keep]
            is_tp = match_predictions(boxes, scores, gt_boxes, iou_thresh=0.5)

            disp = stretch_for_display(img)
            disp = vis_tp_fp(disp.copy(), boxes, scores, is_tp, gt_boxes, conf=0.3)

            self.tblogger.add_image(
                f"val/predictions/{img_id}", disp, self.epoch + 1, dataformats="HWC"
            )

    def log_precision_recall(self, pr_curve):
        """Logs single-class precision/recall to TensorBoard, plus a full
        PR curve, every eval.

        `COCOEvaluator` only surfaces aggregate COCO AP/AR (val/COCOAP50,
        val/COCOAP50_95, logged just above this call) -- not precision or
        recall at a specific confidence threshold, and not a PR curve at
        all. See yolox/evaluators/pr_metrics.py for why those had to be
        computed directly (greedy IoU matching) rather than read out of
        pycocotools' internal COCOeval state.

        `pr_curve` is precomputed by the caller (evaluate_and_save_model,
        via compute_pr_curve) since it's also needed there for F1-based
        best-checkpoint selection -- recomputing it here would walk every
        validation prediction a second time for no reason.

        `val/precision` and `val/recall` are reported at `self.exp.test_conf`
        -- the same confidence threshold actually used to filter detections
        at inference (see COCOEvaluator.evaluate's `postprocess` call), so
        these answer "what precision/recall does the model get as deployed"
        rather than an aggregate across all thresholds.
        """
        precision, recall = precision_recall_at_threshold(pr_curve, self.exp.test_conf)
        # precision is NaN when nothing scored >= test_conf yet (common
        # early in training) -- skip the scalar point rather than log a
        # misleading 0, which would look like "bad model" instead of "no
        # confident predictions yet". recall is always well-defined (0 in
        # that case, since the ground-truth count is unaffected).
        if not np.isnan(precision):
            self.tblogger.add_scalar("val/precision", precision, self.epoch + 1)
        self.tblogger.add_scalar("val/recall", recall, self.epoch + 1)

        log_pr_curve_to_tensorboard(self.tblogger, pr_curve, self.epoch + 1)

    def log_weight_histograms(self):
        """Logs a weight and gradient distribution histogram per named
        parameter tensor to TensorBoard, once per epoch.

        Heavier than `log_weight_stats`' pooled-aggregate scalars (one
        histogram per parameter tensor -- ~150-200 for YOLOX-S -- rather
        than a handful of numbers), so it runs at the coarser per-epoch
        cadence used by the other eval-time TensorBoard additions here
        (log_prediction_images, log_precision_recall) instead of every
        print_interval iterations. This is what answers "which layer" if
        the pooled aggregate in log_weight_stats looks unstable -- e.g. a
        single layer's weights or gradients drifting/exploding can be
        invisible in a global mean/std pooled across every parameter, but
        is visible as that one tag's histogram spreading out or shifting
        over epochs.

        Uses whatever `.grad` currently holds, i.e. the last training
        iteration's gradient this epoch (gradients aren't cleared until
        the next iteration's zero_grad()) -- a fair per-epoch snapshot for
        stability monitoring even though it isn't every iteration's
        gradient the way log_weight_stats' scalars are.
        """
        model = self.model.module if is_parallel(self.model) else self.model

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            try:
                self.tblogger.add_histogram(
                    f"weights/{name}", p.detach().reshape(-1).float().cpu(), self.epoch + 1
                )
                if p.grad is not None:
                    self.tblogger.add_histogram(
                        f"grads/{name}", p.grad.detach().reshape(-1).float().cpu(), self.epoch + 1
                    )
            except Exception:
                # As in log_weight_stats: a single degenerate tensor (e.g.
                # NaN/Inf from real instability -- confirmed to happen in
                # practice for a fresh, randomly-initialized model: the
                # single-element head.obj_preds.*.bias gradient can be NaN
                # early on if a batch assigns zero foreground anchors) must
                # not take down the rest of the run or the other
                # parameters' histograms. logger.opt(exception=True), not
                # logger.warning(..., exc_info=True) -- the latter is the
                # stdlib `logging` kwarg, not loguru's API; it's silently
                # accepted but does not attach a traceback.
                logger.opt(exception=True).warning(f"log_weight_histograms failed for '{name}', skipping")

    def save_ckpt(self, ckpt_name, update_best_ckpt=False, ap=None, f1=None, f1_threshold=None):
        if self.rank == 0:
            save_model = self.ema_model.ema if self.use_model_ema else self.model
            logger.info("Save weights to {}".format(self.file_name))
            ckpt_state = {
                "start_epoch": self.epoch + 1,
                "model": save_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "best_metric": self.best_metric,
                "curr_ap": ap,
                # curr_f1 is the best-over-all-thresholds F1 (see
                # best_f1_over_thresholds) -- the metric that actually
                # decides update_best_ckpt -- and curr_f1_threshold is the
                # confidence threshold that achieves it. For best_ckpt.pth
                # specifically, curr_f1_threshold is a genuinely useful
                # number: it's the threshold you'd want to set exp.test_conf
                # to in order to actually realize this checkpoint's best F1
                # at inference, rather than whatever test_conf happened to
                # be during training.
                "curr_f1": f1,
                "curr_f1_threshold": f1_threshold,
            }
            save_checkpoint(
                ckpt_state,
                update_best_ckpt,
                self.file_name,
                ckpt_name,
            )

            if self.args.logger == "wandb":
                self.wandb_logger.save_checkpoint(
                    self.file_name,
                    ckpt_name,
                    update_best_ckpt,
                    metadata={
                        "epoch": self.epoch + 1,
                        "optimizer": self.optimizer.state_dict(),
                        "best_metric": self.best_metric,
                        "curr_ap": ap,
                        "curr_f1": f1,
                        "curr_f1_threshold": f1_threshold,
                    }
                )
