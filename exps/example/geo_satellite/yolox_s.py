#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""YOLOX-S fine-tuning exp for detecting GEO satellites in SatSim-generated
16-bit FITS imagery (see satsim/geo_survey_dataset/ in the SatSim repo for
the dataset generation + COCO conversion pipeline this consumes).

Data prerequisites:
  1. Generate imagery: satsim/geo_survey_dataset/generate_dataset.sh
  2. Convert to COCO:  satsim/geo_survey_dataset/convert_to_coco.py
       --input-dir <satsim output root>
     This writes instances_{train,val,test}.json under
     <data-root>/annotations/, with `file_name` entries relative to
     <data-root> -- no images are copied. `self.data_dir` below must be set
     to that same --data-root.

Run (after downloading yolox_s.pth from the README's model zoo):
  python tools/train.py -f exps/example/geo_satellite/yolox_s.py \
      -d 1 -b 16 --fp16 -c /path/to/yolox_s.pth
"""

import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()

        # ---------------- model config ---------------- #
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Single class: SatSim's satnet annotator always writes
        # class_id=1/class_name="Satellite" for the rendered target (see
        # satsim/io/satnet.py); confirmed as the only category produced by
        # convert_to_coco.py against real pilot output.
        self.num_classes = 1

        # ---------------- dataloader config ---------------- #
        # 960x960 (not the 640 stock default): our sensors are 1024-2048px
        # with a ~10-20px point-source target, and YOLOX resizes every image
        # to input_size before the network sees it -- 640 would shrink a
        # 2048px frame ~3.2x, likely destroying the target's signal.
        # multiscale_range=5 (inherited default) gives an effective
        # multiscale training range of ~800-1120.
        self.input_size = (960, 960)
        self.test_size = (960, 960)

        # Set to the --data-root used with convert_to_coco.py (defaults to
        # its --input-dir, i.e. the satsim generate_dataset.sh output root).
        # Override via env var so this file doesn't need hand-editing per
        # machine/run.
        self.data_dir = os.environ.get(
            "SATSIM_GEO_DATA_DIR",
            "/path/to/geo_survey_dataset_output",  # TODO: set SATSIM_GEO_DATA_DIR or edit this
        )
        # Filenames only -- COCODataset joins {data_dir}/annotations/{json_file}.
        self.train_ann = "instances_train.json"
        self.val_ann = "instances_val.json"
        self.test_ann = "instances_test.json"

        # --------------- transform config ----------------- #
        # HSV jitter is disabled: it assumes an 8-bit BGR image (clip bounds
        # are 0-255) and is meaningless anyway on a single-band image
        # replicated into 3 identical channels (saturation is always 0, hue
        # undefined). yolox/data/data_augment.py::augment_hsv now raises
        # ValueError if called on float input as a guard against silently
        # corrupting the normalized [0,1] FITS data -- hsv_prob must stay 0.
        self.hsv_prob = 0.0
        # Other augmentation defaults (mosaic/mixup/flip/degrees/translate/
        # shear) are inherited from yolox/exp/yolox_base.py unchanged: they
        # operate purely geometrically (or via weighted-average blending,
        # for mixup) and were verified to work correctly on our float32
        # data end-to-end (see satsim/geo_survey_dataset/README.md). A
        # rotated/sheared/flipped point-source PSF blob looks essentially
        # the same, so there's no strong reason to weaken them for this
        # single-class, near-point-source-object dataset.

    def get_dataset(self, cache: bool = False, cache_type: str = "ram"):
        from yolox.data import COCODataset, TrainTransform

        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.train_ann,
            name="",  # zero-copy: file_name paths in the json are already
                      # relative to data_dir (see convert_to_coco.py); the
                      # base Exp's get_dataset() omits `name`, which would
                      # default to COCODataset's "train2017" and break path
                      # resolution against our layout.
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=50,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob,
            ),
            cache=cache,
            cache_type=cache_type,
        )

    def get_eval_dataset(self, **kwargs):
        from yolox.data import COCODataset, ValTransform

        testdev = kwargs.get("testdev", False)
        legacy = kwargs.get("legacy", False)

        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.val_ann if not testdev else self.test_ann,
            name="",  # same zero-copy reasoning as get_dataset() above;
                      # base Exp defaults this to "val2017"/"test2017".
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )
