import argparse
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint, wrap_fp16_model

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import build_dataloader, build_dataset, replace_ImageToTensor
from mmdet.models import build_detector
import pandas as pd
from pandas import DataFrame
import numpy as np
from pycocotools.coco import COCO


def parse_args():
    parser = argparse.ArgumentParser(description="MMDet test (and eval) a model")
    # config 파일 경로 (학습 시킬 때 사용했던 config 파일, work_dir에도 복사되어있음)
    parser.add_argument("config", help="test config file path")
    # checkpoint가 저장되어있는 work_dir 경로
    parser.add_argument("--work_dir", help="the directory to save the file containing evaluation metrics")
    # 사용할 checkpoint epoch
    parser.add_argument("--epoch", default="latest", help="Checkpoint file's epoch")

    parser.add_argument("--show_score_thr", type=float, default=0.05, help="score threshold (default: 0.05)")
    
    # faster-rcnn 인 경우 
    parser.add_argument("--faster", type=bool, default=False, help="if detector is faster r-cnn , set True")

    args = parser.parse_args()
    return args


def make_csv(output, cfg, faster):
    # submission 양식에 맞게 output 후처리
    prediction_strings = []
    file_names = []
    coco = COCO(cfg.data.test.ann_file)
    img_ids = coco.getImgIds()
    class_num = len(cfg.data.test.classes)
    if faster:
        for i, out in enumerate(output):
            prediction_string = ""
            image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
            for j in range(class_num):
                for o in out[j]:
                    # xmin, ymin, xmax, ymax -> ymin, xmin, ymax, xmax
                    prediction_string += (
                        str(j)
                        + " "
                        + str(o[4])
                        + " "
                        + str(o[1])
                        + " "
                        + str(o[0])
                        + " "
                        + str(o[3])
                        + " "
                        + str(o[2])
                        + " "
                    )

            prediction_strings.append(prediction_string)
            file_names.append(image_info["file_name"])
    else:
        for i, out in enumerate(output):
            prediction_string = ""
            image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
            for j in range(class_num):
                for o in out[j]:
                    prediction_string += (
                        str(j)
                        + " "
                        + str(o[4])
                        + " "
                        + str(o[0])
                        + " "
                        + str(o[1])
                        + " "
                        + str(o[2])
                        + " "
                        + str(o[3])
                        + " "
                    )

            prediction_strings.append(prediction_string)
            file_names.append(image_info["file_name"])

    submission = pd.DataFrame()
    submission["PredictionString"] = prediction_strings
    submission["image_id"] = file_names
    submission.to_csv(os.path.join(cfg.work_dir, "submission.csv"), index=None)
    print(f"submission.csv is saved in {cfg.work_dir}")


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.work_dir:
        cfg.work_dir = args.work_dir
    cfg.data.test.test_mode = True

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )

    checkpoint_path = os.path.join(cfg.work_dir, f"{args.epoch}.pth")

    # build detector
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))

    # ckpt load
    checkpoint = load_checkpoint(model, checkpoint_path, map_location="cpu")

    model.CLASSES = dataset.CLASSES
    model = MMDataParallel(model.cuda(), device_ids=[0])
    # cal ouput
    faster = args.faster
    output = single_gpu_test(model, data_loader, show_score_thr=args.show_score_thr)
    make_csv(output, cfg, faster)


if __name__ == "__main__":
    main()
