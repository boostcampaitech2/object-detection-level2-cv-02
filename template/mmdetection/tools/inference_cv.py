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

from ensemble_boxes import *

# usage
# python tools/inference_cv.py --config /fold1 /fold2 /fold3 --work_dir ./work_dirs/kfold_test


def parse_args():
    parser = argparse.ArgumentParser(description="MMDet test (and eval) a model")
    # config 파일 경로 (학습 시킬 때 사용했던 config 파일, work_dir에도 복사되어있음)
    parser.add_argument("--config",  nargs="+", help="test config file path")
    # checkpoint가 저장되어있는 work_dir 경로
    parser.add_argument("--work_dir", help="the directory to save the file containing evaluation metrics")
    # 사용할 checkpoint epoch
    parser.add_argument("--epoch", default="latest", help="Checkpoint file's epoch")

    parser.add_argument("--show_score_thr", type=float, default=0.05, help="score threshold (default: 0.05)")
    # k-fold
    parser.add_argument("--k_fold", type=int, default=3, help="num of k_fold")
    parser.add_argument("--annotation", default="/opt/ml/detection/dataset/test.json")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold (default: 0.5)")
    parser.add_argument("--ensemble", type = str, default = 'wbf', help = 'nms : nms, softnms : soft, non-maximum weighted : nmw, weighted boxes fusion : wbf')

    args = parser.parse_args()
    return args


def make_csv(output, cfg):
    # submission 양식에 맞게 output 후처리
    prediction_strings = []
    file_names = []
    coco = COCO(cfg.data.test.ann_file)
    img_ids = coco.getImgIds()
    class_num = len(cfg.data.test.classes)
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

def ensemble_all_fold(args):
    submission_files = []
    for i in range(args.k_fold):
        submission_files.append(args.work_dir + args.config[i] + "/submission.csv")
    submission_df = [pd.read_csv(file) for file in submission_files]

    image_ids = submission_df[0]["image_id"].tolist()
    coco = COCO(args.annotation)

    prediction_strings = []
    file_names = []
    iou_thr = args.iou

    for i, image_id in enumerate(image_ids):
        prediction_string = ""
        boxes_list = []
        scores_list = []
        labels_list = []
        image_info = coco.loadImgs(i)[0]

        for df in submission_df:
            predict_string = df[df["image_id"] == image_id]["PredictionString"].tolist()[0]
            predict_list = str(predict_string).split()

            if len(predict_list) == 0 or len(predict_list) == 1:
                continue

            predict_list = np.reshape(predict_list, (-1, 6))
            box_list = []

            for box in predict_list[:, 2:6].tolist():
                box[0] = float(box[0]) / image_info["width"]
                box[1] = float(box[1]) / image_info["height"]
                box[2] = float(box[2]) / image_info["width"]
                box[3] = float(box[3]) / image_info["height"]
                box_list.append(box)

            boxes_list.append(box_list)
            scores_list.append(list(map(float, predict_list[:, 1].tolist())))
            labels_list.append(list(map(int, predict_list[:, 0].tolist())))

        iou_thr = 0.5  # 대회 규정(GT와 iou 0.5 이상만 private map 계산)에 맞게 0.5로 지정함 
        skip_box_thr = 0.05 # condfidence score가 thr이하인 box는 skip - 박스 너무 많으면 늘려도 됨
        sigma = 0.1 # soft nms parameter
        weights = [1,1,1] # 모델 별 weights : k-fold defalut 모델 3개 이므로 동일한 weight 줌

        if len(boxes_list):
            if args.ensemble == 'nms':
                boxes, scores, labels = nms(boxes_list, scores_list, labels_list, iou_thr=iou_thr)
            elif args.ensemble == 'soft':
                boxes, scores, labels = soft_nms(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, sigma=sigma, thresh=skip_box_thr)
            elif args.ensemble == 'nmw':
                boxes, scores, labels = non_maximum_weighted(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
            else:
                boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
            for box, score, label in zip(boxes, scores, labels):
                prediction_string += (
                    str(label)
                    + " "
                    + str(score)
                    + " "
                    + str(box[0] * image_info["width"])
                    + " "
                    + str(box[1] * image_info["height"])
                    + " "
                    + str(box[2] * image_info["width"])
                    + " "
                    + str(box[3] * image_info["height"])
                    + " "
                )

        prediction_strings.append(prediction_string)
        file_names.append(image_id)

    submission = pd.DataFrame()
    submission["PredictionString"] = prediction_strings
    submission["image_id"] = file_names
    submission.to_csv(args.work_dir + "/submission_cv_all.csv")
    print(f"save {args.work_dir}/submission_cv_all.csv")


def main(fold_i):
    args = parse_args()

    cfg_file = args.work_dir + args.config[fold_i-1] + '/config.py'
    cfg = Config.fromfile(cfg_file)
    if args.work_dir:
        cfg.work_dir = args.work_dir
        
    cfg.work_dir = args.work_dir + f'/fold{fold_i}/'
    
    cfg.data.test.test_mode = True

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )
    
    # fold별 best map pth 파일 지정
    pth_list = os.listdir(cfg.work_dir)
    for pth in pth_list:
        if pth.startswith('best_bbox_mAP'):
            args.epoch = pth
            break
    checkpoint_path = os.path.join(cfg.work_dir, args.epoch)

    # build detector
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))

    # ckpt load
    checkpoint = load_checkpoint(model, checkpoint_path, map_location="cpu")

    model.CLASSES = dataset.CLASSES
    model = MMDataParallel(model.cuda(), device_ids=[0])
    # cal ouput
    output = single_gpu_test(model, data_loader, show_score_thr=args.show_score_thr)
    make_csv(output, cfg)
    
    return args

if __name__ == "__main__":
    k_fold = 3
    args = None
    for fold_i in range(1, k_fold+1):
        print(f"---------------fold{fold_i} inference---------------")
        args = main(fold_i)
        
    ensemble_all_fold(args)