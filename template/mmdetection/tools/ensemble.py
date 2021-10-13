# Usage
# python tools/ensemble.py --submission_files /opt/ml/detection/mmdetection/tools/submission_ensemble_iou4.csv /opt/ml/detection/mmdetection/tools/submission_ensemble_iou5.csv /opt/ml/detection/mmdetection/tools/submission_ensemble_iou6.csv --output_csv /opt/ml/detection/mmdetection/tools/final_output.csv
# python tools/ensemble.py --submission_files work_dirs/htc_swin_b/submission.csv work_dirs/htc_swin_b_384/submission.csv work_dirs/htc_swin_b_bifpn/submission.csv work_dirs/hd/output.csv --output_csv final_output.csv

import pandas as pd
from ensemble_boxes import *
import numpy as np
from pycocotools.coco import COCO
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="MMDet ensemble models")
    parser.add_argument("--submission_files", nargs="+", help="submission.csv files path")
    parser.add_argument("--annotation", default="/opt/ml/detection/dataset/test.json")
    parser.add_argument("--output_csv", default="submission.csv")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold (default: 0.5)")
    parser.add_argument("--ensemble", type = str, default = 'wbf', help = 'nms : nms, softnms : soft, non-maximum weighted : nmw, weighted boxes fusion : wbf')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    submission_df = [pd.read_csv(file) for file in args.submission_files]

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
        weights = [1] * len(submission_df) # 모든 모델 동일한 weights. 변경하고 싶으면 weights더 주고 싶은 모델에 [2,1,1]식으로 주면 됨

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
    submission.to_csv(args.output_csv)

    submission.head()


if __name__ == "__main__":
    main()
