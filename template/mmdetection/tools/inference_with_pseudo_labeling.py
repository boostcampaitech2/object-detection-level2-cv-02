import mmcv
from mmcv import Config
from mmdet.datasets import build_dataloader, build_dataset, replace_ImageToTensor
from mmdet.models import build_detector
from mmdet.apis import single_gpu_test
from mmcv.runner import load_checkpoint
import os
from mmcv.parallel import MMDataParallel
import pandas as pd
from pandas import DataFrame
from pycocotools.coco import COCO
import numpy as np
import json
from tqdm import tqdm
import argparse


class Coco_json:
    def __init__(self, train_json_path=None):

        # Final Json Dict
        self.obj = {}
        self.test_obj = {}

        # if the train json path is given as a parameter
        if train_json_path:
            with open(train_json_path, "r") as train_json:
                self.obj = json.load(train_json)

        else:
            # info
            self.obj["info"] = {}
            self.obj["info"]["year"] = 0
            self.obj["info"]["version"] = ""
            self.obj["info"]["description"] = ""
            self.obj["info"]["contributor"] = ""
            self.obj["info"]["year"] = None
            self.obj["info"]["date_created"] = ""

            # licenses
            self.obj["licenses"] = []

            # Create init licenses
            self.temp_licenses = {}
            self.temp_licenses["id"] = 0
            self.temp_licenses["name"] = ""
            self.temp_licenses["url"] = ""
            self.obj["licenses"].append(temp_licenses)

            # images
            self.obj["images"] = []

            # categories
            self.obj["categories"] = []

            # annotations
            self.obj["annotations"] = []

        # calculate len
        self.image_len = len(self.obj["images"])
        self.annotations_len = len(self.obj["annotations"])

    def __call__(self):
        return self.obj

    def __str__(self):
        return str(self.obj)

    def add_test_data(self, output, test_json_path):
        with open(test_json_path, "r") as test_json:
            self.test_obj = json.load(test_json)

        class_nums = len(self.test_obj["categories"])

        print("------------- Start Merging train and test images -------------")
        for i, each in tqdm(enumerate(self.test_obj["images"])):
            each["id"] = self.image_len + i

        # Merge train images and test images
        self.obj["images"].extend(self.test_obj["images"])

        print("------------- End -------------")
        print()
        print()
        print("------------- Start Adding test bbox Data -------------")
        cnt = 0
        for i, each in tqdm(enumerate(output)):
            for clss in range(class_nums):
                for box in each[clss]:
                    tmp_box_dict = {}
                    tmp_box_dict["image_id"] = self.image_len + i
                    tmp_box_dict["category_id"] = clss
                    tmp_box_data = [
                        float(box[0]),
                        float(box[1]),
                        float(box[2] - box[0]),
                        float(box[3] - box[1]),
                    ]

                    tmp_box_dict["bbox"] = tmp_box_data
                    tmp_box_dict["area"] = tmp_box_data[2] * tmp_box_data[3]
                    tmp_box_dict["iscrowd"] = 0
                    tmp_box_dict["id"] = self.annotations_len + cnt
                    cnt += 1
                    self.obj["annotations"].append(tmp_box_dict)
        print("------------- End -------------")

        self.image_len += len(self.test_obj["images"])
        self.annotations_len += cnt

    def export_json_file(self, file_path, file_name):
        with open(file_path + file_name, "w") as f:
            json.dump(self.obj, f, indent=4)


def parse_args():
    parser = argparse.ArgumentParser(description="MMDet test (and eval) a model and Create pseudo labeling data file")
    # config 파일 경로 (학습 시킬 때 사용했던 config 파일, work_dir에도 복사되어있음)
    parser.add_argument("config", help="test config file path")
    # peudo labeling 된 새로운 Json파일을 생성할 경로
    parser.add_argument("--new_train_path", default="/opt/ml/detection/dataset/", help="Path to create new train file")
    # peudo labeling 된 새로운 json 파일을 생성할 이름
    parser.add_argument("--new_train_name", default="pseudo_train.json", help="The name of the new file to be created")
    # Original Train json 파일 경로
    parser.add_argument(
        "--original_train", default="/opt/ml/detection/dataset/train.json", help="Path of original train json"
    )
    # checkpoint가 저장되어있는 work_dir 경로
    parser.add_argument("--work_dir", help="the directory to save the file containing evaluation metrics")
    # 사용할 checkpoint epoch
    parser.add_argument("--epoch", default="latest", help="Checkpoint file's epoch")

    parser.add_argument("--show_score_thr", type=float, default=0.05, help="score threshold (default: 0.05)")

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
    submission.to_csv(os.path.join(cfg.work_dir, args.epoch + "_submission.csv"), index=None)
    print(f"submission.csv is saved in {cfg.work_dir}")


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.work_dir:
        cfg.work_dir = args.work_dir
    cfg.data.test.test_mode = True

    # build dataset & dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )

    # checkpoint path
    checkpoint_path = os.path.join(cfg.work_dir, args.epoch + ".pth")

    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))  # build detector
    checkpoint = load_checkpoint(model, checkpoint_path, map_location="cpu")  # ckpt load

    model.CLASSES = dataset.CLASSES
    model = MMDataParallel(model.cuda(), device_ids=[0])

    output = single_gpu_test(model, data_loader, show_score_thr=args.show_score_thr)  # output 계산

    dt = Coco_json(train_json_path=args.original_train)
    dt.add_test_data(output, cfg.data.test.ann_file)

    # Json 파일 출력할 경로와 이름 지정
    dt.export_json_file(args.new_train_path, args.new_train_name)
    make_csv(output, cfg)


if __name__ == "__main__":
    main()
