import argparse
import collections
from pycocotools.coco import COCO

import data_loader.data_loaders as module_data
import model.model as module_arch


import albumentations as A
import torch

from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm

from parse_config import ConfigParser


def inference_fn(test_data_loader, model, device):
    outputs = []
    for images in tqdm(test_data_loader):
        # gpu 계산을 위해 image.to(device)
        images = list(image.to(device) for image in images)
        output = model.inference(images)
        for out in output:
            outputs.append(
                {"boxes": out["boxes"].tolist(), "scores": out["scores"].tolist(), "labels": out["labels"].tolist()}
            )
    return outputs


def main(config):
    test_dataset = config.init_obj("inference_data_loader", module_data)
    score_threshold = 0.05
    check_point = config["inference_check_point"]

    test_data_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)

    model = config.init_obj("arch", module_arch)
    model.to(device)
    st_dict = torch.load(check_point)["state_dict"]

    model.load_state_dict(st_dict)
    model.eval()

    outputs = inference_fn(test_data_loader, model, device)
    prediction_strings = []
    file_names = []
    coco = COCO(config["inference_data_loader"]["args"]["annotation"])

    # submission 파일 생성
    for i, output in enumerate(outputs):
        prediction_string = ""
        image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
        for box, score, label in zip(output["boxes"], output["scores"], output["labels"]):
            if score > score_threshold:
                # label[1~10] -> label[0~9]
                prediction_string += (
                    str(label - 1)
                    + " "
                    + str(score)
                    + " "
                    + str(box[0])
                    + " "
                    + str(box[1])
                    + " "
                    + str(box[2])
                    + " "
                    + str(box[3])
                    + " "
                )
        prediction_strings.append(prediction_string)
        file_names.append(image_info["file_name"])
    submission = pd.DataFrame()
    submission["PredictionString"] = prediction_strings
    submission["image_id"] = file_names
    submission.to_csv(config["inference_file_name"], index=None)
    print(submission.head())


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--ifn", "--inference_file_name"], type=str, target="inference_file_name"),
        CustomArgs(["--ifcp", "--inference_check_point"], type=str, target="inference_check_point"),
        CustomArgs(
            ["--ifds", "--inference_data_set"],
            type=str,
            target="inference_data_loader;type",
        ),
    ]
    config = ConfigParser.from_args(args, options)

    main(config)
