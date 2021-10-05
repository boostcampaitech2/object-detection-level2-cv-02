import cv2
import os
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# from base import BaseModel
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class FasterRCNN_ResNet50_FPN(nn.Module):
    def __init__(self, num_classes: int = 11):
        super().__init__()
        self.superM = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = self.superM.roi_heads.box_predictor.cls_score.in_features
        self.superM.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    def forward(self, x, y):
        x = self.superM(x, y)
        return x

    def inference(self, x):
        x = self.superM(x)
        return x


# class MnistModel(BaseModel):
#     def __init__(self, num_classes=10):
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, num_classes)

#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         x = x.view(-1, 320)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)
