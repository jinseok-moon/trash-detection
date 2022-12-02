import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import torchvision.models as models


class MyFasterRCNN(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = models.detection.fasterrcnn_resnet50_fpn_v2(
            weights=models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
