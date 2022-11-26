import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import torchvision.models as models


class JSFasterRCNN(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.nets = models.detection.fasterrcnn_resnet50_fpn_v2(
            weights=models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
            num_classes=num_classes
        )

    def forward(self, x):
        return self.nets(x)
