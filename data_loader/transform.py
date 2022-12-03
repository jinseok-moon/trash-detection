import torch
import data_loader.vision.transforms as T


def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


class BaseTransform:
    def __init__(self, train):
        self.transform = get_transform(train)