from torchvision import transforms


class BaseTransform:
    def __init__(self):
        self.transfrom = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ])
