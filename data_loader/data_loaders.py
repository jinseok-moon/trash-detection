from torchvision import transforms
from base import BaseDataLoader
from data_loader.dataset import TrashDataSet


class TrashDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, transform=None, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        tsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ])
        self.transform = transform if transform else tsfm
        self.data_dir = data_dir
        self.dataset = TrashDataSet(data_dir, transform)
        self.data_loader = {}
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
