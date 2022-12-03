import os
import numpy as np
import glob
from torch.utils.data import Dataset
from PIL import Image


# TODO albumentation transform (jsmoon, 2022-11-23)
class TrashDataSet(Dataset):
    def __init__(self, root, transform=None):
        from pycocotools.coco import COCO
        self.root = root
        annot_file_list = glob.glob(root + '/*/*.json')
        self.coco = [COCO(annot_file) for annot_file in annot_file_list]
        self.data_length_per_batch = len(self.coco[0].imgs.keys())
        self.data_size = self.data_length_per_batch * len(self.coco)
        self.transform = transform
        self.cats = self.coco[0].cats

    def __getitem__(self, index):
        image_batch, image_index = index // 500, index % 500
        ann_ids = self.coco[image_batch].getAnnIds(imgIds=image_index)
        target = self.coco[image_batch].loadAnns(ann_ids)
        image_info = self.coco[image_batch].loadImgs(image_index)[0]

        with open(os.path.join(self.root, image_info['file_name']), 'rb') as f:
            img = Image.open(f).convert('RGB')

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return self.data_size

    def get_cats(self):
        return self.cats
