import os
import os.path as osp
import numpy as np
import random
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image, ImageFile
from dataset.autoaugment import ImageNetPolicy
import math

ImageFile.LOAD_TRUNCATED_IMAGES = True


class cityscapes_pseudo_SP_DataSet(data.Dataset):
    def __init__(self, data_root, label_root, list_path, max_iters=None, resize_size=(1024, 512), crop_size=(512, 1024),
                 mean=(128, 128, 128), scale=False, mirror=True, ignore_label=255, set='val', autoaug=False,
                 synthia=False, threshold=1.0):
        self.data_root = data_root
        self.label_root = label_root
        self.list_path = list_path
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.resize_size = resize_size
        self.autoaug = autoaug
        self.h = crop_size[0]
        self.w = crop_size[1]

        self.img_ids = [i_id.strip() for i_id in open(list_path)]

        self.files = []
        self.set = set

        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

        for name in self.img_ids:
            img_file = osp.join(self.data_root, "leftImg8bit/%s/%s" % (self.set, name))
            label_file = osp.join(self.label_root, "%s/%s" % (self.set, name))
            if threshold != 1.0:
                label_file = osp.join(self.label_root, "pseudo_%.1f/%s/%s" % (threshold, self.set, name))
            if synthia:
                label_file = osp.join(self.label_root, "pseudo_SYNTHIA/%s/%s" % (self.set, name))
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        name = datafiles["name"]

      
        image = image.resize((self.resize_size[0], self.resize_size[1]), Image.BICUBIC)
        label = label.resize((self.resize_size[0], self.resize_size[1]), Image.NEAREST)

        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.uint8)
       
        label_copy = label


        size = image.shape
        image = image[:, :, ::-1] 
        image -= self.mean
        image = image.transpose((2, 0, 1))


        return image.copy(), label_copy.copy(), np.array(
            size), name


if __name__ == '__main__':
    # dst = cityscapes_pseudo_DataSet('./data/Cityscapes/data', './dataset/cityscapes_list/train.txt', mean=(0,0,0), set = 'train', autoaug=True)
    dst = cityscapes_pseudo_DataSet('./data/Cityscapes', './dataset/cityscapes_list/train.txt', mean=(0, 0, 0),
                                    set='train', autoaug=True)
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, _, _, _ = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            img = Image.fromarray(np.uint8(img))
            img.save('Cityscape_Demo.jpg')
        break
