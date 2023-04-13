import os
import torch
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt


class MEGC2019(torch.utils.data.Dataset):
    """MEGC2019 dataset class with 3 categories"""

    def __init__(self, imgList, transform=None):
        self.imgPath = []
        self.label = []
        self.dbtype = []
        with open(imgList, 'r') as f:
            for textline in f:
                texts = textline.strip('\n').split(' ')
                self.imgPath.append(texts[0])
                self.label.append(int(texts[1]))
                self.dbtype.append(int(texts[2]))
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.open("".join(self.imgPath[idx]), 'r').convert('RGB')
        # plt.imshow(img)
        # plt.show()
        if self.transform is not None:
            img = self.transform(img)
        return img, self.label[idx]

    def __len__(self):
        return len(self.imgPath)


class MEGC2019_SI_useNP(torch.utils.data.Dataset):
    """MEGC2019_SI dataset class with 3 categories and other side information"""

    def __init__(self, imgList, transform=None):
        self.imgPath = []
        self.label = []
        self.dbtype = []
        with open(imgList, 'r') as f:
            for textline in f:
                texts = textline.strip('\n').split(' ')
                self.imgPath.append(texts[0])
                self.label.append(int(texts[1]))
                self.dbtype.append(int(texts[2]))
        self.transform = transform

    def __getitem__(self, idx):
        img = np.load(self.imgPath[idx])
        of = torch.FloatTensor(img['of'])
        # of_patches = torch.FloatTensor(img['of_patches'])
        of = of.permute(2, 0, 1)

        # if self.transform is not None:
        #     of = self.transform(of)
        return {"data": of, "class_label": self.label[idx], 'db_label': self.dbtype[idx]}

    def __len__(self):
        return len(self.imgPath)
