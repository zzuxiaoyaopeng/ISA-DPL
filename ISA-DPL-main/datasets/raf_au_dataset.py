import os
import cv2
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class Dataset_RAF_AU(Dataset):
    def __init__(self, img_path, label_path, transform=None):
        self.img_path = img_path
        self.label_path = label_path
        self.transform = transform

        image_list = []
        label_list = []

        with open(label_path) as f:
            img_label_list = f.read().splitlines()
        for info in img_label_list:
            image_name, label_name = info.split(';')
            image_list.append(image_name)
            label_list.append(label_name)

        self.image_list = image_list
        self.label_list = label_list

    def __getitem__(self, index):
        global img
        img = self.image_list[index]
        label = self.label_list[index]  # 21个数值,逗号隔开的
        label_info = label.split(',')  # 列表存储21个值

        path = os.path.join(self.img_path, img)
        try:
            with open(path, 'rb') as f:
                img = cv2.imread(path)
                img = Image.fromarray(img)
        except IOError:
            print('Cannot load image ' + path)

        if self.transform is not None:
            img = self.transform(img)

        return img, label_info

    def __len__(self):
        return len(self.image_list)
