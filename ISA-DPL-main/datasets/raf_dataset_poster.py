import os
import cv2
import numpy as np
import torch
from PIL import Image
import random
from torch.utils.data import Dataset


class Dataset_RAF(Dataset):
    def __init__(self, img_path, label_path, transform=None, phase='train'):
        self.img_path = img_path
        self.label_path = label_path
        self.transform = transform
        self.phase = phase

        image_list = []
        label_list = []

        with open(label_path) as f:
            img_label_list = f.read().splitlines()
        for info in img_label_list:
            image_name, label_name = info.split(',')
            image_list.append(image_name)
            label_list.append(int(label_name))

        self.image_list = image_list
        self.label_list = label_list

        self.aug_func = [flip_image, add_gaussian_noise]

    def __getitem__(self, index):
        global img
        img_name = self.image_list[index]
        label_name = self.label_list[index]

        path = os.path.join(self.img_path, img_name)
        try:
            with open(path, 'rb') as f:
              img = cv2.imread(path)
        except IOError:
            print('Cannot load image ' + path)

        if self.phase == 'train':
            if random.uniform(0, 1) > 0.5:
                index = random.randint(0, 1)
                img = self.aug_func[index](img)

        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img.copy())

        return img, label_name

    def __len__(self):
        return len(self.image_list)

    # 统计整个数据集各个分类的大致分布
    def get_weights(self):
        labels = sorted(set(self.label_list))  # 0~6
        total = len(self) / len(labels)  # 12271/7
        label_list = torch.tensor(self.label_list)
        counts = [total/(label_list == l).sum() for l in labels]
        weights = torch.tensor(counts)

        return weights


def add_gaussian_noise(image_array, mean=0.0, var=30):
    std = var ** 0.5
    noisy_img = image_array + np.random.normal(mean, std, image_array.shape)
    noisy_img_clipped = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img_clipped

def flip_image(image_array):
    return cv2.flip(image_array, 1)
