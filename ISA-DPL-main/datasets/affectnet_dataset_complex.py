import torch.utils.data as data
from PIL import Image, ImageFile
import os
import random
import cv2
import numpy as np
ImageFile.LOAD_TRUNCATED_IAMGES = True  # 自动剔除坏数据


def switch_expression(expression_argument):
    switcher = {
        0: 'neutral',
        1: 'Happiness',
        2: 'Sadness',
        3: 'Surprise',
        4: 'Fear',
        5: 'Disgust',
        6: 'Anger',
        7: 'Contempt',
        8: 'None',
        9: 'Uncertain',
        10: 'No-Face'
    }
    return switcher.get(expression_argument, 0)


class AffectNet(data.Dataset):
    def __init__(self, root, fileList, num_classes=7, transform=None, phase='train'):
        self.root = root
        self.label_list = []
        self.cls_num = num_classes
        self.imgList, self.num_per_cls_dict = self.default_reader(fileList, self.cls_num)
        self.transform = transform
        self.fileList = fileList
        self.aug_func = [flip_image, add_gaussian_noise]
        self.phase = phase

    def __getitem__(self, index):
        global img
        imgPath, target_expression = self.imgList[index]
        img_complete_path = os.path.join(self.root, imgPath)

        try:
            with open(img_complete_path, 'rb') as f:
                img = cv2.imread(img_complete_path)
                img = img[:, :, ::-1]  # BGR to RGB
        except IOError:
            print('Cannot load image ' + img_complete_path)

        if self.phase == 'train':
            if random.uniform(0, 1) > 0.5:
                index = random.randint(0, 1)
                img = self.aug_func[index](img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target_expression

    def __len__(self):
        return len(self.imgList)

    def get_labels(self):
        return self.label_list

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

    def default_reader(self, fileList, num_classes):
        imgList = []
        if fileList.find('validation.csv') > -1:
            start_index = 0
            max_samples = 100000
        else:
            start_index = 1
            max_samples = 20000

        num_per_cls_dict = dict()
        for i in range(0, num_classes):
            num_per_cls_dict[i] = 0

        if num_classes == 7:
            exclude_list = [7, 8, 9, 10]
        else:
            exclude_list = [8, 9, 10]

        expression_0 = 0
        expression_1 = 0
        expression_2 = 0
        expression_3 = 0
        expression_4 = 0
        expression_5 = 0
        expression_6 = 0
        expression_7 = 0

        # f = open('../datasets/AffectNet/labels/validation.csv', 'r')
        # lines = f.readlines()
        # random.shuffle(lines)

        # if fileList.find('occlusion') > -1:
        #     fp = open(fileList, 'r')
        #     for names in fp.readlines():
        #         _, target, image_path, _ = names.split('/')
        #         image_path = image_path.strip()
        #
        #         for line in lines:
        #             if line.find(image_path) > -1:
        #
        #                 imgPath = line.strip().split(',')[0]
        #                 (x, y, w, h) = line.strip().split(',')[1:5]
        #
        #                 expression = int(line.strip().split(',')[6])
        #                 if expression not in exclude_list:
        #                     imgList.append([imgPath, (int(x), int(y), int(w), int(h)), expression])
        #                     num_per_cls_dict[expression] = num_per_cls_dict[expression] + 1
        #     fp.close()
        #     return imgList, num_per_cls_dict
        #
        # elif fileList.find('pose') > -1:
        #     fp = open(fileList, 'r')
        #     for names in fp.readlines():
        #         target, image_path = names.split('/')
        #         image_path = image_path.strip()
        #         for line in lines:
        #             if line.find(image_path) > -1:
        #                 imgPath = line.strip().split(',')[0]
        #                 (x, y, w, h) = line.strip().split(',')[1:5]
        #                 expression = int(line.strip().split(',')[6])
        #                 if expression not in exclude_list:
        #                     imgList.append([imgPath, (int(x), int(y), int(w), int(h)), expression])
        #                     num_per_cls_dict[expression] = num_per_cls_dict[expression] + 1
        #
        #     fp.close()
        #     return imgList, num_per_cls_dict

        # else:  # training or validation affectnet set
        fp = open(fileList, 'r')
        for line in fp.readlines()[start_index:]:
            imgPath = line.strip().split(',')[0]  # 包含分卷子地址
            (x, y, w, h) = line.strip().split(',')[1:5]
            landmarks = line.strip().split(',')[5]
            expression = int(line.strip().split(',')[6])

            if expression == 0:
                expression_0 = expression_0 + 1
                if expression_0 > max_samples:
                    continue

            if expression == 1:
                expression_1 = expression_1 + 1
                if expression_1 > max_samples:
                    continue

            if expression == 2:
                expression_2 = expression_2 + 1
                if expression_2 > max_samples:
                    continue

            if expression == 3:
                expression_3 = expression_3 + 1
                if expression_3 > max_samples:
                    continue

            if expression == 4:
                expression_4 = expression_4 + 1
                if expression_4 > max_samples:
                    continue

            if expression == 5:
                expression_5 = expression_5 + 1
                if expression_5 > max_samples:
                    continue

            if expression == 6:
                expression_6 = expression_6 + 1
                if expression_6 > max_samples:
                    continue

            if expression == 7:
                expression_7 = expression_7 + 1
                if expression_7 > max_samples:
                    continue

            # Adding only list of first 8 expressions
            if expression not in exclude_list:
                imgList.append([imgPath, expression])
                num_per_cls_dict[expression] = num_per_cls_dict[expression] + 1
                self.label_list.append(expression)

        fp.close()
        return imgList, num_per_cls_dict


def add_gaussian_noise(image_array, mean=0.0, var=30):
    std = var ** 0.5
    noisy_img = image_array + np.random.normal(mean, std, image_array.shape)
    noisy_img_clipped = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img_clipped

def flip_image(image_array):
    return cv2.flip(image_array, 1)
