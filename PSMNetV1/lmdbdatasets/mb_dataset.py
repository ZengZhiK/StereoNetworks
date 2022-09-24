# -*- coding: UTF-8 -*-
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import random
import glob
import warnings
from .data_io import get_transform, readPFM

warnings.filterwarnings('ignore', '.*output shape of zoom.*')


def load_disp(path):
    if '.png' in path:
        data = Image.open(path)
        data = np.ascontiguousarray(data, dtype=np.float32) / 256
        return data
    else:
        return readPFM(path)[0]


def get_mb_list(filepath):
    left_list = glob.glob(f'{filepath}/*/im0.png')
    left_list = sorted(left_list)
    right_list = [img.replace('im0', 'im1') for img in left_list]
    disp_list = [img.replace('im0.png', 'disp0GT.pfm') for img in left_list]
    return left_list, right_list, disp_list


class MbDataset(Dataset):
    def __init__(self, data_dir, training=False):
        left, right, left_disparity = get_mb_list(data_dir)
        self.training = training
        self.left = left
        self.right = right
        self.disp_L = left_disparity

    def __len__(self):
        return len(self.left)

    def __getitem__(self, index):
        left = self.left[index]
        right = self.right[index]
        disp_L = self.disp_L[index]

        left_img = Image.open(left).convert('L')
        right_img = Image.open(right).convert('L')
        dataL = load_disp(disp_L)
        # 视差无效值处理
        dataL[dataL == np.inf] = 0
        default_disp = 192.0

        if self.training:
            w, h = left_img.size
            crop_h, crop_w = 320, 640
            x1 = random.randint(0, w - crop_w)
            y1 = random.randint(0, h - crop_h)

            # random crop
            left_img = left_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            right_img = right_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            dataL = Image.fromarray(dataL)
            dataL = dataL.crop((x1, y1, x1 + crop_w, y1 + crop_h))
        else:
            w, h = left_img.size
            orgin_width = w
            # 读取数据集最大视差范围
            with open(left.replace('im0.png', 'calib.txt')) as f:
                lines = f.readlines()
                max_disp = int(int(lines[6].split('=')[-1]))
                # 如果数据集最大视差大于默认视差，则进行缩放
                if max_disp > default_disp:
                    s = max_disp * 1.0 / default_disp
                    w = w // s
                    h = h // s

            h1 = h % 64
            w1 = w % 64
            # 保证为64的整数倍
            h1 = h - h1
            w1 = w - w1
            h1 = int(h1)
            w1 = int(w1)




            w1, h1 = 640, 448





            scale = w1 * 1.0 / orgin_width
            # print(w1, h1)
            left_img = left_img.resize((w1, h1), Image.ANTIALIAS)
            right_img = right_img.resize((w1, h1), Image.ANTIALIAS)

            dataL = Image.fromarray(dataL * scale)
            dataL = dataL.resize((w1, h1), Image.NEAREST)

        left_img = np.ascontiguousarray(left_img, dtype=np.float32)
        right_img = np.ascontiguousarray(right_img, dtype=np.float32)
        dataL = np.ascontiguousarray(dataL, dtype=np.float32)

        preprocess = get_transform()
        left_img = preprocess(left_img.copy())
        right_img = preprocess(right_img.copy())
        dataL = np.expand_dims(dataL.copy(), 0)
        return [left_img, right_img], dataL


if __name__ == '__main__':
    __spec__ = None

    from boxx import show
    from torch.utils.data import DataLoader

    mb_path = r"D:\pic\MiddV3\trainingH"
    mb_dataset = MbDataset(mb_path)
    dataloader = DataLoader(dataset=mb_dataset, batch_size=1, shuffle=True)
    for cnt, a in enumerate(dataloader, 1):  # show(a[0][0]), show(a[0][1]), show(a[1])
        aa = 1
