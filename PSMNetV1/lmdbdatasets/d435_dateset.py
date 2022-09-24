# -*- coding: UTF-8 -*-
import os
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

from .data_io import get_transform


def get_d435_list(filepath):
    left_list = []
    for entry in os.scandir(filepath):
        if 'left' in entry.path:
            left_list.append(entry.path)
    left_list = sorted(left_list)
    right_list = [img.replace('left', 'right') for img in left_list]
    disp_list = [img.replace('left', 'gt-disp') for img in left_list]
    return left_list, right_list, disp_list


class D435Dataset(Dataset):
    def __init__(self, data_dir):
        left_list, right_list, disp_list = get_d435_list(data_dir)
        self.left = left_list
        self.right = right_list
        self.disp = disp_list

    def load_image(self, filename):
        return Image.open(filename).convert('L')
        
    def load_disp(self, filename):
        data = Image.open(filename)
        data = np.array(data, dtype=np.float32) / 100.
        return data
    
    def __len__(self):
        return len(self.left)

    def __getitem__(self, index):
        left_img = self.load_image(self.left[index])
        right_img = self.load_image(self.right[index])
        if self.disp:
            disp_img = self.load_disp(self.disp[index])
            # disp_img = np.expand_dims(disp_img.copy(), 0)
            disp_img = Image.fromarray(disp_img)
        w, h = left_img.size
        h1 = h % 64
        w1 = w % 64
        # 保证为64的整数倍
        h1 = h - h1
        w1 = w - w1
        h1 = int(h1)
        w1 = int(w1)
        # BILINEAR
        left_img = left_img.resize((w1, h1), Image.ANTIALIAS)
        right_img = right_img.resize((w1, h1), Image.ANTIALIAS)
        disp_img = disp_img.resize((w1, h1), Image.NEAREST)  # * (w1 / w) 缩放要乘尺度，此处为1

        left_img = np.ascontiguousarray(left_img, dtype=np.float32)
        right_img = np.ascontiguousarray(right_img, dtype=np.float32)
        disp_img = np.ascontiguousarray(disp_img, dtype=np.float32)
        disp_img = np.expand_dims(disp_img.copy(), 0)

        preprocess = get_transform()
        left_img = preprocess(left_img.copy())
        right_img = preprocess(right_img.copy())
        if self.disp:
            return [left_img, right_img], disp_img
        else:
            return [left_img, right_img], left_img


if __name__ == '__main__':
    __spec__ = None
    from boxx import show
    from torch.utils.data import DataLoader

    d435_path = r"D:\pic\STAR_D435_passive_0103"
    d435_dataset = D435Dataset(d435_path)
    dataloader = DataLoader(dataset=d435_dataset, batch_size=1, shuffle=True)
    for cnt, a in enumerate(dataloader, 1):  # show(a[0][0]), show(a[0][1]), show(a[1])
        aa = 1
