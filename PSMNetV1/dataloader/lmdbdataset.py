# -*- coding: utf-8 -*-
"""
author:     tianhe
time  :     2022-09-20 15:13
"""
# FlyingThings3D + 存存数据 50mm 的LMDB 适配4通道网络版
import random
import numpy as np
import cv2
import lmdb
import base64
from PIL import Image
from PIL import ImageFilter
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def read_all_lines(filename):
    with open(filename) as f:
        lines = [line.rstrip() for line in f.readlines()]
    return lines


def load_path(list_filename, idx=[0, 1, 2]):
    # 左图，右图，视差图
    lines = read_all_lines(list_filename)
    # lines=lines[:1]
    splits = [line.split() for line in lines]
    left_images = [x[idx[0]] for x in splits]
    right_images = [x[idx[1]] for x in splits]
    disp_images = [x[idx[2]] for x in splits]
    return left_images, right_images, disp_images


class LMDBDataset(Dataset):
    def __init__(self, datapath, list_filename, training=True, scale_factor=None, filename_idx=[0, 1, 2],
                 disp_scale=100, out_half_disp=False):
        self.datapath = datapath
        self.left_filenames, self.right_filenames, self.disp_filenames = load_path(list_filename, idx=filename_idx)
        self.training = training
        self.scale_factor = scale_factor  # resize的尺度因子
        self.disp_scale = disp_scale  # 视差图的放大系数
        self.out_half_disp = out_half_disp  # 输出视差图是否下采样一半

        env = lmdb.open(datapath, max_readers=32, readonly=True)
        self.txn = env.begin()

    def load_rgb(self, filename):
        Key = '_'.join((x for x in (filename[:-4].split('/'))))
        tmp = self.txn.get(Key.encode())
        str_decode = base64.b64decode(tmp)
        nparr = np.frombuffer(str_decode, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = img[..., ::-1]  # BGR转RGB
        return img

    def load_disp(self, filename):
        Key = '_'.join((x for x in filename[:-4].split('/')))
        tmp = self.txn.get(Key.encode())
        str_decode = base64.b64decode(tmp)
        nparr = np.frombuffer(str_decode, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_ANYDEPTH)
        return (img.astype('float32')) / self.disp_scale

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):
        left_img = self.load_rgb(self.left_filenames[index])  # show(left_img), show(right_img), show(disp)
        right_img = self.load_rgb(self.right_filenames[index])
        disp = self.load_disp(self.disp_filenames[index])

        left_img = Image.fromarray(left_img).convert('L')
        right_img = Image.fromarray(right_img).convert('L')

        w, h = left_img.size
        w_ori, h_ori = w, h

        if self.training:
            # ---------------------------------------- 数据增强 ---------------------------------------- #
            if np.random.randint(0, 2, 1) == 1:
                angle = 0.1
                px = 2.0
                px2 = random.uniform(-px, px)  # 正负偏移一定的像素
                angle2 = random.uniform(-angle, angle)  # 正负偏移一定的角度
                right_img = right_img.rotate(angle2, translate=[0, px2])

            # 颜色不对称增强
            if np.random.randint(0, 2, 1) == 1:  # 50%几率随机使用不对称颜色增强
                random_brightness = np.random.uniform(0.5, 1.5, 2)
                random_gamma = np.random.uniform(0.5, 1.5, 2)
                random_contrast = np.random.uniform(0.5, 1.5, 2)
                left_img = transforms.functional.adjust_brightness(left_img, random_brightness[0])
                left_img = transforms.functional.adjust_gamma(left_img, random_gamma[0])
                left_img = transforms.functional.adjust_contrast(left_img, random_contrast[0])
                right_img = transforms.functional.adjust_brightness(right_img, random_brightness[1])
                right_img = transforms.functional.adjust_gamma(right_img, random_gamma[1])
                right_img = transforms.functional.adjust_contrast(right_img, random_contrast[1])

            # 高斯模糊
            if np.random.randint(0, 2, 1)[0] == 1:  # 50%几率随机使用高斯模糊
                radius = np.random.randint(0, 4, 1)[0]
                left_img = left_img.filter(ImageFilter.GaussianBlur(radius=radius))
                right_img = right_img.filter(ImageFilter.GaussianBlur(radius=radius))

            # 尺度变换
            if np.random.randint(0, 2, 1)[0] == 1:  # 50%几率图像缩小
                w, h = 832, 468  # 640 360
                if self.scale_factor is not None:
                    if isinstance(self.scale_factor, (tuple, list)):  # 直接指定大小
                        w = self.scale_factor[0]
                        h = self.scale_factor[1]
                    else:
                        w = int(w_ori * self.scale_factor)
                        h = int(h_ori * self.scale_factor)

                left_img = left_img.resize((w, h), Image.ANTIALIAS)
                right_img = right_img.resize((w, h), Image.ANTIALIAS)
                scale = w / w_ori
                disp = Image.fromarray(disp * scale)
                disp = disp.resize((w, h), Image.NEAREST)
                disp = np.ascontiguousarray(disp, dtype=np.float32)

            # # 加噪
            # if np.random.randint(0, 2, 1)[0] == 1:
            #     left_img = np.ascontiguousarray(left_img, dtype=np.float32)
            #     right_img = np.ascontiguousarray(right_img, dtype=np.float32)
            #     stdv = np.random.uniform(0.0, 5.0)
            #     left_img = (left_img + stdv * np.random.randn(*left_img.shape)).clip(0.0, 255.0)
            #     right_img = (right_img + stdv * np.random.randn(*right_img.shape)).clip(0.0, 255.0)
            # ------------------------------------------------------------------------------------------ #
            left_img = np.ascontiguousarray(left_img, dtype=np.float32)
            right_img = np.ascontiguousarray(right_img, dtype=np.float32)
            # crop_w, crop_h = 768, 384  # 对应一半是384 192
            crop_w, crop_h = 512, 256  # 对应一半是384 192
            x1 = random.randint(0, w - crop_w)
            y1 = random.randint(0, h - crop_h)
            # 扩充维度
            left_img = np.expand_dims(left_img, -1)
            right_img = np.expand_dims(right_img, -1)
            # random crop
            left_img = left_img[y1:y1 + crop_h, x1:x1 + crop_w, :]
            right_img = right_img[y1:y1 + crop_h, x1:x1 + crop_w, :]
            disp = disp[y1:y1 + crop_h, x1:x1 + crop_w]

            if self.out_half_disp:
                disp = disp[::2, ::2] / 2

            left_img = transforms.ToTensor()(left_img.copy())
            right_img = transforms.ToTensor()(right_img.copy())
            disp = np.expand_dims(disp.copy(), 0)
            return [left_img, right_img], disp
        else:
            # resize为小分辨率
            if self.scale_factor is not None:
                if isinstance(self.scale_factor, (tuple, list)):  # 直接指定大小
                    w = self.scale_factor[0]
                    h = self.scale_factor[1]
                else:
                    w = int(w_ori * self.scale_factor)
                    h = int(h_ori * self.scale_factor)

            # resize为64的整数倍
            h = int(h - (h % 64))
            w = int(w - (w % 64))
            # print(w, h)
            scale = w / w_ori
            left_img = left_img.resize((w, h), Image.ANTIALIAS)
            right_img = right_img.resize((w, h), Image.ANTIALIAS)
            disp = Image.fromarray(disp * scale)
            disp = disp.resize((w, h), Image.NEAREST)

            left_img = np.ascontiguousarray(left_img, dtype=np.float32)
            right_img = np.ascontiguousarray(right_img, dtype=np.float32)
            disp = np.ascontiguousarray(disp, dtype=np.float32)

            if self.out_half_disp:
                disp = disp[::2, ::2] / 2

            left_img = transforms.ToTensor()(left_img.copy())
            right_img = transforms.ToTensor()(right_img.copy())
            disp = np.expand_dims(disp.copy(), 0)
            return [left_img, right_img], disp

    # def __getitem__(self, index):
    #     left_img = self.load_rgb(self.left_filenames[index])  # show(left_img), show(right_img), show(disp)
    #     right_img = self.load_rgb(self.right_filenames[index])
    #     disp = self.load_disp(self.disp_filenames[index])
    #
    #     left_img = Image.fromarray(left_img).convert('L')
    #     right_img = Image.fromarray(right_img).convert('L')
    #
    #     w, h = left_img.size
    #     w_ori, h_ori = w, h
    #
    #     if self.training:
    #         left_img = np.ascontiguousarray(left_img, dtype=np.float32)
    #         right_img = np.ascontiguousarray(right_img, dtype=np.float32)
    #         # crop_w, crop_h = 768, 384  # 对应一半是384 192
    #         crop_w, crop_h = 384, 192  # 对应一半是384 192
    #         x1 = 0
    #         y1 = 0
    #         # 扩充维度
    #         left_img = np.expand_dims(left_img, -1)
    #         right_img = np.expand_dims(right_img, -1)
    #         # random crop
    #         left_img = left_img[y1:y1 + crop_h, x1:x1 + crop_w, :]
    #         right_img = right_img[y1:y1 + crop_h, x1:x1 + crop_w, :]
    #         disp = disp[y1:y1 + crop_h, x1:x1 + crop_w]
    #
    #         left_img = transforms.ToTensor()(left_img.copy())
    #         right_img = transforms.ToTensor()(right_img.copy())
    #         disp = np.expand_dims(disp.copy(), 0)
    #         return [left_img, right_img], disp


if __name__ == '__main__':
    __spec__ = None

    from boxx import show
    from torch.utils.data import DataLoader

    # lmdb_path = r"I:\depth_calc\dataset\rl_data\rl100mm_flythings_lmdb"
    # trainlist = '../filenames/rl-lmdb-indoor-100mm.txt'

    # lmdb_train_dataset = LMDBNewDataset4chs(lmdb_path, trainlist, training=True, scale_factor=None)
    # # lmdb_train_dataset = LMDBNewDataset(lmdb_path, trainlist, training=True, scale_factor=0.5)
    # # lmdb_train_dataset = LMDBNewDataset(lmdb_path, trainlist, training=True, scale_factor=(640, 360))
    # dataloader = DataLoader(dataset=lmdb_train_dataset, batch_size=1, shuffle=False)
    # # dataloader = DataLoader(dataset=lmdb_train_dataset, batch_size=4, shuffle=True)
    # all = 0
    # for cnt, a in enumerate(dataloader, 1):  # show(a[0][0]), show(a[0][1]), show(a[1])
    #     # print(a[1].min().item(), a[1].max().item())
    #     print(a[1].max().item())
    #     all += a[1].max().item()
    #     aa = 1

    # # face data
    # lmdb_path = r"O:\3d_recons\stereo_data\spec_d435_lmdb"
    # # trainlist = '../filenames/filelist-face-d435_spec.txt'
    # trainlist = '../filenames/filelist-spec-d435-lmdb-train.txt'
    # lmdb_train_dataset = LMDBNewDataset4chs(lmdb_path, trainlist, training=True, scale_factor=None, filename_idx=[2, 3, 4])
    # dataloader = DataLoader(dataset=lmdb_train_dataset, batch_size=1, shuffle=False)
    # for cnt, a in enumerate(dataloader, 1):  # show(a[0][0]), show(a[0][1]), show(a[1])
    #     print(a[1].max().item())
    #     aa = 1

    # rl lmdb outdoor data
    lmdb_path = r'F:\3d_recons\stereo_data\IRSDataset_lmdb'
    trainlist = r'F:\3d_recons\stereo_data\IRSDataset_lmdb\filenames\speckle_valid_Flything3D_train_2221_new.txt'
    lmdb_train_dataset = LMDBDataset(lmdb_path, trainlist, training=True, scale_factor=None,
                                     filename_idx=[0, 1, 2], disp_scale=1, out_half_disp=False)
    dataloader = DataLoader(dataset=lmdb_train_dataset, batch_size=1, shuffle=True)
    for cnt, a in enumerate(dataloader, 1):  # show(a[0][0]), show(a[0][1]), show(a[1])
        print(a[1].max().item())
        aa = 1
