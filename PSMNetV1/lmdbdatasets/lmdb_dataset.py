# -*- coding: UTF-8 -*-
import random
import numpy as np
import cv2
import lmdb
import base64
from PIL import Image
from PIL import ImageFilter
from torch.utils.data import Dataset
import torchvision
from .data_io import get_transform, read_all_lines


class LMDBDataset(Dataset):
    def __init__(self, datapath, list_filename, training=True):
        self.datapath = datapath
        self.left_filenames, self.right_filenames, self.disp_filenames = self.load_path(list_filename)
        self.training = training
        env = lmdb.open(datapath, max_readers=32, readonly=True)
        self.txn = env.begin()

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]
        disp_images = [x[2] for x in splits]
        return left_images, right_images, disp_images

    def load_image1(self, filename):
        Key = '_'.join(((x) for x in (filename[:-4].split('/'))))
        tmp = self.txn.get(Key.encode())
        str_decode = base64.b64decode(tmp)
        nparr = np.frombuffer(str_decode, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = img[..., ::-1]  # BGR转RGB
        return img

    def load_disp1(self, filename):
        disparityKey = '_'.join(((x) for x in (filename[:-4].split('/'))))
        fmtKey = '_'.join(((x) for x in (filename[:-4].split('/')))) + '_fmt'
        fmt_tmp = self.txn.get(fmtKey.encode()).decode()
        Size = (int(fmt_tmp.split(',')[0][1:]), int(fmt_tmp.split(',')[1][:-1]))
        #    fmt,height,width = fmt_tmp[0],int(fmt_tmp[1]),int(fmt_tmp[2])
        buffer = self.txn.get(disparityKey.encode())
        str_decode = base64.b64decode(buffer)
        nparr = np.frombuffer(str_decode, np.uint8)
        dis_tmp = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        disparityleft = np.zeros((Size[1], Size[0]))
        disparityleft = (dis_tmp[:, :, 0] + 0.01 * dis_tmp[:, :, 1] + 0.0001 * dis_tmp[:, :, 2])
        return np.ascontiguousarray(disparityleft, dtype=np.float32)

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):
        left_img = self.load_image1(self.left_filenames[index])
        right_img = self.load_image1(self.right_filenames[index])
        disparity = self.load_disp1(self.disp_filenames[index])

        if self.training:
            # ##############################################数据增强###############################################
            left_img = Image.fromarray(left_img)
            right_img = Image.fromarray(right_img)
            # 彩色图转灰度图
            left_img = left_img.convert('L')
            right_img = right_img.convert('L')

            # # 右图y方向增强
            # if(np.random.randint(0,2,1) == 1):
            # angle = 0.1
            # px = 2.0
            # right_img = np.asarray(right_img)
            # y_proprecess = RandomVdisp(angle, px)
            # right_img = y_proprecess(right_img)
            # right_img = Image.fromarray(right_img)
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
                left_img = torchvision.transforms.functional.adjust_brightness(left_img, random_brightness[0])
                left_img = torchvision.transforms.functional.adjust_gamma(left_img, random_gamma[0])
                left_img = torchvision.transforms.functional.adjust_contrast(left_img, random_contrast[0])
                right_img = torchvision.transforms.functional.adjust_brightness(right_img, random_brightness[1])
                right_img = torchvision.transforms.functional.adjust_gamma(right_img, random_gamma[1])
                right_img = torchvision.transforms.functional.adjust_contrast(right_img, random_contrast[1])

            ###################高斯模糊###################
            if np.random.randint(0, 2, 1)[0] == 1:  # 50%几率随机使用高斯模糊
                radius = np.random.randint(0, 4, 1)[0]
                left_img = left_img.filter(ImageFilter.GaussianBlur(radius=radius))
                right_img = right_img.filter(ImageFilter.GaussianBlur(radius=radius))
                # print("test")
            ####################高斯模糊###################

            ###################尺度变换###################
            if np.random.randint(0, 2, 1)[0] == 1:  # 50%几率图像缩小1.875倍

                # if 1:
                left_img = left_img.resize((512, 288), Image.ANTIALIAS)
                right_img = right_img.resize((512, 288), Image.ANTIALIAS)
                disparity = disparity / 1.875
                disparity = Image.fromarray(disparity)
                # disparity = disparity.resize((512, 288), Image.ANTIALIAS)
                disparity = disparity.resize((512, 288), Image.NEAREST)
                disparity = np.ascontiguousarray(disparity, dtype=np.float32)
                # scale = np.random.uniform(1.0,1.875)
                # left_img = left_img.resize((960//scale, 540//scale),Image.ANTIALIAS)
                # right_img = right_img.resize((960//scale, 540//scale),Image.ANTIALIAS)
                # disparity = disparity.resize((960//scale, 540//scale),Image.ANTIALIAS) / scale
            ####################尺度变换###################

            ###################################################数据增强##################################################
            left_img = np.ascontiguousarray(left_img, dtype=np.float32)
            right_img = np.ascontiguousarray(right_img, dtype=np.float32)
            h, w = np.shape(left_img)
            crop_w, crop_h = 512, 256
            new_w = w
            new_h = h
            x1 = random.randint(0, new_w - crop_w)
            y1 = random.randint(0, new_h - crop_h)
            # 扩充维度
            left_img = np.expand_dims(left_img, -1)
            right_img = np.expand_dims(right_img, -1)
            # random crop
            left_img = left_img[y1:y1 + crop_h, x1:x1 + crop_w, :]
            right_img = right_img[y1:y1 + crop_h, x1:x1 + crop_w, :]
            disparity = disparity[y1:y1 + crop_h, x1:x1 + crop_w]

            # to tensor, normalize,仅做归一化处理
            preprocess = get_transform()
            left_img = preprocess(left_img.copy())
            right_img = preprocess(right_img.copy())
            # disparity = np.expand_dims(disparity.copy(), 0)
            disparity = disparity.copy()
            return left_img, right_img, disparity
        else:
            h, w, _ = np.shape(left_img)
            left_img = Image.fromarray(left_img)
            right_img = Image.fromarray(right_img)
            # 彩色图转灰度图
            left_img = left_img.convert('L')
            right_img = right_img.convert('L')
            # 扩大图片为32整数倍
            w1, h1 = 960, 576
            left_img = left_img.resize((w1, h1))
            right_img = right_img.resize((w1, h1))

            left_img = np.ascontiguousarray(left_img, dtype=np.float32)
            right_img = np.ascontiguousarray(right_img, dtype=np.float32)

            preprocess = get_transform()
            left_img = preprocess(left_img.copy())
            right_img = preprocess(right_img.copy())

            disparity = Image.fromarray(disparity)
            disparity = disparity.resize((w1, h1), Image.BILINEAR)
            disparity = np.ascontiguousarray(disparity, dtype=np.float32)
            # disparity = np.expand_dims(disparity.copy(), 0)
            disparity = disparity.copy()
            return left_img, right_img, disparity


if __name__ == '__main__':
    __spec__ = None

    from boxx import show
    from torch.utils.data import DataLoader

    lmdb_path = r'F:\3d_recons\stereo_data\IRSDataset_lmdb'
    # trainlist = '../filenames/stage1.txt'  # 不同stage改这里
    trainlist = '../filenames/stage1-train.txt'  # 不同stage改这里
    lmdb_train_dataset = LMDBDataset(lmdb_path, trainlist, training=True)
    dataloader = DataLoader(dataset=lmdb_train_dataset, batch_size=1, shuffle=True)
    for cnt, a in enumerate(dataloader, 1):  # show(a[0][0]), show(a[0][1]), show(a[1])
        aa = 1
