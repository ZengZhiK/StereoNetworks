import torch.utils.data as data
import torchvision
import random
from PIL import Image, ImageFilter
from . import preprocess
from . import readpfm as rp
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def default_loader(path):
    return Image.open(path).convert('RGB')


def disparity_loader(path):
    return rp.readPFM(path)


class SceneFlowDataset(data.Dataset):
    def __init__(self, left, right, left_disparity, training, loader=default_loader, dploader=disparity_loader):
        self.left = left
        self.right = right
        self.dispL = left_disparity
        self.loader = loader
        self.dploader = dploader
        self.training = training

    def __getitem__(self, index):
        left = self.left[index]
        right = self.right[index]
        dispL = self.dispL[index]

        left_img = self.loader(left)
        right_img = self.loader(right)
        dispL, scaleL = self.dploader(dispL)
        dispL = np.ascontiguousarray(dispL, dtype=np.float32)

        if self.training:
            # ---------------------------------------- 数据增强 ---------------------------------------- #
            # 50%的几率随机右图y方向增强
            if np.random.randint(0, 2, 1) == 1:
                angle = 0.1
                px = 2.0
                random_px = random.uniform(-px, px)  # 正负偏移一定的像素
                random_angle = random.uniform(-angle, angle)  # 正负偏移一定的角度
                right_img = right_img.rotate(random_angle, translate=(0, random_px))

            # 50%几率随机使用不对称颜色增强
            if np.random.randint(0, 2, 1) == 1:
                random_brightness = np.random.uniform(0.5, 1.5, 2)
                random_gamma = np.random.uniform(0.5, 1.5, 2)
                random_contrast = np.random.uniform(0.5, 1.5, 2)
                left_img = torchvision.transforms.functional.adjust_brightness(left_img, random_brightness[0])
                left_img = torchvision.transforms.functional.adjust_gamma(left_img, random_gamma[0])
                left_img = torchvision.transforms.functional.adjust_contrast(left_img, random_contrast[0])
                right_img = torchvision.transforms.functional.adjust_brightness(right_img, random_brightness[1])
                right_img = torchvision.transforms.functional.adjust_gamma(right_img, random_gamma[1])
                right_img = torchvision.transforms.functional.adjust_contrast(right_img, random_contrast[1])

            # 50%几率随机使用高斯模糊
            if np.random.randint(0, 2, 1) == 1:
                radius = np.random.randint(0, 4, 1)[0]
                left_img = left_img.filter(ImageFilter.GaussianBlur(radius=radius))
                right_img = right_img.filter(ImageFilter.GaussianBlur(radius=radius))

            # 50%几率图像缩小
            if np.random.randint(0, 2, 1) == 1:
                w, h = left_img.size
                tw, th = 832, 468
                left_img = left_img.resize((tw, th), Image.ANTIALIAS)
                right_img = right_img.resize((tw, th), Image.ANTIALIAS)
                dispL = dispL / (w / tw)
                dispL = Image.fromarray(dispL)
                dispL = dispL.resize((tw, th), Image.NEAREST)
                dispL = np.ascontiguousarray(dispL, dtype=np.float32)

            # 随机剪裁
            left_img = np.ascontiguousarray(left_img, dtype=np.float32)
            right_img = np.ascontiguousarray(right_img, dtype=np.float32)
            h, w, c = np.shape(left_img)
            tw, th = 768, 384
            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)
            left_img = left_img[y1:y1 + th, x1:x1 + tw, :]
            right_img = right_img[y1:y1 + th, x1:x1 + tw, :]
            dispL = dispL[y1:y1 + th, x1:x1 + tw]
            # normalize, to tensor仅做归一化处理
            processed = preprocess.get_transform(augment=False)
            left_img = processed(left_img.copy())
            right_img = processed(right_img.copy())
            dispL = dispL.copy()
            return left_img, right_img, dispL
        else:
            processed = preprocess.get_transform(augment=False)
            left_img = processed(left_img)
            right_img = processed(right_img)
            return left_img, right_img, dispL

    def __len__(self):
        return len(self.left)
