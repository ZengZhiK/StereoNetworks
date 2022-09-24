# -*- coding: UTF-8 -*-
import random
import numpy as np
import cv2


class RandomVdisp(object):
    """Random vertical disparity augmentation"""
    def __init__(self, angle, px, diff_angle=0, order=2, reshape=False):
        self.angle = angle  # 角度值
        self.reshape = reshape  
        self.order = order
        self.diff_angle = diff_angle
        self.px = px  # 偏移像素确定

    def __call__(self, input):
        # 输入的是numpy矩阵
        px2 = random.uniform(-self.px,self.px)  #正负偏移一定的像素
        angle2 = random.uniform(-self.angle,self.angle)  #正负偏移一定的角度
        # 得到右图图像中心
        image_center = (np.random.uniform(0,input.shape[0]), np.random.uniform(0,input.shape[1]))
        # 求解旋转矩阵
        rot_mat = cv2.getRotationMatrix2D(image_center, angle2, 1.0)
        # 仿射变换
        input = cv2.warpAffine(input, rot_mat, input.shape[1::-1], flags=cv2.INTER_LINEAR)
        # 平移
        trans_mat = np.float32([[1, 0, 0], [0, 1, px2]])
        input = cv2.warpAffine(input, trans_mat, input.shape[1::-1], flags=cv2.INTER_LINEAR)
        # print("y-disp")
        return input

