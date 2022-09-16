# -*- coding: utf-8 -*-
"""
author:     tianhe
time  :     2022-09-16 15:48
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.dispnetsimplev2 import DispNetSimpleV2
from dataloader import listkitti2015 as lt
from dataloader import kittidataset as DS

# from boxx import show

# 路径设置
datapath = '/tianhe01/Datasets/KITTI2015/data_scene_flow/training/'
savemodelpath = './checkpoint1'

# 超参数
batch_size = 16
learning_rate = 1e-4
num_epochs = 100

use_cuda = True

num_workers = 4

train_left_img, train_right_img, train_left_disp, test_left_img, test_right_img, test_left_disp = lt.dataloader(
    datapath)
test_loader = DataLoader(
    DS.KITTIDataset(test_left_img, test_right_img, test_left_disp, training=False),
    batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

# 模型
model = DispNetSimpleV2()
if use_cuda:
    model.cuda()
state_dict = torch.load(savemodelpath + '/checkpoint_200.tar')
model.load_state_dict(state_dict['state_dict'])


# 损失
def compute_loss(predict, dispL, maxdisp=192):
    assert len(predict.shape) == 4  # [B 1 H W]
    assert len(dispL.shape) == 3  # [B H W]
    dispL = dispL.unsqueeze(1)  # [B H W] -> [B 1 H W]
    dispL = F.interpolate(dispL, (predict.shape[2], predict.shape[3]), mode='bilinear', align_corners=True)
    mask = (dispL < maxdisp).detach_()
    loss = F.l1_loss(predict[mask], dispL[mask], reduction='mean')
    return loss


# 测试
def test(imgL, imgR, dispL, downsampling=64, maxdisp=192):
    model.eval()
    if use_cuda:
        imgL, imgR, dispL = imgL.cuda(), imgR.cuda(), dispL.cuda()
    # 视差图掩膜
    mask = (dispL < maxdisp).detach_()
    # 维度调整
    if imgL.shape[2] % downsampling != 0:
        times = imgL.shape[2] // downsampling
        top_pad = (times + 1) * downsampling - imgL.shape[2]
    else:
        top_pad = 0

    if imgL.shape[3] % downsampling != 0:
        times = imgL.shape[3] // downsampling
        right_pad = (times + 1) * downsampling - imgL.shape[3]
    else:
        right_pad = 0

    imgL = F.pad(imgL, (0, right_pad, top_pad, 0))
    imgR = F.pad(imgR, (0, right_pad, top_pad, 0))

    with torch.no_grad():
        output = model(imgL, imgR)  # [B 1 H W]
        output = output.squeeze(1)  # [B H W]
    # 维度裁剪
    if top_pad != 0 and right_pad != 0:
        predict = output[:, top_pad:, :-right_pad]
    elif top_pad == 0 and right_pad != 0:
        predict = output[:, :, :-right_pad]
    elif top_pad != 0 and right_pad == 0:
        predict = output[:, top_pad:, :]
    else:
        predict = output

    if len(dispL[mask]) == 0:
        loss = torch.tensor(0)
    else:
        loss = F.l1_loss(predict[mask], dispL[mask], reduction='mean')
    return loss.item(), predict


def main():
    # ------------------------------------ test ------------------------------------ #
    total_test_loss = 0
    for batch_idx, (imgL, imgR, dispL) in enumerate(test_loader):
        loss, predict = test(imgL, imgR, dispL)
        total_test_loss += loss
        if (batch_idx + 1) % 1 == 0:
            print('Step [{}/{}], Test Loss: {:.4f}'.format(batch_idx + 1, len(test_loader), loss))
    print('Mean Total Test Loss: {:.4f}'.format(total_test_loss / len(test_loader)))


if __name__ == '__main__':
    main()
