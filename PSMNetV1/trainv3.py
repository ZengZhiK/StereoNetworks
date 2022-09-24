# -*- coding: utf-8 -*-
"""
author:     tianhe
time  :     2022-09-12 11:45
"""
import time
import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torch.utils.tensorboard import SummaryWriter

from lmdbdatasets import LMDBDataset
from models import stackhourglassv3

torch.cuda.empty_cache()

# tensorboard
writer = SummaryWriter("./trainv3logs")
# 路径设置
lmdb_path = '/3d_recons/stereo_data/IRSDataset_lmdb'
trainlist = './filenames/stage1-train.txt'
testlist = './filenames/stage1-test.txt'
savemodelpath = './checkpointv3'

# 超参数
batch_size = 2
learning_rate = 1e-3
num_epochs = 50

num_workers = 4

print('batch_size: {}, learning_rate: {}, num_epochs: {}'.format(
    batch_size, learning_rate, num_epochs))

# 数据
lmdb_train_dataset = LMDBDataset(lmdb_path, trainlist, training=True)
lmdb_test_dataset = LMDBDataset(lmdb_path, testlist, training=False)
train_loader = DataLoader(dataset=lmdb_train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=lmdb_test_dataset, batch_size=batch_size, shuffle=False)

# 模型
model = stackhourglassv3(maxdisp=192).cuda()

# 优化器
optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))


# 损失
def compute_loss(predict, dispL, maxdisp=192):
    assert len(predict.shape) == 4  # [B 1 H W]
    assert len(dispL.shape) == 3  # [B H W]
    dispL = dispL.unsqueeze(1)  # [B H W] -> [B 1 H W]
    mask = (dispL < maxdisp).detach_()
    loss = F.smooth_l1_loss(predict[mask], dispL[mask], reduction='mean')
    return loss


# 训练
def train(imgL, imgR, dispL):
    model.train()
    imgL, imgR, dispL = imgL.cuda(), imgR.cuda(), dispL.cuda()  # [B C H W]

    predict1, predict2, predict3 = model(imgL, imgR)
    loss1 = compute_loss(predict1, dispL)
    loss2 = compute_loss(predict2, dispL)
    loss3 = compute_loss(predict3, dispL)
    loss = 0.5 * loss1 + 0.7 * loss2 + loss3

    # 每次计算梯度前，将上一次梯度置零
    optimizer.zero_grad()
    # 计算梯度
    loss.backward()
    # 更新权重
    optimizer.step()
    return loss.item(), predict3


# 测试
def test(imgL, imgR, dispL, downsampling=16, maxdisp=192):
    model.eval()
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
        predict = output  # [B H W]

    if len(dispL[mask]) == 0:
        loss = torch.tensor(0)
    else:
        loss = F.l1_loss(predict[mask], dispL[mask], reduction='mean')
    return loss.item(), predict


# 学习率调整
def adjust_learning_rate(lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    start_full_time = time.time()
    for epoch in range(num_epochs):
        # ---------------------------------- training ---------------------------------- #
        total_train_loss = 0
        for batch_idx, (imgL_crop, imgR_crop, dispL_crop) in enumerate(train_loader):
            start_time = time.time()
            loss, predict = train(imgL_crop, imgR_crop, dispL_crop)
            total_train_loss += loss
            if (batch_idx + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Time: {:.2f}'.format(
                    epoch + 1, num_epochs, batch_idx + 1, len(train_loader), loss, time.time() - start_time))
            if (batch_idx + 1) == 100:
                imgL_crop = torchvision.utils.make_grid(imgL_crop, nrow=4, padding=10, normalize=True)
                imgR_crop = torchvision.utils.make_grid(imgR_crop, nrow=4, padding=10, normalize=True)
                predict = torchvision.utils.make_grid(predict, nrow=4, padding=10, normalize=True)
                dispL_crop = torchvision.utils.make_grid(dispL_crop.unsqueeze(1), nrow=4, padding=10, normalize=True)
                writer.add_image('training imgL crop', imgL_crop, epoch + 1)
                writer.add_image('training imgR crop', imgR_crop, epoch + 1)
                writer.add_image('training predict', predict, epoch + 1)
                writer.add_image('training gt', dispL_crop, epoch + 1)
        print('Epoch {}, Mean Total Training Loss: {:.4f}'.format(epoch + 1, total_train_loss / len(train_loader)))
        writer.add_scalars('loss', {'train loss': total_train_loss / len(train_loader)}, epoch)
        writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], epoch)
        # -- save training model -- #
        savefilename = savemodelpath + '/checkpoint_' + str(epoch + 1) + '.tar'
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'train_loss': total_train_loss / len(train_loader),
        }, savefilename)
        # -- adjust learning rate -- #
        if epoch + 1 == 50:
            adjust_learning_rate(learning_rate / 2)
        if epoch + 1 == 80:
            adjust_learning_rate(learning_rate / 4)
        if epoch + 1 == 120:
            adjust_learning_rate(learning_rate / 8)
        if epoch + 1 == 170:
            adjust_learning_rate(learning_rate / 10)

        # ------------------------------------ test ------------------------------------ #
        total_test_loss = 0
        for batch_idx, (imgL, imgR, dispL) in enumerate(test_loader):
            loss, predict = test(imgL, imgR, dispL)
            total_test_loss += loss
            if (batch_idx + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Test Loss: {:.4f}'.format(
                    epoch + 1, num_epochs, batch_idx + 1, len(test_loader), loss))
            if (batch_idx + 1) == 100:
                imgL = torchvision.utils.make_grid(imgL, nrow=4, padding=10, normalize=True)
                imgR = torchvision.utils.make_grid(imgR, nrow=4, padding=10, normalize=True)
                error = torchvision.utils.make_grid(torch.abs(predict.cpu() - dispL).unsqueeze(1), nrow=4, padding=10,
                                                    normalize=True)
                predict = torchvision.utils.make_grid(predict.unsqueeze(1), nrow=4, padding=10, normalize=True)
                dispL = torchvision.utils.make_grid(dispL.unsqueeze(1), nrow=4, padding=10, normalize=True)
                writer.add_image('test imgL', imgL, epoch + 1)
                writer.add_image('test imgR', imgR, epoch + 1)
                writer.add_image('test predict', predict, epoch + 1)
                writer.add_image('test gt', dispL, epoch + 1)
                writer.add_image('test error', error, epoch + 1)
        print('Epoch {}, Mean Total Test Loss: {:.4f}'.format(epoch + 1, total_test_loss / len(test_loader)))
        writer.add_scalars('loss', {'test loss': total_test_loss / len(test_loader)}, epoch)
    print('Full Training Time: {:.2f} HR'.format((time.time() - start_full_time) / 3600))


if __name__ == '__main__':
    main()
