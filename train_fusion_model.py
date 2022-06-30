# -*- coding: utf-8 -*-
"""
@Time ： 2022/6/21 18:00
@Auth ： zxc (https://github.com/linklist2)
@File ：train_fusion_model.py
@IDE ：PyCharm
@Function ：训练融合网络
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loader.msrs_data import MSRS_data
from models.cls_model import Illumination_classifier
from models.common import gradient, clamp
from models.fusion_model import PIAFusion


def init_seeds(seed=0):
    # Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html
    # cudnn seed 0 settings are slower and more reproducible, else faster and less reproducible
    import torch.backends.cudnn as cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)
    cudnn.benchmark, cudnn.deterministic = (False, True) if seed == 0 else (True, False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch PIAFusion')
    parser.add_argument('--dataset_path', metavar='DIR', default='datasets/msrs_train',
                        help='path to dataset (default: imagenet)')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='fusion_model',
                        choices=['fusion_model'])
    parser.add_argument('--save_path', default='pretrained')  # 模型存储路径
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=30, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch_size', default=128, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--image_size', default=64, type=int,
                        metavar='N', help='image size of input')
    parser.add_argument('--loss_weight', default='[3, 7, 50]', type=str,
                        metavar='N', help='loss weight')
    parser.add_argument('--cls_pretrained', default='pretrained/best_cls.pth',
                        help='use cls pre-trained model')
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--cuda', default=True, type=bool,
                        help='use GPU or not.')

    args = parser.parse_args()

    init_seeds(args.seed)

    train_dataset = MSRS_data(args.dataset_path)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # 如果是融合网络
    if args.arch == 'fusion_model':
        model = PIAFusion()
        model = model.cuda()

        # 加载预训练的分类模型
        # one-hot标签[白天概率，夜晚概率]
        cls_model = Illumination_classifier(input_channels=3)
        cls_model.load_state_dict(torch.load(args.cls_pretrained))
        cls_model = cls_model.cuda()
        cls_model.eval()

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(args.start_epoch, args.epochs):
            if epoch < args.epochs // 2:
                lr = args.lr
            else:
                lr = args.lr * (args.epochs - epoch) / (args.epochs - args.epochs // 2)

            # 修改学习率
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            model.train()
            train_tqdm = tqdm(train_loader, total=len(train_loader))
            for vis_image, vis_y_image, _, _, inf_image, _ in train_tqdm:
                vis_y_image = vis_y_image.cuda()
                vis_image = vis_image.cuda()
                inf_image = inf_image.cuda()
                optimizer.zero_grad()
                fused_image = model(vis_y_image, inf_image)
                # 强制约束范围在[0,1], 以免越界值使得最终生成的图像产生异常斑点
                fused_image = clamp(fused_image)

                # 使用预训练的分类模型，得到可见光图片属于白天还是夜晚的概率
                pred = cls_model(vis_image)
                day_p = pred[:, 0]
                night_p = pred[:, 1]
                vis_weight = day_p / (day_p + night_p)
                inf_weight = 1 - vis_weight

                # pixel l1_loss
                loss_illum = F.l1_loss(inf_weight[:, None, None, None] * fused_image,
                                       inf_weight[:, None, None, None] * inf_image) + F.l1_loss(
                    vis_weight[:, None, None, None] * fused_image,
                    vis_weight[:, None, None, None] * vis_y_image)

                # auxiliary intensity loss
                loss_aux = F.l1_loss(fused_image, torch.max(vis_y_image, inf_image))

                # gradient loss
                gradinet_loss = F.l1_loss(gradient(fused_image), torch.max(gradient(inf_image), gradient(vis_y_image)))
                t1, t2, t3 = eval(args.loss_weight)
                loss = t1 * loss_illum + t2 * loss_aux + t3 * gradinet_loss

                train_tqdm.set_postfix(epoch=epoch, loss_illum=t1 * loss_illum.item(), loss_aux=t2 * loss_aux.item(),
                                       gradinet_loss=t3 * gradinet_loss.item(),
                                       loss_total=loss.item())
                loss.backward()
                optimizer.step()

            torch.save(model.state_dict(), f'{args.save_path}/fusion_model_epoch_{epoch}.pth')
