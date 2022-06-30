# -*- coding: utf-8 -*-
"""训练照明感知网络"""
import argparse
import os
import random

import numpy as np
from torch import optim
from torchvision import datasets, transforms
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.cls_model import Illumination_classifier


def test(model, test_loader):
    # test
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    prec1 = correct / float(len(test_loader.dataset))
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), 100. * prec1))
    return prec1


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
    parser.add_argument('--dataset_path', metavar='DIR', default='datasets/cls_dataset',
                        help='path to dataset (default: imagenet)')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='cls_model',
                        choices=['cls_model', 'fusion_model'])
    parser.add_argument('--save_path', default='pretrained')
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
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
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--image_size', default=64, type=int,
                        metavar='N', help='image size of input')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--cuda', default=True, type=bool,
                        help='use GPU or not.')

    args = parser.parse_args()

    init_seeds(args.seed)

    train_dataset = datasets.ImageFolder(
        args.dataset_path,
        transforms.Compose([
            transforms.ToTensor(),
        ]))

    # 划分验证集以测试模型性能， 训练与验证比例=9：1
    image_nums = len(train_dataset)
    train_nums = int(image_nums * 0.9)
    test_nums = image_nums - train_nums
    train_dataset, test_dataset = torch.utils.data.random_split(dataset=train_dataset, lengths=[train_nums, test_nums])

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # 如果是光照感知网络
    if args.arch == 'cls_model':
        model = Illumination_classifier(input_channels=3)
        model = model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best_prec1 = 0.0
        for epoch in range(args.start_epoch, args.epochs):
            # 自定义学习率衰减计划， 按照PIAFusion的代码，前一半epoch保持恒定学习率，后一半epoch学习率按照如下方式衰减
            if epoch < args.epochs // 2:
                lr = args.lr
            else:
                lr = args.lr * (args.epochs - epoch) / (args.epochs - args.epochs // 2)

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            model.train()
            train_tqdm = tqdm(train_loader, total=len(train_loader))
            # 白天one-hot label[1,0] ,夜晚label[0,1]
            for image, label in train_tqdm:
                image = image.cuda()
                label = label.cuda()
                optimizer.zero_grad()
                output = model(image)
                loss = F.cross_entropy(output, label)
                train_tqdm.set_postfix(epoch=epoch, loss_total=loss.item())
                loss.backward()
                optimizer.step()

            prec1 = test(model, test_loader)
            # 保存最佳模型权重
            if best_prec1 < prec1:
                torch.save(model.state_dict(), f'{args.save_path}/best_cls.pth')
                best_prec1 = prec1
