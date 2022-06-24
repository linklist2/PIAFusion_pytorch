"""测试融合网络"""
import argparse
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from data_loader.msrs_data import MSRS_data
from models.common import YCrCb2RGB, RGB2YCrCb, clamp
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
    parser.add_argument('--dataset_path', metavar='DIR', default='test_data/MSRS',
                        help='path to dataset (default: imagenet)')  # 测试数据存放位置
    parser.add_argument('-a', '--arch', metavar='ARCH', default='fusion_model',
                        choices=['fusion_model'])
    parser.add_argument('--save_path', default='results/fusion')  # 融合结果存放位置
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--fusion_pretrained', default='pretrained/fusion_model_epoch_29.pth',
                        help='use cls pre-trained model')
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--cuda', default=True, type=bool,
                        help='use GPU or not.')

    args = parser.parse_args()

    init_seeds(args.seed)

    test_dataset = MSRS_data(args.dataset_path)
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # 如果是融合网络
    if args.arch == 'fusion_model':
        model = PIAFusion()
        model = model.cuda()
        model.load_state_dict(torch.load(args.fusion_pretrained))
        model.eval()
        test_tqdm = tqdm(test_loader, total=len(test_loader))
        for _, vis_y_image, cb, cr, inf_image, name in test_tqdm:
            vis_y_image = vis_y_image.cuda()
            cb = cb.cuda()
            cr = cr.cuda()
            inf_image = inf_image.cuda()

            # 测试转为Ycbcr的数据再转换回来的输出效果，结果与原图一样，说明这两个函数是没有问题的。
            # t = YCbCr2RGB2(vis_y_image[0], cb[0], cr[0])
            # transforms.ToPILImage()(t).save(name[0])

            fused_image = model(vis_y_image, inf_image)
            fused_image = clamp(fused_image)

            rgb_fused_image = YCrCb2RGB(fused_image[0], cb[0], cr[0])
            rgb_fused_image = transforms.ToPILImage()(rgb_fused_image)
            rgb_fused_image.save(f'{args.save_path}/{name[0]}')
